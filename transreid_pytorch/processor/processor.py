import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss import dkd_loss, photometric_order_consistency_loss


def _build_best_metric(mAP, cmc):
    return (float(mAP), float(cmc[0]), float(cmc[4]), float(cmc[9]))


def _is_better_metric(current, best, eps=1e-12):
    for current_value, best_value in zip(current, best):
        if current_value > best_value + eps:
            return True
        if current_value < best_value - eps:
            return False
    return False


def _get_photometric_order_beta(model):
    base_model = model.module if hasattr(model, 'module') else model
    order_module = getattr(base_model, 'photometric_order', None)
    if order_module is None:
        return None
    return float(order_module.beta.detach().cpu())


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             aux_loss_fn,
             teacher_model,
             aux_loader,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if teacher_model is not None:
            teacher_model.to(local_rank)
            teacher_model.eval()
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    kd_loss_meter = AverageMeter()
    order_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    use_distill = cfg.DISTILL.ENABLED and teacher_model is not None and aux_loader is not None
    use_order_loss = bool(cfg.DISTILL.PHOTOMETRIC_ORDER_LOSS)
    if use_order_loss and not use_distill:
        raise RuntimeError(
            'Photometric-order consistency requires enabled paired distillation data.'
        )
    if use_order_loss and not cfg.MODEL.PHOTOMETRIC_ORDER:
        raise RuntimeError(
            'Photometric-order consistency requires MODEL.PHOTOMETRIC_ORDER=True.'
        )

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        reranking_k1=cfg.TEST.RE_RANKING_K1,
        reranking_k2=cfg.TEST.RE_RANKING_K2,
        reranking_lambda=cfg.TEST.RE_RANKING_LAMBDA,
    )
    scaler = amp.GradScaler()
    best_metric = (float('-inf'), float('-inf'), float('-inf'), float('-inf'))
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        main_loss_meter.reset()
        aux_loss_meter.reset()
        kd_loss_meter.reset()
        order_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        if teacher_model is not None:
            teacher_model.eval()
        aux_iter = iter(aux_loader) if use_distill else None
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            if use_distill:
                try:
                    aux_batch = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(aux_loader)
                    aux_batch = next(aux_iter)
                if use_order_loss:
                    if len(aux_batch) != 8:
                        raise RuntimeError(
                            'The paired loader did not return the aligned dark order view.'
                        )
                    dark_img, gt_img, aux_vid, aux_cam, aux_view, _, _, order_dark_img = aux_batch
                    order_dark_img = order_dark_img.to(device)
                else:
                    dark_img, gt_img, aux_vid, aux_cam, aux_view, _, _ = aux_batch
                dark_img = dark_img.to(device)
                gt_img = gt_img.to(device)
                aux_target = aux_vid.to(device)
                aux_cam = aux_cam.to(device)
                aux_view = aux_view.to(device)
                aux_cam_label = torch.zeros_like(aux_cam) if cfg.MODEL.SIE_CAMERA else aux_cam
                aux_view_label = torch.zeros_like(aux_view) if cfg.MODEL.SIE_VIEW else aux_view
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                main_loss = loss_fn(score, feat, target, target_cam)
                loss = main_loss
                aux_loss = None
                kd_loss_value = None
                order_loss_value = None
                if use_distill:
                    auxiliary_forward_kwargs = dict(
                        cam_label=aux_cam_label,
                        view_label=aux_view_label,
                        auxiliary=True,
                    )
                    # Only photometric-order models accept this optional pair.
                    # Plain single-path PAL training must not receive the keyword.
                    if use_order_loss:
                        auxiliary_forward_kwargs['photometric_order_pair'] = (
                            order_dark_img,
                            gt_img,
                        )
                    auxiliary_outputs = model(
                        dark_img,
                        aux_target,
                        **auxiliary_forward_kwargs,
                    )
                    if use_order_loss:
                        aux_score, aux_feat, order_outputs = auxiliary_outputs
                        predicted_order, target_order = order_outputs
                        order_loss_value = photometric_order_consistency_loss(
                            predicted_order,
                            target_order,
                            confidence_threshold=cfg.DISTILL.PHOTOMETRIC_ORDER_CONFIDENCE,
                        )
                    else:
                        aux_score, aux_feat = auxiliary_outputs
                    with torch.no_grad():
                        teacher_score, _ = teacher_model(
                            gt_img,
                            aux_target,
                            cam_label=aux_cam_label,
                            view_label=aux_view_label,
                            auxiliary=True,
                        )
                    aux_loss = aux_loss_fn(aux_score, aux_feat, aux_target, aux_cam)
                    kd_loss_value = dkd_loss(
                        aux_score,
                        teacher_score,
                        aux_target,
                        alpha=cfg.DISTILL.ALPHA,
                        beta=cfg.DISTILL.BETA,
                        temperature=cfg.DISTILL.TEMPERATURE,
                    )
                    loss = main_loss + cfg.DISTILL.AUX_LOSS_WEIGHT * aux_loss + cfg.DISTILL.LAMBDA_KD * kd_loss_value
                    if use_order_loss:
                        loss = loss + cfg.DISTILL.PHOTOMETRIC_ORDER_LOSS_WEIGHT * order_loss_value

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            main_loss_meter.update(main_loss.item(), img.shape[0])
            if use_distill:
                aux_loss_meter.update(aux_loss.item(), dark_img.shape[0])
                kd_loss_meter.update(kd_loss_value.item(), dark_img.shape[0])
                if use_order_loss:
                    order_loss_meter.update(order_loss_value.item(), dark_img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        if use_distill:
                            if use_order_loss:
                                logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Main: {:.3f}, Aux: {:.3f}, KD: {:.3f}, Order: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                            .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, main_loss_meter.avg, aux_loss_meter.avg, kd_loss_meter.avg, order_loss_meter.avg, acc_meter.avg, base_lr))
                            else:
                                logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Main: {:.3f}, Aux: {:.3f}, KD: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                            .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, main_loss_meter.avg, aux_loss_meter.avg, kd_loss_meter.avg, acc_meter.avg, base_lr))
                        else:
                            logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                        .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    if use_distill:
                        if use_order_loss:
                            logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Main: {:.3f}, Aux: {:.3f}, KD: {:.3f}, Order: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                        .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, main_loss_meter.avg, aux_loss_meter.avg, kd_loss_meter.avg, order_loss_meter.avg, acc_meter.avg, base_lr))
                        else:
                            logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Main: {:.3f}, Aux: {:.3f}, KD: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                        .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, main_loss_meter.avg, aux_loss_meter.avg, kd_loss_meter.avg, acc_meter.avg, base_lr))
                    else:
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))
        order_beta = _get_photometric_order_beta(model)
        if order_beta is not None and (
                not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0
        ):
            logger.info(
                "Photometric-order beta after epoch {}: {:.6f}".format(
                    epoch,
                    order_beta,
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    current_metric = _build_best_metric(mAP, cmc)
                    if _is_better_metric(current_metric, best_metric):
                        best_metric = current_metric
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                        logger.info("Best model saved with mAP: {:.1%}, Rank-1: {:.1%}, Rank-5: {:.1%}, Rank-10: {:.1%}"
                                    .format(best_metric[0], best_metric[1], best_metric[2], best_metric[3]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                current_metric = _build_best_metric(mAP, cmc)
                if _is_better_metric(current_metric, best_metric):
                    best_metric = current_metric
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                    logger.info("Best model saved with mAP: {:.1%}, Rank-1: {:.1%}, Rank-5: {:.1%}, Rank-10: {:.1%}"
                                .format(best_metric[0], best_metric[1], best_metric[2], best_metric[3]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        reranking_k1=cfg.TEST.RE_RANKING_K1,
        reranking_k2=cfg.TEST.RE_RANKING_K2,
        reranking_lambda=cfg.TEST.RE_RANKING_LAMBDA,
    )

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


