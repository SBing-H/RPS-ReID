from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import torch.distributed as dist


def load_teacher_weight(model, weight_path, logger):
    if not weight_path:
        raise RuntimeError('DISTILL.TEACHER_WEIGHT must be set when DISTILL.ENABLED is True.')
    if not os.path.exists(weight_path):
        raise RuntimeError('Teacher weight does not exist: {}'.format(weight_path))

    checkpoint = torch.load(weight_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model_state = model.state_dict()
    loaded_state = {}
    skipped = []
    aux_loaded = False
    for key, value in checkpoint.items():
        clean_key = key.replace('module.', '')
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            loaded_state[clean_key] = value
            if clean_key.startswith('aux_classifier.'):
                aux_loaded = True
        elif clean_key == 'classifier.weight' and 'aux_classifier.weight' in model_state:
            if model_state['aux_classifier.weight'].shape == value.shape:
                loaded_state['aux_classifier.weight'] = value
                aux_loaded = True
            else:
                skipped.append(clean_key)
        else:
            skipped.append(clean_key)

    model.load_state_dict(loaded_state, strict=False)
    logger.info('Loaded {} teacher tensors from {}'.format(len(loaded_state), weight_path))
    if skipped:
        logger.info('Skipped {} incompatible teacher tensors'.format(len(skipped)))
    if not aux_loaded:
        logger.warning('Teacher aux_classifier was not loaded; DKD logits may be random.')


def load_finetune_weight(model, weight_path, logger):
    if not weight_path:
        raise RuntimeError("MODEL.PRETRAIN_PATH must be set when MODEL.PRETRAIN_CHOICE is 'finetune'.")
    if not os.path.exists(weight_path):
        raise RuntimeError('Finetune weight does not exist: {}'.format(weight_path))
    model.load_param(weight_path)
    logger.info('Loaded finetune weight from {}'.format(weight_path))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    #  logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Running with config:\n{}".format(cfg))


    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, aux_loader, aux_num_classes = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, aux_num_class=aux_num_classes)
    if cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        load_finetune_weight(model, cfg.MODEL.PRETRAIN_PATH, logger)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    aux_loss_func = None
    teacher_model = None
    if cfg.DISTILL.ENABLED:
        if aux_loader is None or aux_num_classes <= 0:
            raise RuntimeError('DISTILL.ENABLED is True, but no auxiliary SRD data was loaded.')
        teacher_model = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            aux_num_class=aux_num_classes,
            is_teacher=True,
        )
        load_teacher_weight(teacher_model, cfg.DISTILL.TEACHER_WEIGHT, logger)
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        teacher_model.eval()
        aux_loss_func, _ = make_loss(cfg, num_classes=aux_num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        aux_loss_func,
        teacher_model,
        aux_loader,
        num_query, args.local_rank
    )
    #  print(cfg.OUTPUT_DIR)
    #  print(cfg.MODEL.PRETRAIN_PATH)
