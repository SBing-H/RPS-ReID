import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking


def _feat_norm_enabled(value):
    if isinstance(value, str):
        return value.lower() in ("yes", "true", "1")
    return bool(value)


def _extract_features(cfg, weight_path, val_loader, num_classes, camera_num, view_num, device, tta_flip=False):
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(weight_path)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs for inference".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    feats = []
    pids = []
    camids = []
    with torch.no_grad():
        for img, pid, camid, camids_batch, target_view, _ in val_loader:
            img = img.to(device)
            camids_batch = camids_batch.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids_batch, view_label=target_view)
            if tta_flip:
                flip_img = torch.flip(img, dims=[3])
                flip_feat = model(flip_img, cam_label=camids_batch, view_label=target_view)
                feat = feat + flip_feat
            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))

    return torch.cat(feats, dim=0), np.asarray(pids), np.asarray(camids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransReID checkpoint ensemble test")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="checkpoint path; pass multiple --weight arguments to ensemble",
    )
    parser.add_argument(
        "--ensemble-weight",
        action="append",
        default=[],
        type=float,
        help="feature ensemble weight; pass once per --weight. Defaults to equal weights.",
    )
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        help="average original and horizontally flipped image features during inference",
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=("feature", "distance"),
        default="feature",
        help="feature averages embeddings before distance; distance averages per-checkpoint distance matrices",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, _, _ = make_dataloader(
        cfg, is_train=False
    )

    weight_paths = args.weight
    if not weight_paths and cfg.TEST.WEIGHT:
        weight_paths = [cfg.TEST.WEIGHT]
    if not weight_paths:
        raise RuntimeError("No checkpoint is provided. Use --weight path/to/checkpoint.pth")
    ensemble_weights = args.ensemble_weight
    if ensemble_weights and len(ensemble_weights) != len(weight_paths):
        raise RuntimeError("--ensemble-weight count must match --weight count.")
    if not ensemble_weights:
        ensemble_weights = [1.0] * len(weight_paths)
    weight_sum = sum(ensemble_weights)
    if weight_sum <= 0:
        raise RuntimeError("Ensemble weights must sum to a positive value.")
    ensemble_weights = [weight / weight_sum for weight in ensemble_weights]

    logger.info("Running checkpoint ensemble with {} weights".format(len(weight_paths)))
    logger.info("Ensemble mode: {}, tta_flip: {}".format(args.ensemble_mode, args.tta_flip))
    for weight_path, ensemble_weight in zip(weight_paths, ensemble_weights):
        logger.info("  weight {:.4f}: {}".format(ensemble_weight, weight_path))

    feature_sets = []
    pids_ref = None
    camids_ref = None
    normalize = _feat_norm_enabled(cfg.TEST.FEAT_NORM)

    for weight_path in weight_paths:
        feats, pids, camids = _extract_features(
            cfg,
            weight_path,
            val_loader,
            num_classes,
            camera_num,
            view_num,
            device,
            tta_flip=args.tta_flip,
        )
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        feature_sets.append(feats)

        if pids_ref is None:
            pids_ref = pids
            camids_ref = camids
        elif not (np.array_equal(pids_ref, pids) and np.array_equal(camids_ref, camids)):
            raise RuntimeError("Validation loader order changed between checkpoints.")

    q_pids = pids_ref[:num_query]
    g_pids = pids_ref[num_query:]
    q_camids = camids_ref[:num_query]
    g_camids = camids_ref[num_query:]

    if args.ensemble_mode == "distance":
        distmat = None
        for feats, ensemble_weight in zip(feature_sets, ensemble_weights):
            qf = feats[:num_query]
            gf = feats[num_query:]
            if cfg.TEST.RE_RANKING:
                print(
                    "=> Enter reranking: k1={}, k2={}, lambda={}".format(
                        cfg.TEST.RE_RANKING_K1,
                        cfg.TEST.RE_RANKING_K2,
                        cfg.TEST.RE_RANKING_LAMBDA,
                    )
                )
                current_distmat = re_ranking(
                    qf,
                    gf,
                    k1=cfg.TEST.RE_RANKING_K1,
                    k2=cfg.TEST.RE_RANKING_K2,
                    lambda_value=cfg.TEST.RE_RANKING_LAMBDA,
                )
            else:
                print("=> Computing DistMat with euclidean_distance")
                current_distmat = euclidean_distance(qf, gf)
            weighted_distmat = current_distmat * ensemble_weight
            distmat = weighted_distmat if distmat is None else distmat + weighted_distmat
    else:
        stacked_feats = torch.stack(feature_sets, dim=0)
        ensemble_weight_tensor = torch.tensor(ensemble_weights, dtype=stacked_feats.dtype).view(-1, 1, 1)
        feats = (stacked_feats * ensemble_weight_tensor).sum(dim=0)
        if normalize:
            print("The ensembled test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        qf = feats[:num_query]
        gf = feats[num_query:]

        if cfg.TEST.RE_RANKING:
            print(
                "=> Enter reranking: k1={}, k2={}, lambda={}".format(
                    cfg.TEST.RE_RANKING_K1,
                    cfg.TEST.RE_RANKING_K2,
                    cfg.TEST.RE_RANKING_LAMBDA,
                )
            )
            distmat = re_ranking(
                qf,
                gf,
                k1=cfg.TEST.RE_RANKING_K1,
                k2=cfg.TEST.RE_RANKING_K2,
                lambda_value=cfg.TEST.RE_RANKING_LAMBDA,
            )
        else:
            print("=> Computing DistMat with euclidean_distance")
            distmat = euclidean_distance(qf, gf)

    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
