# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    local_loss_weight = float(cfg.MODEL.LOCAL_LOSS_WEIGHT)
    if local_loss_weight < 0.0 or local_loss_weight > 1.0:
        raise ValueError('MODEL.LOCAL_LOSS_WEIGHT must be in [0, 1].')
    global_loss_weight = 1.0 - local_loss_weight

    def combine_global_local_loss(global_loss, local_losses):
        local_loss = sum(local_losses) / len(local_losses)
        return global_loss_weight * global_loss + local_loss_weight * local_loss

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target,target_cam):
            return F.cross_entropy(score, target)

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        local_id_losses = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = combine_global_local_loss(xent(score[0], target), local_id_losses)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        local_tri_losses = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = combine_global_local_loss(triplet(feat[0], target)[0], local_tri_losses)
                    else:
                        TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        local_id_losses = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = combine_global_local_loss(F.cross_entropy(score[0], target), local_id_losses)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        local_tri_losses = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = combine_global_local_loss(triplet(feat[0], target)[0], local_tri_losses)
                    else:
                        TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


