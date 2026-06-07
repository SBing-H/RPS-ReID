import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def get_transformer_extra_drop_path_kwargs(cfg):
    return dict(
        extra_drop_path=cfg.MODEL.EXTRA_DROP_PATH,
        extra_drop_path_layers=cfg.MODEL.EXTRA_DROP_PATH_LAYERS,
    )


def get_transformer_ngtse_kwargs(cfg, enabled=None):
    use_ngtse = cfg.MODEL.NGTSE if enabled is None else enabled
    return dict(
        ngtse=use_ngtse,
        ngtse_mode=cfg.MODEL.NGTSE_MODE,
        ngtse_layers=cfg.MODEL.NGTSE_LAYERS,
        ngtse_kernel=cfg.MODEL.NGTSE_KERNEL,
        ngtse_beta=cfg.MODEL.NGTSE_BETA,
        ngtse_gate_type=cfg.MODEL.NGTSE_GATE_TYPE,
        ngtse_detach_gate=cfg.MODEL.NGTSE_DETACH_GATE,
        ngtse_residual_gate=cfg.MODEL.NGTSE_RESIDUAL_GATE,
        ngtse_weighted_structure=cfg.MODEL.NGTSE_WEIGHTED_STRUCTURE,
        ngtse_attention_gate_temp=cfg.MODEL.NGTSE_ATTENTION_GATE_TEMP,
    )


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg, aux_num_classes=0):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.aux_classifier = None
        if aux_num_classes > 0:
            self.aux_classifier = nn.Linear(self.in_planes, aux_num_classes, bias=False)
            self.aux_classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)


    def forward(self, x, label=None, auxiliary=False, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.dropout_rate > 0:
            feat = self.dropout(feat)

        if auxiliary:
            if self.aux_classifier is None:
                raise RuntimeError('Auxiliary classifier is not initialized.')
            return self.aux_classifier(feat), global_feat

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            elif 'module' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    #  def load_param(self, trained_path):
        #  param_dict = torch.load(trained_path, map_location = 'cpu')
        #  for i in param_dict:
            #  try:
                #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            #  except:
                #  continue
        #  print('Loading pretrained model from {}'.format(trained_path))


class build_transformer(nn.Module):
    def __init__(
            self,
            num_classes,
            camera_num,
            view_num,
            cfg,
            factory,
            aux_num_classes=0,
            ngtse_enabled=None,
            local_feature=False,
    ):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=local_feature, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV, **get_transformer_extra_drop_path_kwargs(cfg), **get_transformer_ngtse_kwargs(cfg, ngtse_enabled))
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.aux_classifier = None
        if aux_num_classes > 0:
            self.aux_classifier = nn.Linear(self.in_planes, aux_num_classes, bias=False)
            self.aux_classifier.apply(weights_init_classifier)

        self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None, auxiliary=False):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        if auxiliary:
            if self.aux_classifier is None:
                raise RuntimeError('Auxiliary classifier is not initialized.')
            return self.aux_classifier(feat_cls), global_feat

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange, aux_num_classes=0, ngtse_enabled=None):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, **get_transformer_extra_drop_path_kwargs(cfg), **get_transformer_ngtse_kwargs(cfg, ngtse_enabled))
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.aux_classifier = None
        if aux_num_classes > 0:
            self.aux_classifier = nn.Linear(self.in_planes, aux_num_classes, bias=False)
            self.aux_classifier.apply(weights_init_classifier)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None, auxiliary=False):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        if auxiliary:
            if self.aux_classifier is None:
                raise RuntimeError('Auxiliary classifier is not initialized.')
            return self.aux_classifier(feat), global_feat

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            key = i.replace('module.', '')
            if key in self.state_dict() and self.state_dict()[key].shape == param_dict[i].shape:
                self.state_dict()[key].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


class DualBranchTransformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, aux_num_classes=0):
        super(DualBranchTransformer, self).__init__()
        if cfg.MODEL.JPM:
            raise RuntimeError('DualBranchTransformer currently supports JPM=False only.')
        self.dual_local = cfg.MODEL.DUAL_LOCAL

        self.raw_branch = build_transformer(
            num_classes,
            camera_num,
            view_num,
            cfg,
            factory,
            aux_num_classes=0,
            ngtse_enabled=False,
            local_feature=self.dual_local,
        )
        self.struct_branch = build_transformer(
            num_classes,
            camera_num,
            view_num,
            cfg,
            factory,
            aux_num_classes=0,
            ngtse_enabled=cfg.MODEL.NGTSE,
            local_feature=self.dual_local,
        )
        self._freeze_unused_branch_heads(self.raw_branch)
        self._freeze_unused_branch_heads(self.struct_branch)
        if self.dual_local:
            self.raw_global_block, self.raw_local_block = self._build_local_blocks(self.raw_branch)
            self.struct_global_block, self.struct_local_block = self._build_local_blocks(self.struct_branch)
            self.divide_length = cfg.MODEL.DEVIDE_LENGTH
            self.shift_num = cfg.MODEL.SHIFT_NUM
            self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
            self.rearrange = cfg.MODEL.RE_ARRANGE

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.in_planes = self.raw_branch.in_planes

        self.fusion = nn.Linear(self.in_planes * 2, self.in_planes, bias=False)
        self.fusion.apply(weights_init_xavier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.ID_LOSS_TYPE == 'arcface':
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            if self.dual_local:
                self.local_classifiers = nn.ModuleList([
                    nn.Linear(self.in_planes, self.num_classes, bias=False)
                    for _ in range(self.divide_length)
                ])
                for classifier in self.local_classifiers:
                    classifier.apply(weights_init_classifier)

        self.aux_classifier = None
        if aux_num_classes > 0:
            self.aux_classifier = nn.Linear(self.in_planes, aux_num_classes, bias=False)
            self.aux_classifier.apply(weights_init_classifier)

        if self.dual_local:
            self.local_fusion = nn.Linear(self.in_planes * 2, self.in_planes, bias=False)
            self.local_fusion.apply(weights_init_xavier)
            self.local_bottlenecks = nn.ModuleList([
                nn.BatchNorm1d(self.in_planes)
                for _ in range(self.divide_length)
            ])
            for bottleneck in self.local_bottlenecks:
                bottleneck.bias.requires_grad_(False)
                bottleneck.apply(weights_init_kaiming)

    @staticmethod
    def _freeze_unused_branch_heads(branch):
        for name in ('classifier', 'bottleneck', 'aux_classifier'):
            module = getattr(branch, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)

    @staticmethod
    def _build_local_blocks(branch):
        block = branch.base.blocks[-1]
        layer_norm = branch.base.norm
        global_block = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        local_block = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        DualBranchTransformer._disable_ngtse(local_block)
        return global_block, local_block

    @staticmethod
    def _disable_ngtse(module):
        for submodule in module.modules():
            if hasattr(submodule, 'use_ngtse'):
                submodule.use_ngtse = False
            if hasattr(submodule, 'ngtse'):
                submodule.ngtse = None

    @staticmethod
    def _extract_branch_feat(branch, x, cam_label=None, view_label=None):
        global_feat = branch.base(x, cam_label=cam_label, view_label=view_label)
        if branch.reduce_feat_dim:
            global_feat = branch.fcneck(global_feat)
        return global_feat

    def _extract_branch_local_feats(self, branch, global_block, local_block, x, cam_label=None, view_label=None):
        features = branch.base(x, cam_label=cam_label, view_label=view_label)
        global_feat = global_block(features)[:, 0]

        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]
        if self.rearrange:
            patches = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            patches = features[:, 1:]

        local_feats = []
        for part_idx in range(self.divide_length):
            start = patch_length * part_idx
            end = patch_length * (part_idx + 1)
            part_tokens = patches[:, start:end]
            part_feat = local_block(torch.cat((token, part_tokens), dim=1))[:, 0]
            local_feats.append(part_feat)
        return global_feat, local_feats

    def forward(self, x, label=None, cam_label=None, view_label=None, auxiliary=False):
        if self.dual_local:
            raw_feat, raw_local_feats = self._extract_branch_local_feats(
                self.raw_branch,
                self.raw_global_block,
                self.raw_local_block,
                x,
                cam_label=cam_label,
                view_label=view_label,
            )
            struct_feat, struct_local_feats = self._extract_branch_local_feats(
                self.struct_branch,
                self.struct_global_block,
                self.struct_local_block,
                x,
                cam_label=cam_label,
                view_label=view_label,
            )
        else:
            raw_feat = self._extract_branch_feat(self.raw_branch, x, cam_label=cam_label, view_label=view_label)
            struct_feat = self._extract_branch_feat(self.struct_branch, x, cam_label=cam_label, view_label=view_label)
        global_feat = self.fusion(torch.cat([raw_feat, struct_feat], dim=1))
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        if auxiliary:
            if self.aux_classifier is None:
                raise RuntimeError('Auxiliary classifier is not initialized.')
            return self.aux_classifier(feat_cls), global_feat

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)
            if self.dual_local:
                local_feats = [
                    self.local_fusion(torch.cat([raw_local_feat, struct_local_feat], dim=1))
                    for raw_local_feat, struct_local_feat in zip(raw_local_feats, struct_local_feats)
                ]
                local_bn_feats = [
                    bottleneck(local_feat)
                    for bottleneck, local_feat in zip(self.local_bottlenecks, local_feats)
                ]
                local_scores = [
                    classifier(self.dropout(local_bn_feat))
                    for classifier, local_bn_feat in zip(self.local_classifiers, local_bn_feats)
                ]
                return [cls_score] + local_scores, [global_feat] + local_feats
            return cls_score, global_feat

        if self.dual_local:
            local_feats = [
                self.local_fusion(torch.cat([raw_local_feat, struct_local_feat], dim=1))
                for raw_local_feat, struct_local_feat in zip(raw_local_feats, struct_local_feats)
            ]
            local_bn_feats = [
                bottleneck(local_feat)
                for bottleneck, local_feat in zip(self.local_bottlenecks, local_feats)
            ]
            if self.neck_feat == 'after':
                return torch.cat([feat] + [local_bn_feat / self.divide_length for local_bn_feat in local_bn_feats], dim=1)
            return torch.cat([global_feat] + [local_feat / self.divide_length for local_feat in local_feats], dim=1)

        if self.neck_feat == 'after':
            return feat
        return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        own_state = self.state_dict()
        for key, value in param_dict.items():
            clean_key = key.replace('module.', '')
            if clean_key in own_state and own_state[clean_key].shape == value.shape:
                own_state[clean_key].copy_(value)
        if self.dual_local:
            self._load_local_block_param(param_dict, own_state)
        print('Loading pretrained model from {}'.format(trained_path))

    def _load_local_block_param(self, param_dict, own_state):
        raw_last_idx = len(self.raw_branch.base.blocks) - 1
        struct_last_idx = len(self.struct_branch.base.blocks) - 1
        mappings = (
            ('raw_branch.base.blocks.{}.'.format(raw_last_idx), 'raw_global_block.0.'),
            ('raw_branch.base.blocks.{}.'.format(raw_last_idx), 'raw_local_block.0.'),
            ('raw_branch.base.norm.', 'raw_global_block.1.'),
            ('raw_branch.base.norm.', 'raw_local_block.1.'),
            ('struct_branch.base.blocks.{}.'.format(struct_last_idx), 'struct_global_block.0.'),
            ('struct_branch.base.blocks.{}.'.format(struct_last_idx), 'struct_local_block.0.'),
            ('struct_branch.base.norm.', 'struct_global_block.1.'),
            ('struct_branch.base.norm.', 'struct_local_block.1.'),
        )
        for key, value in param_dict.items():
            clean_key = key.replace('module.', '')
            for source_prefix, target_prefix in mappings:
                if clean_key.startswith(source_prefix):
                    target_key = target_prefix + clean_key[len(source_prefix):]
                    if target_key in own_state and own_state[target_key].shape == value.shape:
                        own_state[target_key].copy_(value)



__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
}

def make_model(cfg, num_class, camera_num, view_num, aux_num_class=0, is_teacher=False):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.DUAL_BRANCH and not is_teacher:
            model = DualBranchTransformer(num_class, camera_num, view_num, cfg, __factory_T_type, aux_num_classes=aux_num_class)
            print('===========building dual-branch transformer student===========')
        elif cfg.MODEL.JPM:
            model = build_transformer_local(
                num_class,
                camera_num,
                view_num,
                cfg,
                __factory_T_type,
                rearrange=cfg.MODEL.RE_ARRANGE,
                aux_num_classes=aux_num_class,
                ngtse_enabled=False if is_teacher else None,
            )
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(
                num_class,
                camera_num,
                view_num,
                cfg,
                __factory_T_type,
                aux_num_classes=aux_num_class,
                ngtse_enabled=False if is_teacher else None,
            )
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg, aux_num_classes=aux_num_class)
        print('===========building ResNet===========')
    return model
