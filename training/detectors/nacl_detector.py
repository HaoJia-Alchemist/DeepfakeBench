'''
# author: Hao Jia
# email: 10431220367@stu.qlu.edu.cn
# date: 2023-0821
# description: Class for the Dual Stream Multi-scala No-local consistency(DSMSNC)

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation
'''

import os
import datetime
import logging
import math
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict
from torch.nn.modules.utils import _pair, _quadruple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from .efficientnet import EfficientNet

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='nacl')
class NaClDetector(AbstractDetector):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_func = self.build_loss(cfg)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        temp_rgb = torch.randn(2, 3, cfg['resolution'], cfg['resolution'])
        # Backbone
        if 'efficientnet' in cfg['model']['backbone']:
            self.rgb_backbone = EfficientNet.from_pretrained(cfg['model']['backbone'], advprop=True,
                                                             num_classes=cfg['model']['num_classes'])
        else:
            raise ValueError("Unsupported Backbone!")
        self.self_enhancement = nn.ModuleDict()
        self.nl_consistency = nn.ModuleDict()
        temp_layers = {}
        start_stage = ""
        temp_layers[''] = temp_rgb
        temp_nlc = {}
        self.relu = nn.ReLU(inplace=False)
        # Top-down b0 to b7
        for feature_layer, patch_size in zip(cfg['model']['multi_scale'], cfg['model']['patch_size']):
            end_stage = feature_layer
            self.backbone_forward(temp_layers[start_stage], temp_layers, start_stage, end_stage)
            self.self_enhancement[end_stage] = SelfEnhancement(temp_layers[end_stage].shape[1])
            temp_layers[end_stage] = self.self_enhancement[end_stage](temp_layers[end_stage])
            # update stage flag
            start_stage = end_stage

        # multi-scale and non-local consistency and fusion module
        if cfg['model']['multi_scale'] and cfg['model']['non_local_consistency'] and cfg['model']['fusion_module']:
            # Bottom-up
            self.attention = nn.ModuleDict()
            last_stage = ""
            for feature_layer, patch_size in zip(cfg['model']['multi_scale'][::-1], cfg['model']['patch_size'][::-1]):
                if last_stage != "":  # Fusion Module
                    B, C, W, H = temp_layers[feature_layer].shape
                    patch_w = W // patch_size
                    up_sample_attention = temp_nlc[last_stage].view(temp_nlc[last_stage].shape[0],
                                                                    temp_nlc[last_stage].shape[1], patch_w, patch_w)
                    self.attention[feature_layer] = AttentionModule(up_sample_attention.shape[1],
                                                                    cfg['model']['attention_num'])
                    up_sample_attention = self.attention[feature_layer](up_sample_attention)
                    up_sample_attention = nn.functional.interpolate(up_sample_attention,
                                                                    scale_factor=patch_size,
                                                                    mode='bilinear')
                    temp_layers[feature_layer] = torch.einsum("bcwh, bdwh -> bcdwh", temp_layers[feature_layer],
                                                              up_sample_attention)
                    temp_layers[feature_layer] = temp_layers[feature_layer].view(B, -1, W, H)

                dcam = DCAM(cfg, temp_layers[feature_layer].shape[1])
                temp_dcamed = dcam((temp_layers[feature_layer], temp_layers[feature_layer]))
                self.nl_consistency[feature_layer] = nn.Sequential(
                    dcam,
                    NLConsistency(temp_dcamed.shape[2], temp_dcamed.shape[1], patch_size)
                )
                temp_nlc[feature_layer] = self.nl_consistency[feature_layer](
                    (temp_layers[feature_layer], temp_layers[feature_layer])
                )
                last_stage = feature_layer
            temp_out = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in temp_nlc.values()]
        # multi-scale and non-local consistency
        elif cfg['model']['multi_scale'] and cfg['model']['non_local_consistency']:
            # Bottom-up
            self.attention = nn.ModuleDict()
            for feature_layer, patch_size in zip(cfg['model']['multi_scale'][::-1], cfg['model']['patch_size'][::-1]):
                dcam = DCAM(cfg, temp_layers[feature_layer].shape[1])
                temp_dcamed = dcam((temp_layers[feature_layer], temp_layers[feature_layer]))
                self.nl_consistency[feature_layer] = nn.Sequential(
                    dcam,
                    NLConsistency(temp_dcamed.shape[2], temp_dcamed.shape[1], patch_size)
                )
                temp_nlc[feature_layer] = self.nl_consistency[feature_layer](
                    (temp_layers[feature_layer], temp_layers[feature_layer])
                )
            temp_out = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in temp_nlc.values()]
        # multi-scale
        elif cfg['model']['multi_scale']:
            temp_out = []
            for feature_layer in cfg['model']['multi_scale'][::-1]:
                temp_out.append(temp_layers[feature_layer])
        # non-local consistency
        elif cfg['model']['non_local_consistency']:
            self.nl_consistency = nn.ModuleDict()
            temp_nlc = {}
            dcam = DCAM(cfg, temp_layers['b7'].shape[1])
            temp_dcamed = dcam((temp_layers['b7'], temp_layers['b7']))
            self.nl_consistency['b7'] = nn.Sequential(
                dcam,
                NLConsistency(temp_dcamed.shape[2], temp_dcamed.shape[1], 12)
            )
            temp_nlc['b7'] = self.nl_consistency['b7'](
                (temp_layers['b7'], temp_layers['b7'])
            )
            temp_out = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in temp_nlc.values()]
        else:
            temp_out = []
        # b7 to final
        self.backbone_forward(temp_layers[start_stage], temp_layers, start_stage, end_stage='b7', final=True)
        temp_out.append(temp_layers['final'])
        temp_out = self._norm_feature(temp_out)
        self.fc = nn.Linear(temp_out.shape[1], 2)
        self.ensemble_classifier_fc = nn.Sequential(nn.Dropout(p=cfg['model']['drop_rate']),
                                                    nn.Linear(temp_out.shape[1], cfg['model']['mid_dim']),
                                                    nn.Hardswish(),
                                                    nn.Linear(cfg['model']['mid_dim'], cfg['model']['num_classes']))


    def build_backbone(self, config):
        pass

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def _norm_feature(self, x):
        x = torch.concat([F.adaptive_avg_pool2d(self.relu(i), (1, 1)) for i in x], dim=1)
        x = x.view(x.size(0), -1)
        return x

    def backbone_forward(self, x, layers, start_stage='', end_stage='b7', final=False):
        """ Returns output of the final convolution layer """
        if start_stage is '':
            # Stem
            x = self.rgb_backbone._swish(self.rgb_backbone._bn0(self.rgb_backbone._conv_stem(x)))
            layers['b0'] = x
        stage_idx = (self.rgb_backbone.stage_map.index(start_stage),
                     self.rgb_backbone.stage_map.index(end_stage))
        # Blocks
        for idx, block in enumerate(self.rgb_backbone._blocks):
            if (idx <= stage_idx[0] and start_stage != '') or idx > stage_idx[1]:
                continue
            drop_connect_rate = self.rgb_backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.rgb_backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            stage = self.rgb_backbone.stage_map[idx]
            if stage:
                layers[stage] = x
                if stage == self.rgb_backbone.escape:
                    return None
        if final:
            # Head
            x = self.rgb_backbone._bn1(self.rgb_backbone._conv_head(x))
            x = self.rgb_backbone._swish(x)
            layers['final'] = x
        return x

    def features(self, data_dict: dict) -> torch.tensor:
        layers = {}
        nlc = {}
        layers[''] = data_dict['image']
        start_stage = ""
        # Top-down
        for feature_layer, patch_size in zip(self.cfg['model']['multi_scale'], self.cfg['model']['patch_size']):
            end_stage = feature_layer
            self.backbone_forward(layers[start_stage], layers, start_stage, end_stage)
            layers[end_stage] = self.self_enhancement[end_stage](layers[end_stage])
            # update stage flag
            start_stage = end_stage
        # multi-scale and non-local consistency and fusion module
        if self.cfg['model']['multi_scale'] and self.cfg['model']['non_local_consistency'] and self.cfg['model']['fusion_module']:
            last_stage = ""
            for feature_layer, patch_size in zip(self.cfg['model']['multi_scale'][::-1], self.cfg['model']['patch_size'][::-1]):
                if last_stage != "":  # Fusion Module
                    B, C, W, H = layers[feature_layer].shape
                    patch_w = W // patch_size
                    up_sample_attention = nlc[last_stage].view(nlc[last_stage].shape[0],
                                                                    nlc[last_stage].shape[1], patch_w, patch_w)
                    up_sample_attention = self.attention[feature_layer](up_sample_attention)
                    up_sample_attention = nn.functional.interpolate(up_sample_attention,
                                                                    scale_factor=patch_size,
                                                                    mode='bilinear')
                    layers[feature_layer] = torch.einsum("bcwh, bdwh -> bcdwh", layers[feature_layer],
                                                              up_sample_attention)
                    layers[feature_layer] = layers[feature_layer].view(B, -1, W, H)
                nlc[feature_layer] = self.nl_consistency[feature_layer](
                    (layers[feature_layer], layers[feature_layer])
                )
                last_stage = feature_layer
            features = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in nlc.values()]
        # multi-scale and non-local consistency
        elif self.cfg['model']['multi_scale'] and self.cfg['model']['non_local_consistency']:
            # Bottom-up
            for feature_layer, patch_size in zip(self.cfg['model']['multi_scale'][::-1], self.cfg['model']['patch_size'][::-1]):
                nlc[feature_layer] = self.nl_consistency[feature_layer](
                    (layers[feature_layer], layers[feature_layer])
                )
            features = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in nlc.values()]
        # multi-scale
        elif self.cfg['model']['multi_scale']:
            features = []
            for feature_layer in self.cfg['model']['multi_scale'][::-1]:
                features.append(layers[feature_layer])
        # non-local consistency
        elif self.cfg['model']['non_local_consistency']:
            nlc['b7'] = self.nl_consistency['b7'](
                (layers['b7'], layers['b7'])
            )
            features = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in nlc.values()]
        else:
            features = []
        self.backbone_forward(layers[start_stage], layers, start_stage, end_stage='b7', final=True)
        features.append(layers['final'])
        features = self._norm_feature(features)
        return features, nlc

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.ensemble_classifier_fc(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred_label = pred_dict['cls']
        if self.cfg['model']['non_local_consistency']:
            mask = data_dict['mask'].squeeze(3)
            pred_nlc = pred_dict['nlc']
        else:
            mask = None
            pred_nlc = None
        loss = self.loss_func(pred_label, label, pred_nlc, mask)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self, prob, label):
        y_pred = np.concatenate(prob)
        y_true = np.concatenate(label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # ee
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        prediction_class = np.where(y_pred > 0.5, 1, 0)
        correct = (prediction_class == y_true).sum().item()
        acc = correct / y_true.size
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features, nlc = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict


# ===================================== other modules for DSMSNLC # =====================================

# ================ SRM # ================
class SRM(nn.Module):
    def __init__(self, cfg):
        super(SRM, self).__init__()
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = np.asarray(
            [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])  # shape=(3,3,5,5)
        self.conv_param = torch.Tensor(filters)
        self.conv_param.requires_grad = False
        self.conv = torch.nn.Conv2d(3, 3, (5, 5), stride=1, padding=2, bias=False)
        self.conv.weight.data = self.conv_param

    def forward(self, x):
        x = self.conv(x)
        return x


# ================ NoisePrint++ # ================
class NoisePrintPP(nn.Module):
    """
    Noise Print ++: a noise extractor.
    """

    def __init__(self, cfg):
        super(NoisePrintPP, self).__init__()
        self.cfg = cfg
        num_levels = 17
        out_channel = 1
        self.dncnn = make_net(3, kernels=[3, ] * num_levels,
                              features=[64, ] * (num_levels - 1) + [out_channel],
                              bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                              acts=['relu', ] * (num_levels - 1) + ['linear', ],
                              dilats=[1, ] * num_levels,
                              bn_momentum=0.1, padding=1)
        self.init_weights()

    def init_weights(self):
        np_weights = self.cfg['model']['noise_extractor']['np_weights']
        assert os.path.isfile(np_weights)
        dat = torch.load(np_weights, map_location=torch.device('cpu'))
        self.load_state_dict(dat)

    def forward(self, x):
        x = self.dncnn(x)
        x = x.expand(-1, 3, -1, -1)
        return x


def conv_with_padding(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding=None):
    if padding is None:
        padding = kernelsize // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, dilation=dilation,
                     padding=padding,
                     bias=bias)


def conv_init(conv, act='linear'):
    r"""
    Reproduces conv initialization from DnCNN
    """
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n))


def batchnorm_init(m, kernelsize=3):
    r"""
    Reproduces batchnorm initialization from DnCNN
    """
    n = kernelsize ** 2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()


def make_activation(act):
    if act is None:
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'linear':
        return None
    else:
        assert (False)


def make_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum=0.1, padding=None):
    r"""
    :param nplanes_in: number of input feature channels
    :param kernels: list of kernel size for convolution layers
    :param features: list of hidden layer feature channels
    :param bns: list of whether to add batchnorm layers
    :param acts: list of activations
    :param dilats: list of dilation factors
    :param bn_momentum: momentum of batchnorm
    :param padding: integer for padding (None for same padding)
    """

    depth = len(features)
    assert (len(features) == len(kernels))

    layers = list()
    for i in range(0, depth):
        if i == 0:
            in_feats = nplanes_in
        else:
            in_feats = features[i - 1]

        elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding,
                                 bias=not (bns[i]))
        conv_init(elem, act=acts[i])
        layers.append(elem)

        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum=bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)

        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)


# ================ BayarConv2d # ================
class BayarConv2d(nn.Module):
    def __init__(self, cfg):
        self.in_channels = 3
        self.out_channels = cfg['model']['noise_extractor']['out_channels']
        self.kernel_size = 5
        self.stride = 1
        self.padding = 2
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, self.kernel_size ** 2 - 1),
                                   requires_grad=True)

    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]),
                                dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x


# ================ DCAM # ================

class DCAM(nn.Module):
    def __init__(self, cfg, in_channels):
        super(DCAM, self).__init__()
        if cfg['model']['dcam']['name'] == "dcbam":
            self.dcam = DCbam(in_channels)
        elif cfg['model']['dcam']['name'] == "dca":
            self.dcam = DCANet()

    def forward(self, x):
        return self.dcam(x[0], x[1])


# ======= DCAM.DCbam # =======
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class DCbam(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(DCbam, self).__init__()
        self.ChannelGate_1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.ChannelGate_2 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        self.fusion_block = nn.Sequential(
            nn.Conv2d(2 * gate_channels, gate_channels, 1),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU()
        )
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x_1, x_2):
        """
        Two Stram Channels Attention Modules
        Args:
            x_1: rgb stream
            x_2: noise stream

        Returns:

        """
        x_2_out = self.ChannelGate_1(x_2)
        x_1 = torch.concat([x_1, x_2_out], dim=1)
        x_1 = self.fusion_block(x_1)
        x_1 = self.ChannelGate_2(x_1)
        if not self.no_spatial:
            x_1 = self.SpatialGate(x_1)
        return x_1


# ======= DCAM.DCANet # =======
class PreAttention(nn.Module):
    def __init__(self):
        super(PreAttention, self).__init__()

    def forward(self, x):
        """

        Args:
            x: x.shape = (B, C, 1, 1)

        Returns:

        """
        B, C, _, _ = x.shape
        q = x.squeeze(dim=2)
        k = x.squeeze(dim=2).transpose(2, 1)
        tmp = F.softmax((q * k).flatten(1), dim=1).view(-1, C, C)
        return torch.unsqueeze(torch.matmul(tmp, q), dim=-1) + x


class PostAttention(nn.Module):
    def __init__(self):
        super(PostAttention, self).__init__()
        self.conv1d = nn.Conv1d(1, 1, 1)

    def forward(self, x, W, H):
        """

        Args:
            x: x.shape = (B, C, 1, 1)

        Returns:

        """
        B, C, _, _ = x.shape
        x = x.view(B, 1, C)
        x = self.conv1d(x)
        x = x.view(B, C, 1, 1)
        x = torch.sigmoid(x)
        return x.expand(B, C, W, H)


class DCANet(nn.Module):
    def __init__(self):
        super(DCANet, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.pre_attention = PreAttention()
        self.post_attention = PostAttention()

    def forward(self, x_1, x_2):
        """

        Args:
            x_1: rgb stream x.shape = (B, C, W, H)
            x_2: noise stream x.shape = (B, C, W, H)

        Returns: y.shape = (B, C, W, H)

        """
        B, C, W, H = x_1.shape
        x = torch.concat([x_1, x_2], dim=1)
        tmp = self.pooling(x)
        tmp = self.pre_attention(tmp)
        tmp = self.post_attention(tmp, W, H)
        return x * tmp


# ================ NLConsistency # ================
class PatchEmbedding(nn.Module):
    """
      Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=12, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        #
        # embed_dim表示切好的图片拉成一维向量后的特征长度
        #
        # 图像共切分为N = HW/P^2个patch块
        # 在实现上等同于对reshape后的patch序列进行一个PxP且stride为P的卷积操作
        # output = {[(n+2p-f)/s + 1]向下取整}^2
        # 即output = {[(n-P)/P + 1]向下取整}^2 = (n/P)^2
        #
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x  # x.shape is [8, 196, 768]


class NLConsistency(nn.Module):
    def __init__(self, img_size, in_channels, patch_size):
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = 768
        self.embed_dim_sqrt = math.sqrt(self.embed_dim)
        super(NLConsistency, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size=img_size, in_channels=in_channels, patch_size=patch_size)

    def forward(self, x):
        x_pe = self.patch_embedding(x)
        x_pe = torch.matmul(x_pe, x_pe.transpose(2, 1)) / self.embed_dim_sqrt
        # return x_pe
        return torch.sigmoid(x_pe)


# ================ Self Enhancement # ================

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=1, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class SelfEnhancement(nn.Module):
    def __init__(self, in_channels, CA=False):
        super(SelfEnhancement, self).__init__()
        self.CA = CA
        self.median_filter = MedianPool2d()
        self.sigmoid = nn.Sigmoid()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        if self.CA:
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.gmp = nn.AdaptiveMaxPool2d(output_size=1)
            self.mlp = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):
        tmp = x - self.median_filter(x)  # noi = x - M(x)
        tmp = x + self.conv2d(self.sigmoid(tmp))  # ne = x + Conv(sigmoid(noi))
        if self.CA:
            tmp = x + tmp * self.sigmoid(
                self.mlp(self.gap(tmp) + self.gmp(tmp)))  # out = x + ne * sigmoid(MLP(GAP(ne) + GMP(ne)))
        return tmp


# ================ Attention Module # ================
class AttentionModule(nn.Module):
    def __init__(self, in_channels=144, out_channels=8):
        super(AttentionModule, self).__init__()
        self.attention_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.attention_block(x)
