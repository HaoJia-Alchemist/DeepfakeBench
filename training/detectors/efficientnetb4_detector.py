'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the EfficientDetector

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

Reference:
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
'''
import math
import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

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
import random

from .efficientnet import EfficientNet
from .nacl_detector import DCAM, NLConsistency, AttentionModule

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='efficientnet-b4')
class EfficientDetector(AbstractDetector):
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
        temp_layers = self.rgb_backbone(temp_rgb)
        self.relu = nn.ReLU()
        # multi-scale and non-local consistency and fusion module
        if cfg['model']['multi_scale'] and cfg['model']['non_local_consistency'] and cfg['model']['fusion_module']:
            # Bottom-up
            self.nl_consistency = nn.ModuleDict()
            temp_nlc = {}
            self.attention = nn.ModuleDict()
            last_stage = ""
            for feature_layer, patch_size in zip(cfg['model']['multi_scale'][::-1], cfg['model']['patch_size'][::-1]):
                if last_stage != "":
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
            self.nl_consistency = nn.ModuleDict()
            temp_nlc = {}
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
                NLConsistency(temp_dcamed.shape[2], temp_dcamed.shape[1], 1)
            )
            temp_nlc['b7'] = self.nl_consistency['b7'](
                (temp_layers['b7'], temp_layers['b7'])
            )
            temp_out = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in temp_nlc.values()]
        else:
            temp_out=[]
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

    def features(self, data_dict: dict) -> torch.tensor:
        layers = self.rgb_backbone(data_dict['image'])
        # multi-scale and non-local consistency and fusion module
        if self.cfg['model']['multi_scale'] and self.cfg['model']['non_local_consistency'] and self.cfg['model']['fusion_module']:
            # Bottom-up
            nlc = {}
            last_stage = ""
            for feature_layer, patch_size in zip(self.cfg['model']['multi_scale'][::-1], self.cfg['model']['patch_size'][::-1]):
                if last_stage != "":
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
            nlc = {}
            for feature_layer, patch_size in zip(self.cfg['model']['multi_scale'][::-1], self.cfg['model']['patch_size'][::-1]):
                nlc[feature_layer] = self.nl_consistency[feature_layer](
                    (layers[feature_layer], layers[feature_layer])
                )
            features = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in nlc.values()]
        # multi-scale
        elif self.cfg['model']['multi_scale']:
            features = []
            nlc = {}
            for feature_layer in self.cfg['model']['multi_scale'][::-1]:
                features.append(layers[feature_layer])
        # non-local consistency
        elif self.cfg['model']['non_local_consistency']:
            nlc = {}
            nlc['b7'] = self.nl_consistency['b7'](
                (layers['b7'], layers['b7'])
            )
            features = [i.reshape((i.shape[0], i.shape[1], int(math.sqrt(i.shape[2])), -1)) for i in nlc.values()]
        else:
            nlc = {}
            features = []
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


    def get_test_metrics(self,prob, label):
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
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'nlc': nlc}
        # if inference:
        #     self.prob.append(
        #         pred_dict['prob']
        #         .detach()
        #         .squeeze()
        #         .cpu()
        #         .numpy()
        #     )
        #     self.label.append(
        #         data_dict['label']
        #         .detach()
        #         .squeeze()
        #         .cpu()
        #         .numpy()
        #     )
        return pred_dict

