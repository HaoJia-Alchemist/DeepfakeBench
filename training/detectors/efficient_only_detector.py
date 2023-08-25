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


@DETECTOR.register_module(module_name='efficient_only_detector')
class EfficientOnlyDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        temp_rgb = torch.randn(2, 3, config['resolution'], config['resolution'])
        # Backbone
        if 'efficientnet' in config['model']['backbone']:
            self.rgb_backbone = EfficientNet.from_pretrained(config['model']['backbone'], advprop=True,
                                                             num_classes=config['model']['num_classes'])
            temp_rgb_features = self.rgb_backbone(temp_rgb)
        else:
            raise ValueError("Unsupported Backbone!")
        temp_out = temp_rgb_features['final']
        self.ensemble_classifier_fc = nn.Sequential(nn.Dropout(p=config['model']['drop_rate']),
                                                    nn.Linear(temp_out.shape[1], config['model']['mid_dim']),
                                                    nn.Hardswish(),
                                                    nn.Linear(config['model']['mid_dim'],
                                                              config['model']['num_classes']))
        self.relu = nn.ReLU(inplace=True)

    def build_backbone(self, config):
        pass

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def _norm_feature(self, x):
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def features(self, data_dict: dict) -> torch.tensor:
        rgb_features = self.rgb_backbone(data_dict['image'])
        features = rgb_features['final']
        features = self._norm_feature(features)
        return features

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.ensemble_classifier_fc(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(pred, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)
        return pred_dict
