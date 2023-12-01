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

from detectors.base_detector import AbstractDetector
from detectors.efficientnet import EfficientNet
from metrics.base_metrics_class import calculate_metrics_for_train

from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='baseline')
class Baseline(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        # Backbone
        if 'efficientnet' in config['model']['backbone']:
            self.rgb_backbone = EfficientNet.from_pretrained(config['model']['backbone'], advprop=True,
                                                             num_classes=config['model']['num_classes'])
        else:
            raise ValueError("Unsupported Backbone!")
        temp_rgb = torch.randn(2, 3, config['resolution'], config['resolution'])
        temp_layers = self.rgb_backbone(temp_rgb)
        temp_features = []
        temp_features.append(temp_layers['final'])
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(config['model']['drop_rate'])
        temp_features = self._norm_feature(temp_features)
        self._fc = nn.Linear(temp_features.shape[1], config['model']['num_classes'])

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def build_backbone(self, config):
        pass

    def features(self, data_dict: dict) -> torch.tensor:
        layers = self.rgb_backbone(data_dict['image'])
        features = []
        nlc = {}
        features.append(layers['final'])
        features = self._norm_feature(features)
        return features, nlc

    def _norm_feature(self, features):
        x = torch.concat([self._avg_pooling(i) for i in features], dim=1)
        x = x.view(x.size(0), -1)
        return x

    def classifier(self, features: torch.tensor) -> torch.tensor:
        x = self._dropout(features)
        x = self._fc(x)
        return x

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred_label = pred_dict['cls']
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

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
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
        # reset the prob and label
        self.prob, self.label = [], []
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features, nlc = self.features(data_dict)
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
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
        return pred_dict
