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

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='efficientnetb4')
class EfficientOnlyDetector(AbstractDetector):
    # def __init__(self, cfg):
    #     super().__init__()
    #     self.cfg = cfg
    #     self.loss_func = self.build_loss(cfg)
    #     self.prob, self.label = [], []
    #     self.correct, self.total = 0, 0
    #
    #     temp_rgb = torch.randn(2, 3, cfg['resolution'], cfg['resolution'])
    #     # Backbone
    #     if 'efficientnet' in cfg['model']['backbone']:
    #         self.rgb_backbone = EfficientNet.from_pretrained(cfg['model']['backbone'], advprop=True,
    #                                                          num_classes=cfg['model']['num_classes'])
    #     else:
    #         raise ValueError("Unsupported Backbone!")
    #     temp_layers = self.rgb_backbone(temp_rgb)
    #     temp_out=[]
    #     temp_out.append(temp_layers['final'])
    #     self.relu = nn.ReLU(inplace=False)
    #     temp_out = self._norm_feature(temp_out)
    #     self.fc = nn.Linear(temp_out.shape[1], 2)
    #     self.ensemble_classifier_fc = nn.Sequential(nn.Dropout(p=cfg['model']['drop_rate']),
    #                                             nn.Linear(temp_out.shape[1], cfg['model']['mid_dim']),
    #                                             nn.Hardswish(),
    #                                             nn.Linear(cfg['model']['mid_dim'], cfg['model']['num_classes']))
    # def build_backbone(self, config):
    #     pass
    #
    # def build_loss(self, config):
    #     # prepare the loss function
    #     loss_class = LOSSFUNC[config['loss_func']]
    #     loss_func = loss_class()
    #     return loss_func
    #
    # def _norm_feature(self, x):
    #     x = torch.concat([F.adaptive_avg_pool2d(self.relu(i), (1, 1)) for i in x], dim=1)
    #     x = x.view(x.size(0), -1)
    #     return x
    #
    # def features(self, data_dict: dict) -> torch.tensor:
    #     layers = self.rgb_backbone(data_dict['image'])
    #     features = []
    #     nlc={}
    #     features.append(layers['final'])
    #     features = self._norm_feature(features)
    #     return features, nlc
    #
    # def classifier(self, features: torch.tensor) -> torch.tensor:
    #     return self.ensemble_classifier_fc(features)
    #
    # def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
    #     label = data_dict['label']
    #     pred_label = pred_dict['cls']
    #     mask = None
    #     pred_nlc = None
    #     loss = self.loss_func(pred_label, label, pred_nlc, mask)
    #     loss_dict = {'overall': loss}
    #     return loss_dict
    #
    # def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
    #     label = data_dict['label']
    #     pred = pred_dict['cls']
    #     # compute metrics for batch data
    #     auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
    #     metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
    #     return metric_batch_dict
    #
    # def get_test_metrics(self):
    #     y_pred = np.concatenate(self.prob)
    #     y_true = np.concatenate(self.label)
    #     # auc
    #     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    #     auc = metrics.auc(fpr, tpr)
    #     # ee
    #     fnr = 1 - tpr
    #     eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    #     # ap
    #     ap = metrics.average_precision_score(y_true, y_pred)
    #     # acc
    #     prediction_class = np.where(y_pred > 0.5, 1, 0)
    #     correct = (prediction_class == y_true).sum().item()
    #     acc = correct / y_true.size
    #     # reset the prob and label
    #     self.prob, self.label = [], []
    #     return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}
    #
    # def forward(self, data_dict: dict, inference=False) -> dict:
    #     # get the features by backbone
    #     features, nlc = self.features(data_dict)
    #     # get the prediction by classifier
    #     pred = self.classifier(features)
    #     # get the probability of the pred
    #     prob = torch.softmax(pred, dim=1)[:, 1]
    #     # build the prediction dict for each output
    #     pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'nlc': nlc}
    #     if inference:
    #         self.prob.append(
    #             pred_dict['prob']
    #             .detach()
    #             .squeeze()
    #             .cpu()
    #             .numpy()
    #         )
    #         self.label.append(
    #             data_dict['label']
    #             .detach()
    #             .squeeze()
    #             .cpu()
    #             .numpy()
    #         )
    #     return pred_dict
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        #FIXME: current load pretrained weights only from the backbone, not here
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        x = self.backbone.features(data_dict['image'])
        return x

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)

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
            features = self.features(data_dict)
            # get the prediction by classifier
            pred = self.classifier(features)
            # # get the probability of the pred
            prob = torch.softmax(pred, dim=1)[:, 1]
            # # build the prediction dict for each output
            pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
            return pred_dict
