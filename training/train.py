# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset
from dataset.dsmsnlc_dataset import DSMSNLCDataset
from dataset.rgbmsnlc_dataset import RGBMSNLCDataset
from dataset.test_dataset import testDataset

from trainer.trainer import Trainer
from detectors import DETECTOR

import argparse
from logger import create_logger

from mmengine.config import Config, DictAction

parser = argparse.ArgumentParser(description='Deepfake Detection Args')
parser.add_argument('--config_file', type=str,
                    default=r'training/config/detector/rgbmsnlc.yaml',
                    help='path to detector YAML file')
parser.add_argument("--opts", action=DictAction, help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()


def mergeConfig(args, config):
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    config.update(args)
    return config

def init_seed(config):
    seed = random.randint(1, 10000) if config['manualSeed'] is None else config['manualSeed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True


def prepare_training_data(config):
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa' or config['model_name'] == 'dsp_fwa':
            train_set = FWABlendDataset(config)
        else:
            raise NotImplementedError(
                'Only facexray, fwa, and dsp_fwa are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config)  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'dsmsnlc':
        train_set = DSMSNLCDataset(config)
    elif 'dataset_type' in config and config['dataset_type'] == 'rgbmsnlc_dataset':
        train_set = RGBMSNLCDataset(config)
    else:
        train_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='train',
        )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=int(config['workers']),
        collate_fn=train_set.collate_fn,
    )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = [test_name]  # specify the current test dataset
        test_set = testDataset(
            config=config,
            mode='test',
        )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'adamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    config = Config.fromfile(args.config_file)
    if args.opts is not None:
        config.merge_from_dict(args.opts)
    # create logger
    config['log_dir'] = os.path.join(config['log_dir'], f"{config['task_name']}_train_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(config['log_dir'], exist_ok=True)
    logger = create_logger(os.path.join(config['log_dir'], 'train.log'))
    logger.info('Save log to {}'.format(config['log_dir']))

    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model']['name']]
    model = model_class(config)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)
    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        test_metrics_all_datasets = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
        )
        logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {test_metrics_all_datasets}!")
    logger.info("Stop Training on best Testing metric {}".format(
        {k: dict(v) for k, v in trainer.best_metrics_all_time.items()}))

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()
