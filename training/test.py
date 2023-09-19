"""
eval pretained model.
"""
import os
import sys

import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle

from mmengine import DictAction, Config
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.test_dataset import testDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger
torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='Deepfake Detection Test Args')
parser.add_argument('--config_file', type=str,
                    default='/home/jh/disk/workspace/DeepfakeBench/training/config/detector/rgbmsnlc.yaml',
                    help='path to detector YAML file')
parser.add_argument("--device_id", type=str, default='1')
parser.add_argument('--checkpoints', type=str,
                    default='/home/jh/disk/logs/DeepfakeBench/rgbmsnlc/rgbmsnlc_DA_FF_all_c23_train_20230910191400/test/FaceForensics++/ckpt_best.pth')
parser.add_argument("--opts", action=DictAction, help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()

device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


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


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader, tsne_dict):
    for i, data_dict in enumerate(tqdm(data_loader)):
        # get data
        data, label, label_spe, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['label_spe'], data_dict['mask'], data_dict['landmark']
    
        # move data to GPU
        data_dict['image'], data_dict['label'], data_dict['label_spe'] = data.to(device), label.to(device), label_spe.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)

        # update predictions by adding label_spe for t-SNE
        predictions['label_spe'] = label_spe.cpu().numpy()

        # deal with the feat, pooling if needed
        if len(predictions['feat'].shape) == 4:
            predictions['feat'] = F.adaptive_avg_pool2d(predictions['feat'], (1, 1)).reshape(predictions['feat'].shape[0], -1)
        predictions['feat'] = predictions['feat'].cpu().numpy()
    
        # tsne
        tsne_dict['feat'].append(predictions['feat'])
        tsne_dict['label_spe'].append(predictions['label_spe'])
    return predictions, tsne_dict
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # tsne dict
    tsne_dict = defaultdict(list)

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        # compute loss for each dataset
        predictions, tsne_dict = test_one_dataset(model, test_data_loaders[key], tsne_dict)
        
        # compute metric for each dataset
        metric_one_dataset = model.get_test_metrics()
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        metric_str = f"dataset: {key}        "
        for k, v in metric_one_dataset.items():
            metric_str += f"testing-metric, {k}: {v}    "
        print(metric_str)

    # print(f"before concat, feat shape is: {tsne_dict['feat'].shape}, label is: {tsne_dict['label_spe']}")
    tsne_dict['feat'] = np.concatenate(tsne_dict['feat'], axis=0)
    # print(f"after concat, feat shape is: {tsne_dict['feat'].shape}, label is: {tsne_dict['label_spe']}")
    tsne_dict['label_spe'] = np.concatenate(tsne_dict['label_spe'], axis=0)
    print('===> Test Done!')
    return metrics_all_datasets, tsne_dict

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    config = Config.fromfile(args.config_file)
    if args.opts is not None:
        config.merge_from_dict(args.opts)
    if args.checkpoints:
        config['model']['checkpoints'] = args.checkpoints
    # print configuration
    print("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    print(params_string)
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model']['name']]
    model = model_class(config).to(device)
    epoch = 0
    if config['model']['checkpoints']:
        try:
            epoch = int(config['model']['checkpoints'].split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(config['model']['checkpoints'], map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
        sys.exit()
    # prepare the metric
    metric_scoring = choose_metric(config)
    
    # start training
    best_metric, tsne_dict = test_epoch(model, test_data_loaders)
    print(f"===> End with testing {metric_scoring}: {best_metric}!")
    print("Stop on best Testing metric {}".format(best_metric))

    # save tsne
    with open(os.path.join(config['log_dir'], f"tsne_dict_{config['model']['name']}_{epoch}.pkl"), 'wb') as f:
        pickle.dump(tsne_dict, f)
    print('===> Save tsne done!')

if __name__ == '__main__':
    main()
