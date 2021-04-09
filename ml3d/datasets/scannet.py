import open3d as o3d
import numpy as np
import torch
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET
from .utils import BEVBox3D

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Scannet(BaseDataset):
    """
    Scannet 3D dataset for Object Detection, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='Scannet',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
            kwargs:
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 18

        self.classes = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.cat_ids2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.cat_ids))
        }

        self.label_to_names = self.get_label_to_names()

        self.train_files = glob(self.dataset_path + '/train/*.pth')
        self.val_files = glob(self.dataset_path + '/val/*.pth')

        self.semantic_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36,
            39
        ]

    def get_label_to_names(self):
        return self.label2cat

    def get_split(self, split):
        return ScannetSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['train', 'training']:
            return self.train_files
        elif split in ['test', 'testing']:
            return self.test_scenes
        elif split in ['val', 'validation']:
            return self.val_files

        raise ValueError("Invalid split {}".format(split))

    def is_tested(self):
        pass

    def save_test_result(self):
        pass


class ScannetSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        self.path_list = dataset.get_split_list(split)

        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        scene = self.path_list[idx]
        pc, feat, labels = torch.load(scene)

        data = {
            'point': pc,
            'feat': feat,
            'calib': None,
            'label': labels.astype(np.int32),
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(Scannet)
