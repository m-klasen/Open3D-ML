from abc import abstractmethod
from tqdm import tqdm
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
import random

import tensorflow as tf
import numpy as np
from ml3d.utils import dataset_helper

from ml3d.datasets.utils import DataProcessing
from sklearn.neighbors import KDTree


class TFDataloader():
    def __init__(self,
                 *args,
                 dataset=None,
                 model=None,
                 no_progress: bool = False,
                 **kwargs):
        self.dataset = dataset
        self.model = model
        self.preprocess = model.preprocess
        self.transform = model.transform
        self.get_batch_gen = model.get_batch_gen
        self.model_cfg = model.cfg

        if self.preprocess is not None:
            cache_dir = getattr(dataset.cfg, 'cache_dir')
            
            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = dataset_helper.Cache(
                self.preprocess,
                cache_dir=cache_dir,
                cache_key=dataset_helper._get_hash(
                    repr(self.preprocess)[:-15]))
            print("cache key : {}".format(repr(self.preprocess)[:-15]))

            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)
                ['name'] not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                for idx in tqdm(range(len(dataset)),
                                desc='preprocess',
                                disable=no_progress):
                    attr = dataset.get_attr(idx)
                    data = dataset.get_data(idx)
                    name = attr['name']

                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None

        self.num_threads = 3  # TODO : read from config
        self.split = dataset.split
        self.pc_list = dataset.path_list
        self.num_pc = len(self.pc_list)

    def read_data(self, key):
        attr = self.dataset.get_attr(key)
        if self.cache_convert is None:
            data = self.dataset.get_data(key)
        else:
            data = self.cache_convert(attr['name'])


        return data, attr

    def get_loader(self):
        gen_func, gen_types, gen_shapes = self.get_batch_gen(self)

        tf_dataloader = tf.data.Dataset.from_generator(gen_func, gen_types,
                                                    gen_shapes)

        tf_dataloader = tf_dataloader.map(map_func=self.transform,
                                    num_parallel_calls=self.num_threads)

        return tf_dataloader

