#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:41:06 2017

@author: Matteo Gadaleta
"""
import torch
import os

import pandas as pd
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.interpolate import interp1d
from cachetools import LRUCache
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


class Physionet2017Dataset(Dataset):
    """
    AF Classification from a short single lead ECG recording:
    the Physionet/Computing in Cardiology Challenge 2017
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        preprocessing=None,
        en_cache=False,
        kfolds=1,
        split_num=0,
        val_size=0.2,
        split_type="train",
        random_seed=123,
        manual_features=False,
    ):

        ### Load labels
        self.labels = pd.read_csv(
            join(root_dir, "REFERENCE-v3.csv"), names=["ref", "label"], index_col="ref"
        )
        self.labels_list = ["N", "A", "O", "~"]

        ### Validation split
        if kfolds > 1:
            # Separate train/val and test
            kf = StratifiedKFold(
                n_splits=kfolds, shuffle=True, random_state=random_seed
            )
            trainval_index, test_index = list(kf.split(self.labels, self.labels))[
                split_num
            ]
            print("Val size:", val_size)
            print("Splits::")
            # print(len(trainval_index))
            print(f'Test indices, length = {len(test_index)}')
            print('\n\n--')
            for ti in test_index:
                print(ti)
            print('--\n\n')
            if val_size == 0:
                if split_type == "train":
                    self.labels = self.labels.iloc[trainval_index]
                elif split_type == "test":
                    self.labels = self.labels.iloc[test_index]
                else:
                    raise Exception("Invalid slit_type")
            else:
                # Separate train and val
                ss = StratifiedShuffleSplit(
                    n_splits=1, test_size=val_size, random_state=random_seed
                )
                train_index, val_index = list(
                    ss.split(trainval_index, self.labels.iloc[trainval_index])
                )[0]
                train_index, val_index = (
                    trainval_index[train_index],
                    trainval_index[val_index],
                )

                if split_type == "train":
                    self.labels = self.labels.iloc[train_index]
                elif split_type == "val":
                    self.labels = self.labels.iloc[val_index]
                elif split_type == "test":
                    self.labels = self.labels.iloc[test_index]
                else:
                    raise Exception("Invalid slit_type")

        self.root_dir = root_dir
        self.manual_features = manual_features
        self.transform = transform

        ### Initialize cache
        self.en_cache = en_cache
        if en_cache:
            maxsize_cache = len(self.labels)
            self.cache = LRUCache(maxsize_cache)

            for idx in tqdm(range(len(self.labels))):
                sample = self.load_data(idx)
                self.cache[idx] = sample

    def labels_distribution(self):
        labels_distribution = []
        for label in self.labels_list:
            label_count = list(self.labels["label"]).count(label)
            labels_distribution.append(label_count)

        labels_distribution = np.array(labels_distribution) / np.sum(
            labels_distribution
        )

        return labels_distribution

    def __len__(self):
        return len(self.labels)

    def load_data(self, idx):

        ref = self.labels.iloc[idx].name
        filename = ref + ".mat"

        label = self.labels.iloc[idx].label
        label = self.labels_list.index(label)
        
        obj = loadmat(join(self.root_dir, filename))
        data = obj["val"][0, :]

        

        if self.manual_features:
            data = feature_extractor(data)

        sample = {"data": data, "label": label}

        if 'peaks' in obj:
            sample['peaks'] = obj['peaks'][0]

        return sample

    def __getitem__(self, idx):

        ### Take from cache if exists
        if self.en_cache:
            sample = self.cache.get(idx)
        else:
            sample = self.load_data(idx)

        ### Transform
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, num_expand_dims):
        self.num_expand_dims = num_expand_dims

    def __call__(self, sample):

        if type(sample) is dict:
            data, label = sample["data"], sample["label"]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            for i in range(self.num_expand_dims):
                data = np.expand_dims(data, 0)
            return {
                "data": torch.from_numpy(data).float(),
                "label": torch.from_numpy(np.array([label])).short(),
            }

        else:
            for i in range(self.num_expand_dims):
                sample = np.expand_dims(sample, 0)
            return torch.from_numpy(sample).float()


class Resample(object):
    def __init__(self, original_fs, out_fs):

        self.original_fs = original_fs
        self.out_fs = out_fs

        self.original_period = 1.0 / original_fs
        self.out_period = 1.0 / out_fs

    def __call__(self, sample):
        data, label = sample["data"], sample["label"]

        t_original = np.arange(0, len(data) / self.original_fs, self.original_period)
        t_out = np.arange(0, len(data) / self.original_fs, self.out_period)

        f = interp1d(t_original, data, kind="linear", fill_value="extrapolate")
        data_out = f(t_out)

        return {"data": data_out, "label": label}


class Crop(object):
    def __init__(self, num_samples, mode="constant"):

        self.num_samples = num_samples

    def __call__(self, sample):
        data, label = sample["data"], sample["label"]

        if len(data) >= self.num_samples:
            start_idx = np.random.randint(len(data) - self.num_samples + 1)
            data = data[start_idx : start_idx + self.num_samples]
        else:
            left_pad = int(np.ceil((self.num_samples - len(data)) / 2))
            right_pad = int(np.floor((self.num_samples - len(data)) / 2))
            data = np.pad(data, (left_pad, right_pad), "constant")

        return {"data": data, "label": label}


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torch.utils.data import DataLoader

    root_dir = "../data/training2017_filtered"
    num_folds = 5
    prev_test_dataset = None

    for split_num in range(num_folds):
        train_dataset = Physionet2017Dataset(
            root_dir,
            transform=transforms.Compose(
                [Resample(300, 200), Crop(200 * 30), ToTensor(1)]
            ),
            kfolds=num_folds,
            val_size=0.2,
            split_num=split_num,
            split_type="train",
            random_seed=123,
        )

        val_dataset = Physionet2017Dataset(
            root_dir,
            transform=transforms.Compose(
                [Resample(300, 200), Crop(200 * 30), ToTensor(1)]
            ),
            kfolds=num_folds,
            val_size=0.2,
            split_num=split_num,
            split_type="val",
            random_seed=123,
        )

        test_dataset = Physionet2017Dataset(
            root_dir,
            transform=transforms.Compose(
                [Resample(300, 200), Crop(200 * 30), ToTensor(1)]
            ),
            kfolds=num_folds,
            val_size=0.2,
            split_num=split_num,
            split_type="test",
            random_seed=123,
        )

        print(
            "TRAIN",
            len(train_dataset),
            "VALIDATION",
            len(val_dataset),
            "TEST",
            len(test_dataset),
        )

        train_labels = list(train_dataset.labels.index)
        val_labels = list(val_dataset.labels.index)
        test_labels = list(test_dataset.labels.index)

        for label in train_labels:
            assert not label in val_labels
            assert not label in test_labels
        for label in val_labels:
            assert not label in train_labels
            assert not label in test_labels
        for label in test_labels:
            assert not label in train_labels
            assert not label in val_labels

        if prev_test_dataset != None:
            for label in test_labels:
                prev_test_labels = list(prev_test_dataset.labels.index)
                assert not prev_test_labels in test_labels
                prev_test_dataset = test_dataset

        print(train_dataset.labels_distribution())
        print(val_dataset.labels_distribution())
        print(test_dataset.labels_distribution())
