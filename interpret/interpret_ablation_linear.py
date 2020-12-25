import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import time
import argparse
import os
import csv
import sys
import shutil
import inspect
from collections import OrderedDict
import itertools
from sklearn import metrics
import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
import math

sys.path.insert(1, './../')
from lib import load_data_saveidx as load_data
from model import MobileNet
from utils import *

if __name__ == "__main__":
    print("Number of command line args =", len(sys.argv))
    n_segments = int(sys.argv[1])
    model_weights = "model_params.torch"
    home_dir = sys.argv[2]
    
    print("script started!!")
    print(f'Model weights = {model_weights}')
    root_dir = home_dir + '/data/training2017_raligned/'

    cuda_flag = torch.cuda.is_available()
    print(cuda_flag)
    ### Check for a free GPU
    if cuda_flag:
        selected_gpu = torch.cuda.current_device()
        print(torch.cuda.get_device_name(selected_gpu))
        torch.cuda.set_device(selected_gpu)

    inputsize = 9000
    en_manualfeatures = False
    trans = (transforms.Compose([load_data.CropWithPeak(inputsize)]))
    kfolds = 5
    randomseed = 123
    splitnum = 0

    data = load_data.Physionet2017Dataset(root_dir, transform=trans,
            kfolds=kfolds,
            split_num=splitnum,
            val_size=0,
            split_type="test",
            random_seed=randomseed,
            en_cache=False,
            manual_features=en_manualfeatures,
           saveidx_dir=os.path.join(home_dir, 'model'))
    print("len of data =", len(data.labels))

    print("labels:", data.labels_list)

    net = MobileNet.MobileNet(16, 4)

    net.load_state_dict(torch.load(home_dir + '/model/' + model_weights, lambda s, v: s))
    _ = net.eval()
    
    if cuda_flag:
        net.cuda()
    
    
    perms = set([])
    perms.add((1,) * (n_segments + 1))
    # perms.add((0,) * n_segments)
    for ii in range(n_segments):
        dummy_ones = [1] * (n_segments + 1)
        dummy_ones[ii] = 0
        perms.add(tuple(dummy_ones))

    for ii in range(n_segments - 1):
        dummy_ones = [1] * (n_segments + 1)
        dummy_ones[ii] = 0
        dummy_ones[ii + 1] = 0
        perms.add(tuple(dummy_ones))

    dummy_ones = [1] * (n_segments + 1)
    dummy_ones[0] = 0
    dummy_ones[-2] = 0
    perms.add(tuple(dummy_ones))

    dummy_ones = [1] * (n_segments + 1)
    dummy_ones[-1] = 0
    perms.add(tuple(dummy_ones))

    print("Length of perms =", len(perms))

    n_samples = data.labels.shape[0]

    scores = []
    score = np.zeros(n_samples)
    labels = np.zeros(n_samples)

    SIZE = 200
    indexes = [i for i in range(n_samples)]
    indexes = indexes[:SIZE]
    
    print("ABLATION>>>")
    #fout.write("ABLATION>>>" + "\n")

    for perm in perms:
        perm = list(perm)
        print("perm:", perm)
        score *= 0
        labels *= 0
        count = 0
        for idx in tqdm.tqdm(indexes):
            sample = data[idx]
            ecg = sample['data']
            peaks = sample['peaks']
            label = sample['label']
            labels[idx] = label

            if perm[-1] == 1:
                new_ecg = masking_linear_interpolation_cont(ecg, peaks, perm[:-1])
            else:
                if len(sample) == 0:
                    stretched = []
                else:
                    stretched = align(sample)
                new_ecg = CropDataOnly(stretched)

            score[idx] = int(predict(np.expand_dims(new_ecg, axis=0), net, cuda_flag).argmax())
            #print("here")
            
        scores.append({'score': score[indexes], 'perm': perm})
        print(sum(score[indexes] == labels[indexes]) / labels[indexes].size)
        #fout.write("\n" + str(sum(score[indexes] == labels[indexes]) / labels[indexes].size)+ "\n")

    ye = np.asarray(scores)

    for i in range(ye.shape[0]):
        if np.sum(ye[i]['perm']) == len(ye[i]['perm']):
            print('Superpixel:', ye[i]['perm'])
            acc_all =  sum(ye[i]['score'] == labels[indexes]) / labels[indexes].size
            print('Accuracy:', acc_all)
            cm_all = confusion_matrix(labels[indexes], ye[i]['score'])
            print(cm_all)
            fraction0 = fraction_AF(cm_all)
            print(f'Original fraction of AF = ', print_list_with_precision(fraction0))

            print('\n\n')

    print("------------\n")
    for i in range(ye.shape[0]):
        print('Superpixel:', ye[i]['perm'])
        acc =  sum(ye[i]['score'] == labels[indexes]) / labels[indexes].size
        acc = acc_all - acc
        print('Accuracy:', acc)
        cm = confusion_matrix(labels[indexes], ye[i]['score'])
        print(cm)
        print(fraction_AF(cm))
        fraction = fraction_AF(cm) - fraction0
        print(f'change in fraction_AF = ', print_list_with_precision(fraction))

        print('\n\n')

