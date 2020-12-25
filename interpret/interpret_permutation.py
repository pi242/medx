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
import glob

sys.path.insert(1, './../')
from lib import load_data_saveidx as load_data
from model import MobileNet
from utils import *

if __name__ == "__main__":
    print("Number of command line args =", len(sys.argv))
    n_segments = int(sys.argv[1])
    model_weights = "model_params.torch"
    home_dir = sys.argv[2]
    align = int(sys.argv[3])
    segmentwise = int(sys.argv[4])
    
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

    n_samples = len(data)
    print(f'n_samples = {n_samples}')
    SIZE = 200
    indexes = [i for i in range(n_samples)]
    indexes = indexes[:SIZE]
    # print("indexes = ", indexes)
    
    # from scipy import signal
    def perturb(sample, perm_sample, segment_num, n_segments, trans, ifalign):
        # print(len(sample['data']))
        peaks = list(sample['peaks']) + [len(sample['data']) - 1]
        # print(peaks)
        perm_peaks = list(perm_sample['peaks']) + [len(perm_sample['data']) - 1]
        current = 0 
        perm_current = 0
        new = []
        for i in range(0, len(peaks)):
            # print(current, peaks[i])
            idxs = create_segment_rr(sample['data'][current:peaks[i]], n_segments)
            # print(f'idxs for {segment_num} = {idxs[segment_num]}')
            dummy = create_segment_rr(perm_sample['data'][perm_current:perm_peaks[i]], n_segments)
            perm_i, perm_j = dummy[segment_num]
            # print("permi, j", perm_i, perm_j)
            [new.extend(sample['data'][current+x:current+y]) for x, y in idxs[:segment_num]]
            # [new.extend(sample['data'][current+x:current+y]) for x, y in [idxs[segment_num]]]
            if ifalign:
                if perm_j - perm_i == 0:
                    resampled = []
                else:
                    resampled = resample_by_interpolation(perm_sample['data'][perm_current+perm_i:perm_current+perm_j], idxs[segment_num][1] - idxs[segment_num][0])
                new.extend(resampled)
            else:
                new.extend(perm_sample['data'][perm_current+perm_i:perm_current+perm_j])
            [new.extend(sample['data'][current+x:current+y]) for x, y in idxs[segment_num + 1:]]
            # resampled = signal.resample(segment, size)
            # aligned.extend(resampled)
            current = peaks[i]
            perm_current = perm_peaks[i]
        new.append(sample['data'][peaks[-1]])
        return trans(np.array(new))


    def perturb_segmentwise(indexes, idx1, segment_num, n_segments, trans, ifalign):
        # print(len(sample['data']))
        sample = data[indexes[idx1]]
        sample_label = sample['label']
        peaks = list(sample['peaks']) + [len(sample['data']) - 1]
        print(idx1)
        current = 0
        new = []
        for i in range(0, len(peaks)):
            idx2 = random.choice(indexes)
            perm_sample = data[indexes[idx2]]
            if idx2 == idx1:
                idx2 = random.choice(indexes)
                perm_sample = data[indexes[idx2]]
            perm_peaks = list(sample['peaks']) + [len(sample['data']) - 1]
            perm_end_idx = random.choice(np.arange(len(perm_peaks)))
            if perm_end_idx == 0:
                perm_start = 0
                perm_end = perm_peaks[perm_end_idx]
            else:
                perm_start = perm_peaks[perm_end_idx - 1]
                perm_end = perm_peaks[perm_end_idx]

            # print(f'idx2 = {idx2}, perm_start = {perm_start}, perm_end = {perm_end}')

            idxs = create_segment_rr(sample['data'][current:peaks[i]], n_segments)
            dummy = create_segment_rr(perm_sample['data'][perm_start:perm_end], n_segments)
            perm_i, perm_j = dummy[segment_num]
            # print(idxs, perm_i, perm_j)
            [new.extend(sample['data'][current+x:current+y]) for x, y in idxs[:segment_num]]
            # [new.extend(sample['data'][current+x:current+y]) for x, y in [idxs[segment_num]]]
            # print(f'org length = {len(sample["data"][current+idxs[segment_num][0]:current+idxs[segment_num][1]])}')
            if ifalign:
                if perm_j - perm_i == 0:
                    resampled = []
                else:
                    # print(idxs[segment_num], np.array(perm_sample['data'][perm_start+perm_i:perm_start+perm_j]).shape)
                    resampled = resample_by_interpolation(perm_sample['data'][perm_start+perm_i:perm_start+perm_j], idxs[segment_num][1] - idxs[segment_num][0])
                # print(f'resampled size = {len(resampled)}')
                new.extend(resampled)
            else:
                new.extend(perm_sample['data'][perm_start+perm_i:perm_start+perm_j])

            [new.extend(sample['data'][current+x:current+y]) for x, y in idxs[segment_num + 1:]]
            # resampled = signal.resample(segment, size)
            # aligned.extend(resampled)
            current = peaks[i]
        new.append(sample['data'][peaks[-1]])
        return trans(np.array(new))

    def perturb_pair_align(sample, perm_sample, snum0, snum1, n_segments, trans):
        # print(len(sample['data']))
        peaks = list(sample['peaks']) + [len(sample['data']) - 1]
        # print(peaks)
        perm_peaks = list(perm_sample['peaks']) + [len(perm_sample['data']) - 1]
        current = 0 
        perm_current = 0
        # current = sample['peaks'][0]
        # perm_current = perm_sample['peaks'][0]
        new = []
        for i in range(len(peaks)):
            # print(current, peaks[i])
            idxs = create_segment_rr(sample['data'][current:peaks[i]], n_segments)
            # print(f'idxs for {segment_num} = {idxs[segment_num]}')
            perm_i0, perm_j0 = create_segment_rr(perm_sample['data'][perm_current:perm_peaks[i]], n_segments)[snum0]
            perm_i1, perm_j1 = create_segment_rr(perm_sample['data'][perm_current:perm_peaks[i]], n_segments)[snum1]
            # print("permi, j", perm_i, perm_j)
            if snum0 == 0 and snum1 == 7:
                if perm_j0 - perm_i0 == 0:
                    resampled = []
                else:
                    org = sample['data'][current+idxs[snum0][0]:current+idxs[snum0][1]]
                    replacement = perm_sample['data'][perm_current+perm_i0:perm_current+perm_j0]
                    # replacement *= (np.mean(org) / np.mean(replacement))
                    # print(org[0], replacement[0])
                    resampled = resample_by_interpolation(replacement, idxs[snum0][1] - idxs[snum0][0])
                new.extend(resampled)
                [new.extend(sample['data'][current+x:current+y]) for x, y in idxs[snum0 + 1:snum1]]
                if perm_j1 - perm_i1 == 0:
                    resampled = []
                else:
                    org = sample['data'][current+idxs[snum1][0]:current+idxs[snum1][1]]
                    replacement = perm_sample['data'][perm_current+perm_i1:perm_current+perm_j1]
                    # replacement *= (np.mean(org) / np.mean(replacement))
                    # print(org[0], replacement[0])
                    resampled = resample_by_interpolation(replacement, idxs[snum1][1] - idxs[snum1][0])
                new.extend(resampled)
            else:
                [new.extend(sample['data'][current+x:current+y]) for x, y in idxs[:snum0]]
                if perm_j0 - perm_i0 == 0:
                    resampled = []
                else:
                    org = sample['data'][current+idxs[snum0][0]:current+idxs[snum0][1]]
                    replacement = perm_sample['data'][perm_current+perm_i0:perm_current+perm_j0]
                    # replacement *= (np.mean(org) / np.mean(replacement))
                    # print(org[0], replacement[0])
                    resampled = resample_by_interpolation(replacement, idxs[snum0][1] - idxs[snum0][0])
                new.extend(resampled)
                if perm_j1 - perm_i1 == 0:
                    resampled = []
                else:
                    org = sample['data'][current+idxs[snum1][0]:current+idxs[snum1][1]]
                    replacement = perm_sample['data'][perm_current+perm_i1:perm_current+perm_j1]
                    # replacement *= (np.mean(org) / np.mean(replacement))
                    # print(org[0], replacement[0])
                    resampled = resample_by_interpolation(replacement, idxs[snum1][1] - idxs[snum1][0])
                new.extend(resampled)
                [new.extend(sample['data'][current+x:current+y]) for x, y in idxs[snum1 + 1:]]

            current = peaks[i]
            perm_current = perm_peaks[i]
        new.append(sample['data'][peaks[-1]])
        return trans(np.array(new))

    def perturb_RR(sample, perm_sample, trans):
        # print(len(sample['data']))
        peaks = list(sample['peaks']) + [len(sample['data']) - 1]
        vec = sample['data']
        perm_vec = perm_sample['data']
        perm_peaks = list(perm_sample['peaks']) + [len(perm_sample['data']) - 1]
        # print(f'Perm peaks = {perm_peaks}')
        current = 0 
        perm_current = 0
        new = []
        for i in range(0, len(peaks)):
            segment = vec[current:peaks[i]]
            if len(segment) == 0:
                resampled = []
            else:
                resampled = resample_by_interpolation(segment, int(perm_peaks[i] - perm_current))
            # print(i, peaks[i] - current, len(segment))
            # print(i, (perm_peaks[i] - perm_current), len(resampled))
            new.extend(resampled)
            current = peaks[i]
            perm_current = perm_peaks[i]
        # print(f'Final length = {len(new)}')
        return trans(np.array(new))



    segs_selection = [(i, i + 1) for i in range(n_segments - 1)] + [(0, 7)] + [(i, i) for i in range(n_segments)] 
    print(f'segs_selection = {segs_selection}')
    scores = []
    score = {k:[] for k in segs_selection + [(n_segments, n_segments), (-1, -1)]}
    labels = []
    count = 0

    for idx1 in tqdm.tqdm(range(len(indexes))):
        sample = data[indexes[idx1]]
        num_peaks = len(sample['peaks'])
        labels.append(sample['label'])

        # print(f'idx1 = {indexes[idx1]}, num_peaks = {num_peaks}')
        for j in range(1, len(indexes) + 1):
            idx2 = indexes[(idx1 + j) % len(indexes)]
            perm_sample = data[idx2]
            new_num_peaks = len(perm_sample['peaks'])
            if new_num_peaks >= num_peaks:
                # print(f'j = {j}, idx2 = {idx2}, num_peaks = {new_num_peaks}')
                break

        # print(f'idx1 = {idx1}, num_peaks = {num_peaks}, idx2 = {idx2}, num_peaks = {new_num_peaks}')
        for (snum0, snum1) in segs_selection:
            # if segment_num != 0:
            #     continue
            # print(f'Segment = {segment_num}')
            # print(snum0, snum1)
            if snum0 != n_segments and snum0 == snum1:
                new_ecg = perturb(sample, perm_sample, snum0, n_segments, CropDataOnly, align)
            try:
                new_ecg = perturb_pair_align(sample, perm_sample, snum0, snum1, n_segments, CropDataOnly)
            except:
                print(f'Exception occured, skipping...')
                continue
            #print(np.expand_dims(masked_ecg, axis0).astype(np.float32).shape)
            #print(sum(masked_ecg))
            preds = predict(np.expand_dims(new_ecg, axis=0), net, cuda_flag)
            score[(snum0, snum1)].append(int(preds.argmax()))
        
        preds = predict(np.expand_dims(sample['data'], axis=0), net, cuda_flag)
        score[(n_segments, n_segments)].append(int(preds.argmax()))

        new_ecg = perturb_RR(sample, perm_sample, CropDataOnly)
        preds = predict(np.expand_dims(new_ecg, axis=0), net, cuda_flag)
        score[(-1, -1)].append(int(preds.argmax()))

    labels = np.array(labels) 
    for k, v in score.items():
        # print(f'\nSegment number = {k}')
        # #fout.write(f'\nSegment number = {k}')
        scores.append({'score': np.array(v), 'segment_num': k})
        # print(sum(np.array(v) == labels) / labels.size)
        # #fout.write("\n" + str(sum(np.array(v) == labels) / labels.size)+ "\n")

    ye = np.asarray(scores)

    # print(ye)

    for i in range(ye.shape[0]):
        if (ye[i]['segment_num'] == (n_segments, n_segments)):
            print('Superpixel:', ye[i]['segment_num'])
            acc_all =  sum(ye[i]['score'] == labels[indexes]) / labels[indexes].size
            print('Accuracy:', acc_all)
            cm_all = confusion_matrix(labels[indexes], ye[i]['score'])
            print(cm_all)
            fraction0 = fraction_AF(cm_all)
            print(f'Original fraction of AF = ', print_list_with_precision(fraction0))

            print('\n\n')

    print("------------\n")
    for i in range(ye.shape[0]):
        print('Superpixel:', ye[i]['segment_num'])
        acc =  sum(ye[i]['score'] == labels[indexes]) / labels[indexes].size
        acc = acc_all - acc
        print('Accuracy:', acc)
        cm = confusion_matrix(labels[indexes], ye[i]['score'])
        print(cm)
        print(fraction_AF(cm))
        fraction = fraction_AF(cm) - fraction0
        print(f'change in fraction_AF = ', print_list_with_precision(fraction))

        print('\n\n')

