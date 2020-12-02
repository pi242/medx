#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:22:13 2017

@author: Matteo Gadaleta
"""

import os
import numpy as np
import pywt
import wfdb
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from tqdm import tqdm
from shutil import copyfile
from lib import terma
#from pmtools import plotting as pmt, notify


def normalize(source_data_dir, output_data_dir):
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    
    file_list = [f for f in os.listdir(source_data_dir) if f.endswith('.mat')]

    for filename in tqdm(file_list):
        data = loadmat(os.path.join(source_data_dir, filename))['val']
        normalized_data = (data - np.mean(data)) / np.std(data)
        savemat(os.path.join(output_data_dir, filename), {'val': normalized_data})

    copyfile(os.path.join(source_data_dir, 'REFERENCE-v3.csv'), 
             os.path.join(output_data_dir, 'REFERENCE-v3.csv'))


def rescale(inp, out):
    if not os.path.exists(out):
        os.makedirs(out)
    
    file_list = [f for f in os.listdir(inp) if f.endswith('.mat')]

    for filename in tqdm(file_list):
        
        header_file = filename.replace('.mat','')
        sig, fields = wfdb.rdsamp(os.path.join(inp, header_file))
        sig = sig.transpose()
        
        savemat(os.path.join(out, filename), {'val': sig})

    copyfile(os.path.join(inp, 'REFERENCE-v3.csv'), 
             os.path.join(out, 'REFERENCE-v3.csv'))


def pad_signal(sig, target_len=9000):
    lpad = int(np.ceil((target_len - len(sig[0])) / 2))
    rpad = int(np.floor((target_len - len(sig[0])) / 2))
    
    padded_sig = np.pad(sig[0], (lpad, rpad), 'edge')
    return np.expand_dims(padded_sig, 0)


def unpad_signal(padded_sig, target_len):
    lpad = int(np.ceil((len(padded_sig[0])-target_len) / 2))
    rpad = int(np.floor((len(padded_sig[0])-target_len) / 2))
    return padded_sig[:, lpad:-rpad] if rpad>0 else padded_sig[:, lpad:]


def filtering(inp, out):
    if not os.path.exists(out):
        os.makedirs(out)
    
    file_list = [f for f in os.listdir(inp) if f.endswith('.mat')]

    
    for filename in tqdm(file_list):
        
        header_file = filename.replace('.mat','')
        sig, fields = wfdb.rdsamp(os.path.join(inp, header_file))
        sig = sig.transpose()
        
        wavelet = pywt.Wavelet('db9')
        
        siglen = len(sig[0])
        en_pad = True if siglen < 9000 else False
        
        if en_pad:
            sig = pad_signal(sig, target_len=9000)
            
        coeffs = pywt.wavedec(sig, wavelet, mode='constant', level=9)
        coeffs[0] *= 0
        filtered_data = pywt.waverec(coeffs, wavelet)
        
        if en_pad:
            filtered_data = unpad_signal(filtered_data, target_len=siglen)
        
        savemat(os.path.join(out, filename), {'val': filtered_data})
        

    copyfile(os.path.join(inp, 'REFERENCE-v3.csv'), 
             os.path.join(out, 'REFERENCE-v3.csv'))


def r_intervals(inp, out):
    if not os.path.exists(out):
        os.makedirs(out)

    copyfile(os.path.join(inp, 'REFERENCE-v3.csv'), 
             os.path.join(out, 'REFERENCE-v3.csv'))
    
    file_list = [f for f in os.listdir(inp) if f.endswith('.mat')]

    for filename in tqdm(file_list):
        
        header_file = filename.replace('.mat','')
        sig = loadmat(os.path.join(inp, filename))['val']

        fs = 300
        f1 = 8
        f2 = 20
        w1 = 97
        w2 = 611
        beta = 8
        en_plot = False

        res = np.asarray(terma.terma_detector(sig[0], fs=fs, f1=f1, f2=f2, w1=w1, w2=w2, beta=beta, en_plot=en_plot))
        if len(res) < 2:
            # could not find any peaks - remove sample
            print('Removing', header_file)
            with open(os.path.join(out, 'REFERENCE-v3.csv'), 'r+') as labels:
                lines = [l for l in labels if not l.startswith(header_file)]
                labels.seek(0)
                labels.truncate()
                labels.write('\n'.join(lines))

            continue
                

        filtered_data = ((res[1:] - res[:-1]) / fs).reshape(1, -1)

        savemat(os.path.join(out, filename), {'val': filtered_data})  


def r_align(inp, out):
    if not os.path.exists(out):
        os.makedirs(out)

    copyfile(os.path.join(inp, 'REFERENCE-v3.csv'), 
             os.path.join(out, 'REFERENCE-v3.csv'))
    
    file_list = [f for f in os.listdir(inp) if f.endswith('.mat')]

    for filename in tqdm(file_list):
        
        header_file = filename.replace('.mat','')
        sig = loadmat(os.path.join(inp, filename))['val']

        fs = 300
        f1 = 8
        f2 = 20
        w1 = 97
        w2 = 611
        beta = 8
        en_plot = False

        res = np.asarray(terma.terma_detector(sig[0], fs=fs, f1=f1, f2=f2, w1=w1, w2=w2, beta=beta, en_plot=en_plot), dtype=int)
        if len(res) < 2:
            # could not find any peaks - remove sample
            print('Removing', header_file)
            with open(os.path.join(out, 'REFERENCE-v3.csv'), 'r+') as labels:
                lines = [l for l in labels if not l.startswith(header_file)]
                labels.seek(0)
                labels.truncate()
                labels.write('\n'.join(lines))
            
            continue 
                

        filtered_data = sig[0, res[0]:res[-1]]
        res -= res[0]

        savemat(os.path.join(out, filename), {'val': filtered_data, 'peaks': res})      


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store_true', help='Normalize data')
    parser.add_argument('-r', action='store_true', help='Rescale data')
    parser.add_argument('-f', action='store_true', help='Filter data')
    parser.add_argument('-w', action='store_true', help='Extract r intervals data')
    parser.add_argument('-a', action='store_true', help='Align on r peaks')
    parser.add_argument('-s', help='source_dir')
    args = parser.parse_args()
    
    if not args.s:
        print("Specify source directory for input data!!")
        
    if args.n:
        #%% Data Normalization
        source_data_dir = str(args.s) + 'data/training2017'
        output_data_dir = str(args.s) + 'data/training2017_normalized'
        print('Normalizing')
        normalize(source_data_dir, output_data_dir)

    if args.r:
        #%% Data Rescaling
        source_data_dir = str(args.s) + 'data/training2017'
        output_data_dir = str(args.s) + 'data/training2017_rescaled'
        print('Rescaling')
        rescale(source_data_dir, output_data_dir)
    
    if args.f:    
        #%% Data Filtering
        source_data_dir = str(args.s) + 'data/training2017'
        output_data_dir = str(args.s) + 'data/training2017_filtered'
        print('Filtering')
        filtering(source_data_dir, output_data_dir)
    
    if args.w:
        #from telegram.ext import Updater
        #%% Data Filtering
        source_data_dir = str(args.s) + 'data/training2017_filtered'
        output_data_dir = str(args.s) + 'data/training2017_rwaves'
        print('R Waves')
        #msg = notify.Messager()
        #msg.send('Extraction r intervals...')
        r_intervals(source_data_dir, output_data_dir)
        #msg.send('Done')

    if args.a:
        #%% Data Filtering
        source_data_dir = str(args.s) + 'data/training2017_filtered'
        output_data_dir = str(args.s) + 'data/training2017_raligned'
        print('R Align')
        r_align(source_data_dir, output_data_dir)
