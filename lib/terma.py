#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:46:48 2018

@author: Matteo Gadaleta
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#from pmtools import plotting as pmt


def neraest_max(sig, idx):

    while True:
        if sig[np.min([len(sig) - 1, idx + 1])] > sig[idx]:
            idx += 1
        elif sig[np.max([0, idx - 1])] > sig[idx]:
            idx -= 1
        else:
            break

    return idx


def terma_detector(sig, fs, f1=8, f2=20, w1=97, w2=611, beta=8, en_plot=False):

    ### Filter
    nyq = fs / 2
    b, a = signal.butter(3, [f1 / nyq, f2 / nyq], "bandpass")
    x_filt = signal.filtfilt(b, a, sig)

    ### Enhancing
    x_en = x_filt ** 2

    ### Event moving average
    ma_event = np.convolve(x_en, np.ones((w1,)) / w1, mode="same")
    ma_cycle = np.convolve(x_en, np.ones((w2,)) / w2, mode="same")

    ### Bias
    bias_window_len = np.min([len(x_en), fs * 20])
    z = np.convolve(x_en, np.ones((bias_window_len,)) / (bias_window_len), mode="same")
    z[: int(bias_window_len / 2)] = z[int(bias_window_len / 2) + 1]
    z[-int(bias_window_len / 2) :] = z[-int(bias_window_len / 2) - 1]

    alpha = beta / 100 * z
    thr1 = ma_cycle + alpha

    ### Blocks of interest
    blocksofinterest = ma_event > thr1

    blocksofinterest_shifted = list(blocksofinterest[1:])
    blocksofinterest_shifted.append(blocksofinterest_shifted[-1])
    blocksofinterest_shifted = np.array(blocksofinterest_shifted)

    edge_mask = blocksofinterest ^ blocksofinterest_shifted

    starting_points = list(np.where((edge_mask * (~blocksofinterest)) == 1)[0])
    ending_points = list(np.where((edge_mask * blocksofinterest) == 1)[0])

    if np.min(starting_points) > np.min(ending_points):
        ending_points.remove(np.min(ending_points))
    if np.max(starting_points) > np.max(ending_points):
        starting_points.remove(np.max(starting_points))

    peaks = []
    for sp, ep in zip(starting_points, ending_points):
        peak = np.argmax(x_en[sp:ep])
        # Fine tuning
        peak = neraest_max(np.abs(sig[sp:ep]), peak)
        peaks.append(peak + sp)

    if en_plot:
        #fig, ax = pmt.tight_subplots(8, figsize=(20, 12), sharex=True)
        fig, ax = plt.sub_plots(8, figsize=(20, 12), sharex = True)
        plt.tight_layout()
        
        ax[0].plot(sig)
        ax[0].set_ylabel("sig")
        ax2 = ax[0].twinx()
        ax2.plot(blocksofinterest, ls="--", lw=0.5, color="g")
        for peak in peaks:
            ax[0].axvline(peak, ls="--", color="r")

        ax[1].plot(x_filt)
        ax[1].set_ylabel("x_filt")
        ax[2].plot(x_en)
        ax[2].set_ylabel("x_en")
        ax[3].plot(ma_event)
        ax[3].set_ylabel("ma_event")
        ax[4].plot(ma_cycle)
        ax[4].set_ylabel("ma_cycle")
        ax[5].plot(z)
        ax[5].set_ylabel("z")
        ax[6].plot(alpha)
        ax[6].set_ylabel("alpha")
        ax[7].plot(thr1)
        ax[7].set_ylabel("thr1")
        [a.grid() for a in ax]

        #fig, ax = pmt.tight_subplots(1, figsize=(20, 12), sharex=True)
        fig, ax =plt.sub_plots(1, figsize=(20, 12), sharex=True)
        plt.tight_layout()
        
        ax.plot(sig)
        for sp, ep in zip(starting_points, ending_points):
            ax.axvspan(sp, ep, color="g", alpha=0.2)
        for peak in peaks:
            ax.axvline(peak, ls="--", color="r")

        plt.show()

    return peaks


if __name__ == "__main__":

    from load_data import Physionet2017Dataset

    root_dir = "../data/training2017_filtered"
    train_dataset = Physionet2017Dataset(
        root_dir,
        transform=None,
        preprocessing=None,
        en_cache=False,
        kfolds=1,
        split_num=0,
        split_type="train",
        random_seed=123,
    )

    sig = train_dataset[2]["data"]

    fs = 300
    f1 = 8
    f2 = 20
    w1 = 97
    w2 = 611
    beta = 8
    en_plot = True

    plt.close("all")
    terma_detector(sig, fs=fs, f1=f1, f2=f2, w1=w1, w2=w2, beta=beta, en_plot=en_plot)
