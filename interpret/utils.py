import torch
import numpy as np
import torch.nn.functional as F
import math

def print_list_with_precision(arr):
    return ['%.2f' % elem for elem in arr]

def segmentation_fn(vec, peaks, active_segments):
    current = 0 
    n_segments = len(active_segments)
    segments = np.zeros_like(vec, dtype=int)

    for next_peak in peaks:
        size = next_peak - current
        
        segment_size = size // n_segments
        rest = size % n_segments 
        
        for i in range(n_segments):
            if active_segments[i] > 0:
                s = current + i*segment_size
                e = s + segment_size
                segments[s:e] = 1
        
        if rest > 0 and active_segments[i] > 0:
            segments[e:e+rest] = 1
        
        current = next_peak
        
    return segments, n_segments


def segment_vec(vec, peaks, SEG):
    total = np.zeros(vec.shape)
    #print(total.shape)
    for i in range(SEG):
        p = [0] * SEG
        p[i] = 1
        #print(p)
        a, b = segmentation_fn(vec, peaks, p)
        total += np.array(a) * (i)
    return total


def predict(batch, net, cuda_flag):
    # labels = np.zeros(batch.shape[0], dtype=int)
    n_sample, n_feat = batch.shape
    if cuda_flag:
        preds = net(torch.as_tensor(batch.reshape(n_sample, 1, n_feat).astype(np.float32)).cuda())
    else:
        preds = net(torch.as_tensor(batch.reshape(n_sample, 1, n_feat).astype(np.float32)))
    
    probs = F.softmax(preds, dim=1)
    return probs.detach().cpu().numpy()


def get_segment_idxs(vec, peaks, n_segments):
    peaks = list(peaks)
    idxs = []
    current = peaks[0]
    for next_peak in peaks[1:]:
        size = next_peak - current
        segment_size = size // n_segments
        rest = size % n_segments

        for i in range(n_segments):
            s = int(current + i*segment_size)
            idxs.append({'segnum':i, 'idx':s})
        current = next_peak
    idxs.append({'segnum':0, 'idx':current})
    return idxs

def masking_linear_interpolation_cont(vec, peaks, active_segments):
    # print(f'active_segments = {active_segments}')
    if  np.sum(active_segments) == len(active_segments):
        return vec.copy()
    # current = 0 
    n_segments = len(active_segments)
    masked_ecg = vec.copy()
    seg_idxs = get_segment_idxs(vec, peaks, n_segments)
    # segi = 0
    s = e = seg_idxs[0]['idx']
    ongoing = False
    for i in range(len(seg_idxs)):
        # print(seg_idxs[i])
        if active_segments[seg_idxs[i]['segnum']] == 1:
            if ongoing == True:
                e = seg_idxs[i]['idx']
                # print(f'In 1: s = {s}, e = {e}')
                #Linear interpolation code
                if e - 1 < 0:
                    continue
                dummy_ip = [masked_ecg[s], masked_ecg[e - 1]]
                if dummy_ip[0] > dummy_ip[1]:
                    dummy_ip = dummy_ip[::-1]
                    dummy = np.interp(np.linspace(dummy_ip[0], dummy_ip[1], e - s), dummy_ip, dummy_ip)[::-1]
                else:
                    dummy = np.interp(np.linspace(dummy_ip[0], dummy_ip[1], e - s), dummy_ip, dummy_ip)
                masked_ecg[s:e] = dummy
            ongoing = False
        else:
            if ongoing == False:
                s = seg_idxs[i]['idx']
                ongoing = True
            # print(f'In 0: s = {s}, e = {e}')
    return masked_ecg

def fraction_AF(confmat):
    total = np.sum(confmat, axis=1)
    af = confmat[:, 1]
    return af.astype(float) / total

def resample_by_interpolation(signal, target_length):
    # output_fs = len(signal)
    # scale = output_fs / input_fs
    # # calculate new length of sample
    # n = round(len(signal) * scale)
    if target_length == 0 or len(signal) == 0:
        return []
    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, target_length, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    # print(f'In resample, length = {len(resampled_signal)}')
    return resampled_signal

def CropDataOnly(sample, num_samples=9000):
    # print(f'In CropData, num_samples={num_samples}')
    if len(sample) >= num_samples:
        start_idx = np.random.randint(len(sample) - num_samples + 1)
        sample = sample[start_idx : start_idx + num_samples]
    else:
        left_pad = int(np.ceil((num_samples - len(sample)) / 2))
        right_pad = int(np.floor((num_samples - len(sample)) / 2))
        sample = np.pad(sample, (left_pad, right_pad), "constant")

    return sample

def align(sample):
    vec = sample['data']
    peaks = sample['peaks']
    current = 0 
    aligned = []
    size = int(math.ceil(np.median(peaks[1:] - peaks[:-1])))
    peaks = list(peaks)
    peaks.append(len(vec) - 1)
    # print(peaks)
    for next_peak in peaks:
        segment = vec[current:next_peak]
        # print(current, next_peak, len(segment))
        resampled = resample_by_interpolation(segment, size)
        aligned.extend(resampled)
        
        current = next_peak
        
    return np.asarray(aligned)