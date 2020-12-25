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

def predict_func(batch):
    labels = np.zeros(batch.shape[0], dtype=int)
    n_sample, n_feat = batch.shape
    if cuda_flag:
        preds = net(torch.as_tensor(batch.reshape(n_sample, 1, n_feat).astype(np.float32)).cuda())
    else:
        preds = net(torch.as_tensor(batch.reshape(n_sample, 1, n_feat).astype(np.float32)))
    
    probs = F.softmax(preds, dim=1)
    return probs.detach().cpu().numpy()