import torch
from torchvision import transforms
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(1, './../')
from lib import load_data_saveidx as load_data
from model import MobileNet

if __name__ == "__main__":
    #torch.set_num_threads(1)
    print("Number of command line args =", len(sys.argv))
    n_segments = int(sys.argv[1])
    model_weights = "model_params.torch"
    home_dir = sys.argv[2].rstrip('/')
    testortrain = "test"
        

    fout = open(home_dir + '/results/interpret_chunks_pi2_lime_linear.txt', 'w+')
    print("script started!!")
    fout.write("script started!!\n")
    print('Model weights = ' + model_weights)
    root_dir = home_dir + '/data/training2017_raligned/'

    inputsize = 9000
    en_manualfeatures = False
    trans = (
            transforms.Compose([load_data.CropWithPeak(inputsize)])
        )
    kfolds = 5
    randomseed = 123
    splitnum = 0

    data = load_data.Physionet2017Dataset(root_dir, transform=trans,
            kfolds=kfolds,
            split_num=splitnum,
            val_size=0,
            split_type=testortrain,
            random_seed=randomseed,
            en_cache=False,
            manual_features=en_manualfeatures,
            saveidx_dir=home_dir)
    print("len of data =", len(data.labels))
    fout.write("len of data = " + str(len(data.labels)) + "\n")

    print("labels:", data.labels_list)
    fout.write("labels:" + str(data.labels_list) + "\n")

    cuda_flag = torch.cuda.is_available()
    print(cuda_flag)
    ### Check for a free GPU
    if cuda_flag:
        selected_gpu = torch.cuda.current_device()
        print(torch.cuda.get_device_name(selected_gpu))
        torch.cuda.set_device(selected_gpu)

    net = MobileNet.MobileNet(16, 4)

    net.load_state_dict(torch.load(home_dir + '/model/' + model_weights, lambda s, v: s))
    _ = net.eval()
    
    if cuda_flag:
        net.cuda()
    
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

    n_samples = data.labels.shape[0]
    # scores = []
    # score = np.zeros(n_samples)
    # labels = np.zeros(n_samples)
    SIZE = n_samples
    # indexes = [i for i in range(n_samples)]
    # indexes = indexes[:SIZE]
    
        
    print("LIME>>>")
    fout.write("LIME>>>\n")
    sys.path.insert(1, home_dir + '/xai/lime_vector')
    # sys.path.insert(1, './lime_vector')
    from lime import lime_vector_linear

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
        
    # N = n_segments
    indexes = [i for i in range(n_samples)][:SIZE]
    # res_all = {j:{i:0 for i in range(n_segments)} for j in range(4)}
    res_all = dict([(j, dict([(i, 0) for i in range(n_segments + 1)])) for j in range(4)])
    res_pos = dict([(j, dict([(i, 0) for i in range(n_segments + 1)])) for j in range(4)])
    res_neg = dict([(j, dict([(i, 0) for i in range(n_segments + 1)])) for j in range(4)])
    res_frac_pos = dict([(j, dict([(i, 0) for i in range(n_segments + 1)])) for j in range(4)])
    label_counts = {j:0 for j in range(4)}
    #print(res_all)
    for idx in indexes:
        print(idx, len(indexes))
        fout.write(str(idx) + "->")
        input_data = data[idx]
        #print(idx)
        true_label = input_data['label']
        print('True label', true_label, data.labels.iloc[idx].label)

        sample = input_data['data']
        peaks = input_data['peaks']
        pred_label = np.argmax(predict_func(np.expand_dims(sample, axis=0)))
        label_counts[pred_label] += 1
        print(f'Pred label {pred_label}')
        # if sample.shape[0] > 9000:
        #     sample = sample[:9000]
        segment = partial(segment_vec, peaks=input_data['peaks'], SEG=n_segments)

        explainer = lime_vector_linear.LimeVectorExplainer()
        s = explainer.explain_instance(
            sample,
            predict_func,
            segmentation_fn=segment,
            hide_color=0,
            num_samples=100,
            peaks=peaks
        )
        exp = s.local_exp[pred_label]

        # print(f'Explanation = {exp}')

        for k, v in exp:
            #print(k, v)
            res_all[pred_label][k] += v
            res_pos[pred_label][k] += max(v, 0.0)
            res_neg[pred_label][k] += max(-1 * v, 0.0)
            res_frac_pos[pred_label][k] += int(v >= 0.0)

    print("\n\n")
    print(label_counts)

    res_dict_all = dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_all[k].items()])) for k in res_all.keys()])
    res_dict_pos = dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_pos[k].items()])) for k in res_pos.keys()])
    res_dict_neg = dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_neg[k].items()])) for k in res_neg.keys()])
    res_dict_frac_pos = dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_frac_pos[k].items()])) for k in res_frac_pos.keys()])
    
    print("\n\n")
    print(res_dict_all)
    print("\n\n")
    print(res_dict_pos)
    print("\n\n")
    print(res_dict_neg)
    print("\n\n")
    print(res_dict_frac_pos)
    
    fout.write("\nALL:\n" + str(res_dict_all) +"\n")
    fout.write("\nPOS:\n" + str(res_dict_pos) +"\n")
    fout.write("\nNEG:\n" + str(res_dict_neg) +"\n")
    fout.write("\nFRAC_POS:\n" + str(res_dict_frac_pos) +"\n")
    fout.close()


    def normalize_dict(scores, m):
        return dict([(k, v / m) for k, v in scores.items()])

    def normi(i, alpha, beta):
        return ((i % 2) + 1) / 2
    #     return np.abs(i - alpha) / beta

    colors = dict([(i, (0.1, 0.8 * normi(i, -5, 20) , 0.3)) for i in range(n_segments)])
    def plot_fig_concat(ax, overlay_idx, scores_dict, class_num, start_peak):
        scores = scores_dict[class_num]
        #m = max([max(scores_dict[k].values()) for k in scores_dict.keys()])
        m = max(scores.values())
        scores = normalize_dict(scores, m)#{k:v/max(scores.values()) for k, v in scores.items()}
        input_data = data[overlay_idx]
        segment = segment_vec(input_data['data'], input_data['peaks'], n_segments)
        left = input_data['peaks'][start_peak]
    #     mid =  input_data['peaks'][start_peak + 1]
        right = input_data['peaks'][start_peak + 1]
    #     left = int((left + mid) / (1.5))
    #     right = int((right + mid) / (1.5))
        segment = segment[left:right]
        dummy = [1] * len(segment)
        for i in range(len(segment)):
            dummy[i] *= scores[segment[i]]
        #plt.figure()
        ax.bar(np.arange(len(dummy)), dummy, width=1, color=[colors[segment[i]] for i in range(len(dummy))])
    #     ax.set_yticks([])
        ax.set_xticks([])
        ax.plot(input_data['data'][left:right], color='r')

    def plot_fig(overlay_idx, scores_dict, class_num, start_peak):
        scores = scores_dict[class_num]
        #m = max([max(scores_dict[k].values()) for k in scores_dict.keys()])
        scores = scores_dict[class_num]
        #m = max([max(scores_dict[k].values()) for k in scores_dict.keys()])
        m = max(scores.values())
        scores = normalize_dict(scores, m)#{k:v/max(scores.values()) for k, v in scores.items()}
        input_data = data[overlay_idx]
        segment = segment_vec(input_data['data'], input_data['peaks'], n_segments)
        left = input_data['peaks'][start_peak]
    #     mid =  input_data['peaks'][start_peak + 1]
        right = input_data['peaks'][start_peak + 1]
        segment = segment[left:right]
        dummy = [1] * len(segment)
        for i in range(len(segment)):
            dummy[i] *= scores[segment[i]]
        plt.figure()
        # plt.yticks([])
        plt.ylabel('Average segment importance (normalized)')
        plt.xticks([])
        plt.bar(np.arange(len(dummy)), dummy, width=1, color=[colors[segment[i]] for i in range(len(dummy))])
        plt.plot(data[overlay_idx]['data'][left:right] / np.max(data[overlay_idx]['data'][left:right]), color='r')
    
    start_peak = 10
    overlay_idx = 250

    plot_fig(overlay_idx, res_dict_all, 1, start_peak)
    plt.savefig(home_dir + '/xai_results/LIME_linear_A_ALL_pi2.png')

    plot_fig(overlay_idx, res_dict_pos, 1, start_peak)
    plt.savefig(home_dir + '/xai_results/LIME_linear_A_POS_pi2.png')

    plot_fig(overlay_idx, res_dict_neg, 1, start_peak)
    plt.savefig(home_dir + '/xai_results/LIME_linear_A_NEG_pi2.png')

    plot_fig(overlay_idx, res_dict_frac_pos, 1, start_peak)
    plt.savefig(home_dir + '/xai_results/LIME_linear_A_FRAC_POS_pi2.png')

    # plot_fig(overlay_idx, SM_ALL, 1, start_peak)
    # plt.savefig(home_dir + '/xai_results/SM_A_ALL_' + str(n_segments) + '_' + model_weights + '_' + testortrain + '.png')

    
    f, ax = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3)
    for i in range(4):
        one = i // 2
        two = i % 2
        print(one, two)
        plot_fig_concat(ax[one][two], overlay_idx, res_dict_all, i, start_peak)
        ax[one][two].set_title("LIME normalized segment importance for class: " + str(data.labels_list[i]))
    # plt.savefig(f'{home_dir}/xai_results/LIME_res6_all_{n_segments}_{model_weights}_{testortrain}.png')
    plt.savefig(home_dir + '/xai_results/LIME_linear_pi2_all.png')

    f, ax = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3)
    for i in range(4):
        one = i // 2
        two = i % 2
        print(one, two)
        plot_fig_concat(ax[one][two], overlay_idx, res_dict_pos, i, start_peak)
        ax[one][two].set_title("LIME normalized segment importance for class: " + str(data.labels_list[i]))
    plt.savefig(home_dir + '/xai_results/LIME_linear_pi2_pos.png')

    f, ax = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3)
    for i in range(4):
        one = i // 2
        two = i % 2
        print(one, two)
        plot_fig_concat(ax[one][two], overlay_idx, res_dict_neg, i, start_peak)
        ax[one][two].set_title("LIME normalized segment importance for class: " + str(data.labels_list[i]))
    plt.savefig(home_dir + '/xai_results/LIME_linear_pi2_neg.png')

    f, ax = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3)
    for i in range(4):
        one = i // 2
        two = i % 2
        print(one, two)
        plot_fig_concat(ax[one][two], overlay_idx, res_dict_frac_pos, i, start_peak)
        ax[one][two].set_title("LIME normalized segment importance for class: " + str(data.labels_list[i]))
    plt.savefig(home_dir + '/xai_results/LIME_linear_pi2_frac_pos.png')
