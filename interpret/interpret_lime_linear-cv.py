import torch
from torchvision import transforms
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sys.path.insert(1, './../')
from lib import load_data_saveidx as load_data
from model import MobileNet
from lime import lime_vector_linear_RR as lime_vector_linear
from utils import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')

matplotlib.rcParams['figure.figsize'] = (10,8)
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}', r'\usepackage{amsfonts}']
matplotlib.rcParams['grid.alpha'] = 0.5

matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['legend.fontsize'] = 20 
matplotlib.rcParams['axes.labelsize'] = 22

if __name__ == "__main__":
        
    print("Number of command line args =", len(sys.argv))
    n_segments = int(sys.argv[1])
    model_weights = "model_params.torch"
    home_dir = sys.argv[2].rstrip('/')
    testortrain = "test"

    cuda_flag = torch.cuda.is_available()
    print(cuda_flag)
    if cuda_flag:
        selected_gpu = torch.cuda.current_device()
        print(torch.cuda.get_device_name(selected_gpu))
        torch.cuda.set_device(selected_gpu)
        
    resultsdirname = 'results'
    print("script started!!")
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

    runs_dir = '/mnt/medx/runs/data/training2017_raligned__netarch_mobilenet__kernel_16__folds_5__seed_123__val_0.0'

    res_dict_all = []

    subdirs = glob.glob(runs_dir.rstrip('/') + '/*')
    for sidx, subdir in enumerate(subdirs):
        print(f'\nsubdir = {subdir}')

        res_all = dict([(j, dict([(i, 0) for i in range(n_segments + 1)])) for j in range(4)])
        label_counts = {j:0 for j in range(4)}

        data = load_data.Physionet2017Dataset(root_dir, transform=trans,
                kfolds=kfolds,
                split_num=splitnum,
                val_size=0,
                split_type=testortrain,
                random_seed=randomseed,
                en_cache=False,
                manual_features=en_manualfeatures,
                saveidx_dir=subdir)

        print("len of data =", len(data.labels))
        print("labels:", data.labels_list)

        n_samples = data.labels.shape[0]
        SIZE = n_samples
        print("LIME>>>")
        indexes = [i for i in range(n_samples)][:SIZE]

        net = MobileNet.MobileNet(16, 4)
        net.load_state_dict(torch.load(f'{subdir.rstrip("/")}/model/' + model_weights, lambda s, v: s))
        _ = net.eval()

        if cuda_flag:
            net.cuda()
        
        def predict_func(batch):
            # labels = np.zeros(batch.shape[0], dtype=int)
            n_sample, n_feat = batch.shape
            if cuda_flag:
                preds = net(torch.as_tensor(batch.reshape(n_sample, 1, n_feat).astype(np.float32)).cuda())
            else:
                preds = net(torch.as_tensor(batch.reshape(n_sample, 1, n_feat).astype(np.float32)))
            
            probs = F.softmax(preds, dim=1)
            return probs.detach().cpu().numpy()

        #print(res_all)
        for idx in tqdm.tqdm(indexes):
            # print(idx, len(indexes))
            # fout.write(str(idx) + "->")
            input_data = data[idx]
            #print(idx)
            true_label = input_data['label']
            # print('True label', true_label, data.labels.iloc[idx].label)

            sample = input_data['data']
            peaks = input_data['peaks']
            pred_label = np.argmax(predict_func(np.expand_dims(sample, axis=0)))

            label_counts[pred_label] += 1
            # print(f'Pred label {pred_label}')
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
                # res_pos[pred_label][k] += max(v, 0.0)
                # res_neg[pred_label][k] += max(-1 * v, 0.0)
                # res_frac_pos[pred_label][k] += int(v >= 0.0)

        res_dict_all.append(dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_all[k].items()])) for k in res_all.keys()]))

        # if sidx == 2:
        #     break

    # print("\n\n")
    # print(label_counts)

    # res_dict_all = dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_all[k].items()])) for k in res_all.keys()])
    # res_dict_pos = dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_pos[k].items()])) for k in res_pos.keys()])
    # res_dict_neg = dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_neg[k].items()])) for k in res_neg.keys()])
    # res_dict_frac_pos = dict([(k, dict([(kk, v / label_counts[k]) for kk, v in res_frac_pos[k].items()])) for k in res_frac_pos.keys()])
    
    print("\n\nres_dict_all:")
    print(res_dict_all)
    print("\n\n")
    # print(res_dict_pos)
    # print("\n\n")
    # print(res_dict_neg)
    # print("\n\n")
    # print(res_dict_frac_pos)
    
    # fout.write("\nALL:\n" + str(res_dict_all) +"\n")
    # fout.write("\nPOS:\n" + str(res_dict_pos) +"\n")
    # fout.write("\nNEG:\n" + str(res_dict_neg) +"\n")
    # fout.write("\nFRAC_POS:\n" + str(res_dict_frac_pos) +"\n")
    # fout.close()

    print(f'\n\n\n\n')
    print(res_dict_all)
    ######################
    ######################
    # SET VALUES HERE
    ######################
    # lime_scores1 = {0: 0.1694197043806493, 1: 0.02197112555027639, 2: 0.047112163220616156, 3: 0.10284437567269593, 4: 0.08311217018257017, 5: 0.005992251727225272, 6: -0.02201567389965099, 7: 0.23163448330985995, 8: 0.4833801404228275}
    lime_scores1 = [[r[1][k] for k in range(len(r[1].keys()))] for r in res_dict_all]
    print(lime_scores1)
    # exit()
    # x = sorted(list(lime_scores1.keys()))
    # y1 = np.array([lime_scores1[k] for k in x])
    # y1 /= np.max(y1)

    # lime_scores2 = {0: 0.38921963048211233, 1: -0.04763070817831745, 2: 0.03470845983156371, 3: 0.04529735198274205, 4: 0.010993622117237556, 5: -0.018384015094872516, 6: -0.005935935932433382, 7: 0.30581411564347816, 8: 0.04207242775259194}
    lime_scores2 = [[r[0][k] for k in range(len(r[1].keys()))] for r in res_dict_all]
    # y2 /= np.max(y2)
    # lime_scores3 = {0: 0.07400122537695478, 1: 0.04740624385416944, 2: 0.02795232185548423, 3: -0.003176540446731818, 4: 0.005318407656864543, 5: 0.034605076023498986, 6: 0.04729659760765272, 7: 0.09229189493505913, 8: 0.18669307028248247}
    lime_scores3 =  [[r[2][k] for k in range(len(r[1].keys()))] for r in res_dict_all]

    n_weights = np.mean(lime_scores2, axis=0)
    a_weights = np.mean(lime_scores1, axis=0)
    o_weights = np.mean(lime_scores3, axis=0)
    ######################
    ######################

    x = np.arange(len(n_weights)).astype(float)  # the label locations
    x[-1] += 0.5 # Move the RR interval group
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, n_weights, width, label='Normal Sinus Rhythm (S)', yerr=np.std(lime_scores2, axis=0))
    rects2 = ax.bar(x        , a_weights, width, label='Atrial Fibrillation (A)', yerr=np.std(lime_scores1, axis=0))
    rects3 = ax.bar(x + width, o_weights, width, label='Other Arrhythmia (O)', yerr=np.std(lime_scores3, axis=0))
    # RR interval spearation line
    ax.axvline(x[-1] - 0.75, c='k', lw=1.5, ls='-', alpha=0.8)
    ax.xaxis.grid() # horizontal grid only

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            place = 0.05 + h if h >= 0 else h - 0.05
            ax.text(rect.get_x()+rect.get_width()/2., place, '%.2f'%(h),
                    ha='center', va='bottom',rotation='vertical')
    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)

    # Labels
    ax.set_xlabel('Feature')
    ax.set_ylabel('Average LIME weight')

    ax.axhline(0, c='k', lw=1.5, ls='-', alpha=0.8)

    # Legend
    ax.legend()

    # X ticks
    ticks = [f"Seg {f:.0f}" for f in x]
    ticks[-1] = 'RR'    
    plt.xticks(x, ticks)
    plt.xticks(rotation=45, ha='right')

    plt.show()

    fig.savefig(home_dir + f'/{resultsdirname}/lime_global_nseg{n_segments}.pdf')#, bbox_inches = 'tight', pad_inches = 0)
