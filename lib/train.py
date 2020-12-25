#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:53:25 2017

@author: Matteo Gadaleta
"""
import torch
from sklearn import metrics
import matplotlib as mpl

mpl.use("Agg")

#import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import os
import csv
import sys
import shutil
import inspect

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from tensorboardX import SummaryWriter
from load_data_saveidx import Physionet2017Dataset, Crop, ToTensor

#from ResNet import ResNet
#from ResNetv2 import ResNetv2
#from ConvNet import VGG, AlexNet
#from Inception import Inception
#from ResInception import ResInception
import sys
sys.path.insert(1, './../model')
from MobileNet import MobileNet
from ConvNet import *
from ResNet import *
from ResNetv2 import *
from Inception import *
#from MLP import MLP, MLP2
from collections import OrderedDict
from utils import get_gpu_memory_map
#import pmtools.notify as _notify

#_msg = _notify.Messager()


def train_single_epoch(net, train_dataloader, n_epoch, loss_fn, optimizer):

    num_batches = int(np.ceil(len(train_dataloader.dataset) / train_dataloader.batch_size))

    ### Training mode
    net.train()
    ct = time.time()
    for idx, sample_batched in enumerate(train_dataloader):

        ### Load batch
        data_batch = Variable(sample_batched["data"].float())
        label_batch = Variable(sample_batched["label"].squeeze().long())

        ### Move to GPU
        if cuda_flag:
            data_batch = data_batch.cuda()
            label_batch = label_batch.cuda()

        # Time Check
        loading_time = time.time() - ct

        ct = time.time()

        ### Evaluate output
        output_batch = net(data_batch)
        loss = loss_fn(output_batch, label_batch)

        ### Update network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(net.parameters(), args.clipgrad)
        optimizer.step()

        ### Some performance metrics
        loss, accuracy, precision, recall, fscore, support = ev_performance(
            output_batch, label_batch, loss_fn
        )

        # Time check
        updating_time = time.time() - ct

        ### Print report
        print(
            "Epoch %d/%d - It %d/%d - Accuracy = %f - Loss = %f - Training Time = %f(%.1f%%) - Loading Time = %f"
            % (
                n_epoch + 1,
                num_epochs,
                idx + 1,
                num_batches,
                accuracy,
                loss,
                updating_time,
                updating_time / (updating_time + loading_time) * 100.0,
                loading_time,
            )
        )

        print("\t lr = %f" % (optimizer.param_groups[0]["lr"]))

        print("\t Loss      = %f" % loss)
        print("\t Accuracy  = %f" % accuracy)
        print("\t Precision = %s" % str(precision))
        print("\t Recall    = %s" % str(recall))
        print("\t Fscore    = %s - Average = %f" % (str(fscore), np.mean(fscore)))
        print("\t Support   = %s" % str(support))

        ### Memory managemen
        # if cuda_flag:
        #     print("GPU Memory: ", get_gpu_memory_map())

        ct = time.time()


def test(net, test_dataloader, loss_fn):

    with torch.no_grad():

        test_start_time = time.time()

        num_batches = int(np.ceil(len(test_dataloader.dataset) / test_dataloader.batch_size))

        ### Evaluation mode
        net.eval()
        test_output = Variable(torch.FloatTensor())
        test_labels = Variable(torch.LongTensor())
        cuda_time = 0
        ct = time.time()
        for idx, sample_batched in enumerate(test_dataloader):

            ### Load batch
            data_batch = Variable(sample_batched["data"].float(), volatile=True)
            label_batch = Variable(sample_batched["label"].squeeze().long())

            ### Move to GPU
            if cuda_flag:
                cuda_ct = time.time()
                data_batch = data_batch.cuda()
                label_batch = label_batch.cuda()
                cuda_time = time.time() - cuda_ct

            # Time Check
            loading_time = time.time() - ct

            ct = time.time()

            ### Evaluate output
            output_batch = net(data_batch)

            # Time check
            testing_time = time.time() - ct

            test_output = (
                torch.cat([test_output, output_batch]) if len(test_output) > 0 else output_batch
            )
            test_labels = (
                torch.cat([test_labels, label_batch]) if len(test_labels) > 0 else label_batch
            )

            ### Print report
            print(
                "(TEST) Epoch %d/%d - Testing batch %d/%d - Testing Time = %f(%.1f%%) - Loading Time = %f - Cuda Time = %f"
                % (
                    n_epoch + 1,
                    num_epochs,
                    idx + 1,
                    num_batches,
                    testing_time,
                    testing_time / (testing_time + loading_time) * 100.0,
                    loading_time,
                    cuda_time,
                )
            )

            ct = time.time()

        test_time = time.time() - test_start_time

        loss, accuracy, precision, recall, fscore, support = ev_performance(
            test_output, test_labels, loss_fn
        )

        print("Loss      = %f" % loss)
        print("Accuracy  = %f" % accuracy)
        print("Precision = %s" % str(precision))
        print("Recall    = %s" % str(recall))
        print("Fscore    = %s - Average = %f" % (str(fscore), np.mean(fscore)))
        print("Support   = %s" % str(support))

        return loss, accuracy, precision, recall, fscore, support, test_time


def ev_performance(output, labels, loss_fn=None):

    if loss_fn == None:
        loss = 0
    else:
        loss = float(loss_fn(output, labels).cpu().data.numpy())

    # 
    pred = output.data.max(1)[1].cpu().numpy()
    labels = labels.cpu().data.numpy()

    accuracy = metrics.accuracy_score(labels, pred)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        labels, pred, labels=[0, 1, 2, 3]
    )

    support = np.array(support, dtype=float)

    return loss, accuracy, precision, recall, fscore, support


if __name__ == "__main__":
    print("training started!!")
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="AF Classification ResNet Training")

    parser.add_argument(
        "--epochs", default=2000, type=int, help="number of total epochs to run"
    )

    parser.add_argument(
        "--batchsize", default=50, type=int, help="mini-batch size (default: 2)"
    )

    parser.add_argument(
        "--inputsize", default=9000, type=int, help="Number of input samples. (default: 9000)"
    )

    parser.add_argument(
        "--optimizer", default="sgd", type=str, help="sgd, rmsprop or adam"
    )
    
    parser.add_argument(
        "--sched", default="step", type=str, help="step or plat"
    )
    
    parser.add_argument(
        "--lr", "--learningrate", default=0.1, type=float, help="initial learning rate (0.1)"
    )
    
    parser.add_argument(
        "--lrgamma",
        default=0.5,
        type=float,
        help="Multiplicative factor of learning rate decay (0.1)",
    )
    
    parser.add_argument(
        "--lrdecaystep", default=150, type=int, help="Period of learning rate decay (100)"
    )
    
    parser.add_argument(
        "--droprate", default=0.3, type=float, help="dropout probability (default: 0.0)"
    )
    
    parser.add_argument(
        "--numworkers", default=4, type=int, help="parallel dataloading threads (default: 0)"
    )
    
    parser.add_argument(
        "--weightdecay", default=0.0005, type=float, help="Weight decay (default: 0)"
    )
    
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="Weight decay (default: 0.9)"
    )

    parser.add_argument(
        "--clipgrad",
        default=2,
        type=float,
        help="L2 norm gradient clipping. 0 to disable. (default: 0)",
    )
    
    parser.add_argument(
        "--noiselossscaling",
        default=0.1,
        type=float,
        help="Factor to multiply to the noise loss weight. (default: 1)",
    )

    parser.add_argument(
        "--datadir", default="training2017_filtered", type=str, help="Data directory name"
    )

    parser.add_argument(
        "--netarchitecture",
        default="mobilenet",
        type=str,
        help="Network architecture (possible values: resnet, resnetv2, alexnet, vgg1, vgg3, inception, resinception, mobilenet)",
    )
    
    parser.add_argument(
        "--manualfeatures",
        default=0,
        type=int,
        help="0 for raw data, 1 for manual extracted features as input (default: 0)",
    )
    
    parser.add_argument(
        "--kernelsize", default=16, type=int, help="1xN convolutional kernel size (default: 16)"
    )

    parser.add_argument(
        "--kfolds",
        default=5,
        type=int,
        help="Number of folds for a K-folds validation split (default: 1)",
    )
    
    parser.add_argument(
        "--splitnum", default=0, type=int, help="Fold selection index (default: 0)"
    )
    
    parser.add_argument(
        "--randomseed", default=123, type=int, help="Random Seed (default: 123)"
    )

    parser.add_argument(
        "--logdir",
        default="",
        type=str,
        help="Text that prepends to the default log dir (optional)",
    )

    parser.add_argument(
        "--val", default=0, type=float, help="Fraction of validation data (default: 0.2)"
    )

    parser.set_defaults(augment=True)

    try:

        ### Parser
        args = parser.parse_args()

        num_epochs = args.epochs
        batch_size = args.batchsize
        lr = args.lr
        lrgamma = args.lrgamma
        lrdecaystep = args.lrdecaystep
        droprate = args.droprate
        num_workers = args.numworkers
        weight_decay = args.weightdecay
        momentum = args.momentum

        datadir = args.datadir

        netarchitecture = args.netarchitecture
        en_manualfeatures = True if args.manualfeatures == 1 else False
        kernelsize = args.kernelsize

        kfolds = args.kfolds
        splitnum = args.splitnum
        randomseed = args.randomseed

        logdir_comment = args.logdir

        ### Log dir
        logdir = (
            "../runs/"
            + (("%s___" % logdir_comment) if len(logdir_comment) else "")
            + "epochs_%d" % num_epochs
            + "__batchsize_%d" % batch_size
            + ("__inputsize_%d" % args.inputsize if args.inputsize != 9000 else "")
            + "__opt_%s" % args.optimizer
            + "__lr_%s" % str(lr)
            + ("__plateau_" if args.sched == "plat" else "")
            + ("__lrgamma_%s" % str(lrgamma) if args.sched != "plat" else "")
            + ("__lrdecaystep_%d" % lrdecaystep if args.sched != "plat" else "")
            + "__droprate_%s" % str(droprate)
            + "__wdec_%s" % str(weight_decay)
            + "__mom_%s" % str(momentum)
            + "__noisescal_%s" % str(args.noiselossscaling)
            + "__clipgrad_%s" % str(args.clipgrad)
            + "__datadir_%s" % datadir
            + "__netarch_%s" % netarchitecture
            + ("__manualfeatures" if en_manualfeatures else "")
            + "__kernel_%d" % kernelsize
            + "__folds_%d" % kfolds
            + "__seed_%d" % randomseed
            + ("__val_%s" % str(args.val) if args.val != 0.2 else "")
            + "/fold_%d" % splitnum
        )

        print(logdir)
        #_msg.send(logdir)
        #time.sleep(np.random.randint(3, 30))
        if os.path.exists(logdir):
        	#print("Directory already exists for" + str(logdir))
            #_msg.send("Directory already exists for %s" % logdir_comment, logdir)
            sys.exit("Logdir already exists... SKIP!!!")
        else:
            os.makedirs(logdir)

        ### Define net
        cuda_flag = torch.cuda.is_available()

        ### Check for a free GPU
        if cuda_flag:
            selected_gpu = torch.cuda.current_device()
            print(torch.cuda.get_device_name(selected_gpu))
            torch.cuda.set_device(selected_gpu)
            # gpu_memory_map_used, gpu_memory_map_total = get_gpu_memory_map()
            # selected_gpu = None
            # for (device_num, used_mem), (_, total_mem) in zip(
            #     gpu_memory_map_used.items(), gpu_memory_map_total.items()
            # ):
            #     free_mem = total_mem - used_mem
            #     if free_mem > 7000:
            #         selected_gpu = device_num
            #         break
            # if selected_gpu == None:
            #     raise Exception("No GPU available for %s" % logdir_comment, logdir)
            # # Set GPU
            # print("GPU selected: ", selected_gpu)
            # torch.cuda.set_device(selected_gpu)

        def save_modelfile(model, logdir):
            model_sourcefile = inspect.getsourcefile(model)
            model_filename = os.path.split(model_sourcefile)[-1]
            shutil.copyfile(model_sourcefile, os.path.join(logdir, model_filename))

        if netarchitecture == "resnet":
            net = ResNet(kernel_size=kernelsize, dropout_prob=droprate, output_size=4)
            save_modelfile(ResNet, logdir)
        elif netarchitecture == "resnetv2":
            net = ResNetv2(kernel_size=kernelsize, dropout_prob=droprate, output_size=4)
            save_modelfile(ResNetv2, logdir)
        elif netarchitecture == "vgg1":
            net = VGG(kernel_size=kernelsize, dropout_prob=droprate, output_size=4, en_conv1x1=True)
            save_modelfile(VGG, logdir)
        elif netarchitecture == "vgg3":
            net = VGG(
                kernel_size=kernelsize, dropout_prob=droprate, output_size=4, en_conv1x1=False
            )
            save_modelfile(VGG, logdir)
        elif netarchitecture == "inception":
            net = Inception(kernel_size=kernelsize, dropout_prob=droprate, output_size=4)
            save_modelfile(Inception, logdir)
        elif netarchitecture == "resinception":
            net = ResInception(kernel_size=kernelsize, dropout_prob=droprate, output_size=4)
            save_modelfile(ResInception, logdir)
        elif netarchitecture == "mobilenet":
            net = MobileNet(kernel_size=kernelsize, dropout_prob=droprate, output_size=4)
            save_modelfile(MobileNet, logdir)
        elif netarchitecture == "alexnet":
            net = AlexNet(kernel_size=kernelsize, dropout_prob=droprate, output_size=4)
            save_modelfile(AlexNet, logdir)
        elif netarchitecture == "mlp":
            net = MLP(input_size=19, layer_dim=128, dropout_prob=droprate, output_size=4)
            save_modelfile(MLP, logdir)
        elif netarchitecture == "mlp_tanh":
            net = MLP(
                input_size=19, layer_dim=128, dropout_prob=droprate, output_size=4, act="tanh"
            )
            save_modelfile(MLP, logdir)
        elif netarchitecture == "mlp2":
            net = MLP2(
                input_size=19, layer1_dim=128, layer2_dim=256, dropout_prob=droprate, output_size=4
            )
            save_modelfile(MLP2, logdir)
        elif netarchitecture == "mlp2_tanh":
            net = MLP2(
                input_size=19,
                layer1_dim=128,
                layer2_dim=256,
                dropout_prob=droprate,
                output_size=4,
                act="tanh",
            )
            save_modelfile(MLP2, logdir)

        print(net)
        if cuda_flag:
            net.cuda()

        net.print_net_parameters_num()

        ### Load data
        # root_dir = os.path.join("/gpfs/home/pivaturi/data/", datadir)
        trans = (
            transforms.Compose([ToTensor(0)])
            if en_manualfeatures
            else transforms.Compose([Crop(args.inputsize), ToTensor(1)])
        )
        train_dataset = Physionet2017Dataset(
            datadir,
            transform=trans,
            kfolds=kfolds,
            split_num=splitnum,
            val_size=args.val,
            split_type="train",
            random_seed=randomseed,
            en_cache=True,
            manual_features=en_manualfeatures,
            saveidx_dir=os.path.join('./../', 'model')
        )

        test_dataset = Physionet2017Dataset(
            datadir,
            transform=trans,
            kfolds=kfolds,
            split_num=splitnum,
            val_size=args.val,
            split_type="test",
            random_seed=randomseed,
            en_cache=True,
            manual_features=en_manualfeatures,
            saveidx_dir=os.path.join('./../', 'model')
        )

        if args.val != 0:
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

        labels_distribution = train_dataset.labels_distribution()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=cuda_flag,
        )
        if args.val != 0:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=cuda_flag,
            )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=cuda_flag,
        )

        ### Loss function
        if args.noiselossscaling == -1:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_weights = torch.Tensor(1 / labels_distribution)
            if args.noiselossscaling != 1:
                loss_weights[3] *= args.noiselossscaling
                print("Noise loss scaled by %f" % args.noiselossscaling)
            if cuda_flag:
                loss_weights = loss_weights.cuda()
            loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

        ### Optimizer
        if args.optimizer == "sgd":
            optimizer = optim.SGD(
                net.parameters(),
                lr=lr,
                momentum=momentum,
                dampening=0,
                weight_decay=weight_decay,
                nesterov=True,
            )
        elif args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(
                net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif args.optimizer == "adam":
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        if args.sched == "plat":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                verbose=True,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, lrdecaystep, gamma=lrgamma)

        # Tensorboard log
        writer = SummaryWriter(log_dir=logdir)
        #    logdir = writer.file_writer.get_logdir()
        CHECK_EVERY = 20

        for n_epoch in range(num_epochs):

            ### Test on train set
            if (n_epoch + 1) % CHECK_EVERY == 0:
                train_loss, train_accuracy, train_precision, train_recall, train_fscore, train_support, train_testtime = test(
                    net, train_dataloader, loss_fn=loss_fn
                )

                writer.add_scalar("train/loss", train_loss, n_epoch)
                writer.add_scalar("train/accuracy", train_accuracy, n_epoch)
                writer.add_scalars(
                    "train/precision",
                    {
                        "N": train_precision[0],
                        "A": train_precision[1],
                        "O": train_precision[2],
                        "~": train_precision[3],
                    },
                    n_epoch,
                )
                writer.add_scalars(
                    "train/recall",
                    {
                        "N": train_recall[0],
                        "A": train_recall[1],
                        "O": train_recall[2],
                        "~": train_recall[3],
                    },
                    n_epoch,
                )
                writer.add_scalars(
                    "train/fscore",
                    {
                        "N": train_fscore[0],
                        "A": train_fscore[1],
                        "O": train_fscore[2],
                        "~": train_fscore[3],
                    },
                    n_epoch,
                )
                writer.add_scalars(
                    "train/support",
                    {
                        "N": train_support[0],
                        "A": train_support[1],
                        "O": train_support[2],
                        "~": train_support[3],
                    },
                    n_epoch,
                )


                ### Test on test set
                test_loss, test_accuracy, test_precision, test_recall, test_fscore, test_support, test_testtime = test(
                    net, test_dataloader, loss_fn=loss_fn
                )

                writer.add_scalar("test/loss", test_loss, n_epoch)
                writer.add_scalar("test/accuracy", test_accuracy, n_epoch)
                writer.add_scalars(
                    "test/precision",
                    {
                        "N": test_precision[0],
                        "A": test_precision[1],
                        "O": test_precision[2],
                        "~": test_precision[3],
                    },
                    n_epoch,
                )
                writer.add_scalars(
                    "test/recall",
                    {
                        "N": test_recall[0],
                        "A": test_recall[1],
                        "O": test_recall[2],
                        "~": test_recall[3],
                    },
                    n_epoch,
                )
                writer.add_scalars(
                    "test/fscore",
                    {
                        "N": test_fscore[0],
                        "A": test_fscore[1],
                        "O": test_fscore[2],
                        "~": test_fscore[3],
                    },
                    n_epoch,
                )
                writer.add_scalars(
                    "test/support",
                    {
                        "N": test_support[0],
                        "A": test_support[1],
                        "O": test_support[2],
                        "~": test_support[3],
                    },
                    n_epoch,
                )

            ### Train
            train_start_time = time.time()
            train_single_epoch(
                net, train_dataloader, n_epoch=n_epoch, loss_fn=loss_fn, optimizer=optimizer
            )
            train_time = time.time() - train_start_time

            ### Write log

            if (n_epoch + 1) % CHECK_EVERY == 0:

                log_dict = OrderedDict(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "train_testtime": train_testtime,
                        "train_precision_N": train_precision[0],
                        "train_precision_A": train_precision[1],
                        "train_precision_O": train_precision[2],
                        "train_precision_~": train_precision[3],
                        "train_recall_N": train_recall[0],
                        "train_recall_A": train_recall[1],
                        "train_recall_O": train_recall[2],
                        "train_recall_~": train_recall[3],
                        "train_fscore_N": train_fscore[0],
                        "train_fscore_A": train_fscore[1],
                        "train_fscore_O": train_fscore[2],
                        "train_fscore_~": train_fscore[3],
                        "train_support_N": train_support[0],
                        "train_support_A": train_support[1],
                        "train_support_O": train_support[2],
                        "train_support_~": train_support[3],
                        "train_time": train_time,
                        
                        "epoch": n_epoch,
                        "test_loss": test_loss,
                        "test_accuracy": test_accuracy,
                        "test_testtime": test_testtime,
                        "test_precision_N": test_precision[0],
                        "test_precision_A": test_precision[1],
                        "test_precision_O": test_precision[2],
                        "test_precision_~": test_precision[3],
                        "test_recall_N": test_recall[0],
                        "test_recall_A": test_recall[1],
                        "test_recall_O": test_recall[2],
                        "test_recall_~": test_recall[3],
                        "test_fscore_N": test_fscore[0],
                        "test_fscore_A": test_fscore[1],
                        "test_fscore_O": test_fscore[2],
                        "test_fscore_~": test_fscore[3],
                        "test_support_N": test_support[0],
                        "test_support_A": test_support[1],
                        "test_support_O": test_support[2],
                        "test_support_~": test_support[3],
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

                with open(os.path.join(logdir, "log.csv"), "a") as csvfile:
                    fieldnames = sorted(list(log_dict.keys()))
                    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if (n_epoch + 1) == CHECK_EVERY:
                        csv_writer.writeheader()
                    csv_writer.writerow(log_dict)
                model_logdir = os.path.join(logdir, "model")
                if not os.path.exists(model_logdir):
                    os.makedirs(model_logdir)
                #torch.save(net.state_dict(), os.path.join(model_logdir, "model_params_" + str(n_epoch) + ".torch"))

            ### Optimizer params
            writer.add_scalar("params/lr", optimizer.param_groups[0]["lr"], n_epoch)

            ### LR scheduler update
            if args.sched == "plat":
                scheduler.step(val_loss)
                if optimizer.param_groups[0]["lr"] < 1e-5:
                    break
            else:
                scheduler.step(n_epoch)

        ### Save model
        model_logdir = os.path.join(logdir, "model")
        if not os.path.exists(model_logdir):
            os.makedirs(model_logdir)
            
        torch.save(net.state_dict(), os.path.join(model_logdir, "model_params.torch"))
        torch.save(net, os.path.join(model_logdir, "model_entire.torch"))
        del net
        torch.cuda.empty_cache()

        print("Finished! %s" % logdir_comment, logdir)
        #_msg.send("Finished! %s" % logdir_comment, logdir)

    except Exception as e:
        #_msg.send("Exception for %s" % logdir_comment, str(e))
        shutil.rmtree(logdir)
        if ("out of memory" in str(e)) or ("No GPU available for" in str(e)):
        	print("Resending jobs from %s" % logdir_comment, logdir)
            # Resend jobs after 30 mins!
            #time.sleep(60 * 30)
            #_msg.send("Resending jobs from %s" % logdir_comment, logdir)
            #os.system("bash send_all_jobs.sh")
        raise e


