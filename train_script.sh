#!/bin/sh
# for i in 0 1 2 3 4
for i in 0
do
python train.py --netarchitecture mobilenet --epochs 10 --logdir log --batchsize 50 --optimizer sgd --lr 0.1 --lrgamma 0.5 --lrdecaystep 150 --droprate 0.3 --weightdecay 0.0005 --momentum 0.9 --noiselossscaling 0.1 --clipgrad 2.0 --datadir training2017_rwaves --kernelsize 16 --kfolds 5 --randomseed 123 --val 0.0 --splitnum $i
done

