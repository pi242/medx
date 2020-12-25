#!/bin/sh
cd lib
for i in 0 1 2 3 4
# for i in 0
do
python3 -u train-iterative.py --netarchitecture mobilenet --epochs 200 --logdir log-$i --batchsize 50 --optimizer sgd --lr 0.1 --lrgamma 0.33 --lrdecaystep 25 --droprate 0.3 --weightdecay 0.0005 --momentum 0.9 --noiselossscaling 0.1 --clipgrad 2.0 --datadir ./../data/training2017_raligned --kernelsize 16 --kfolds 5 --randomseed 123 --val 0.0 --splitnum $i
done

