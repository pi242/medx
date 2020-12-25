#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:17:34 2018

@author: Matteo Gadaleta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Inception(nn.Module):
    
    def __init__(self, kernel_size, output_size, dropout_prob=0):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        
        self.stem = Stem(kernel_size)

        inceptionA_layers = [InceptionModule(in_channels=384, kernel_size=kernel_size, pool_kernels=96, onexone_kernels=96, single_kernels=[64, 96], double_kernels=[64, 96, 96]) for i in range(3)]
        self.inceptionA = nn.Sequential(*inceptionA_layers)
        self.reductionA = ReductionModule(in_channels=384, kernel_size=kernel_size, single_kernels=384, double_kernels=[192, 224, 256])

        inceptionB_layers = [InceptionModule(in_channels=1024, kernel_size=kernel_size, pool_kernels=128, onexone_kernels=384, single_kernels=[192, 256], double_kernels=[192, 224, 256]) for i in range(5)]
        self.inceptionB = nn.Sequential(*inceptionB_layers)
        self.reductionB = ReductionModule(in_channels=1024, kernel_size=kernel_size, single_kernels=192, double_kernels=[256, 320, 320])

        inceptionC_layers = [InceptionModule(in_channels=1536, kernel_size=kernel_size, pool_kernels=256, onexone_kernels=256, single_kernels=[384, 512], double_kernels=[384, 512, 512]) for i in range(2)]
        self.inceptionC = nn.Sequential(*inceptionC_layers)
        
        self.denselayer = nn.Linear(1536, output_size)
        
        ### Weigths initialization
        self.apply(self.init_weights)

    def forward(self, x):
        
        x = self.stem(x)
        
        x = self.inceptionA(x)
        x = self.reductionA(x)
        
        x = self.inceptionB(x)
        x = self.reductionB(x)
        
        x = self.inceptionC(x)
        
        x = F.avg_pool1d(x, x.shape[2])
        
        x = x.view(x.size(0), -1) # Flatten
        
        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            
        x = self.denselayer(x)
            
        return x
        
    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal(m.weight.data)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            
    def print_net_parameters_num(self):
        
        tot_params = 0
        for name, p in self.named_parameters():
            print('%s \t Num params: %d \t Shape: %s' % (name, np.prod(p.data.shape), str(p.data.shape)))
            tot_params += np.prod(p.data.shape)
        print('TOTAL NUMBER OF PARAMETERS: ', tot_params)

class Stem(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

        ### Padding (same)
        lpad  = int(kernel_size/2)
        rpad  = int(np.ceil(kernel_size/2))-1
#        self.pad = nn.ReplicationPad1d((lpad, rpad))
        self.pad = nn.ConstantPad1d((lpad, rpad), 0)

        self.conv1_1 = BasicConv(1, 32, kernel_size=kernel_size, padding='valid', stride=2)
        self.conv1_2 = BasicConv(32, 32, kernel_size=kernel_size, padding='valid')
        self.conv1_3 = BasicConv(32, 64, kernel_size=kernel_size)
        self.conv1_4r = BasicConv(64, 96, kernel_size=kernel_size, padding='valid', stride=2)

        self.conv2_1l = BasicConv(160, 64, kernel_size=1)
        self.conv2_2l = BasicConv(64, 96, kernel_size=kernel_size, padding='valid')
        
        self.conv2_1r = BasicConv(160, 64, kernel_size=1)
        self.conv2_2r = BasicConv(64, 64, kernel_size=kernel_size)
        self.conv2_3r = BasicConv(64, 96, kernel_size=kernel_size, padding='valid')

        self.conv3_1l = BasicConv(192, 192, kernel_size=kernel_size, padding='valid', stride=2)
        
    def forward(self, x):
        
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        
        x1l = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=2)
        x1r = self.conv1_4r(x)
        
        x = torch.cat([x1l, x1r], 1)
        
        x2l = self.conv2_1l(x)
        x2l = self.conv2_2l(x2l)
        
        x2r = self.conv2_1r(x)
        x2r = self.conv2_2r(x2r)
        x2r = self.conv2_3r(x2r)
        
        x = torch.cat([x2l, x2r], 1)
        
        x3l = self.conv3_1l(x)
        
        x3r = F.max_pool1d(x, kernel_size=self.kernel_size, stride=2)
        
        x = torch.cat([x3l, x3r], 1)
        
        return x

        


class InceptionModule(nn.Module):

    def __init__(self, in_channels, kernel_size, pool_kernels=96, onexone_kernels=96, single_kernels=[64, 96], double_kernels=[64, 96, 96]):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        ### Padding (same)
        lpad  = int(kernel_size/2)
        rpad  = int(np.ceil(kernel_size/2))-1
#        self.pad = nn.ReplicationPad1d((lpad, rpad))
        self.pad = nn.ConstantPad1d((lpad, rpad), 0)
        
        ### Pool Branch
        self.branch_pool = BasicConv(in_channels, pool_kernels, kernel_size=1)
        
        ### 1x1 Branch
        self.branch1x1 = BasicConv(in_channels, onexone_kernels, kernel_size=1)
        
        ### Single Branch
        self.branchsingle_1 = BasicConv(in_channels, single_kernels[0], kernel_size=1)
        self.branchsingle_2 = BasicConv(single_kernels[0], single_kernels[1], kernel_size=kernel_size)
        
        ### Double Branch
        self.branchdouble_1 = BasicConv(in_channels, double_kernels[0], kernel_size=1)
        self.branchdouble_2 = BasicConv(double_kernels[0], double_kernels[1], kernel_size=kernel_size)
        self.branchdouble_3 = BasicConv(double_kernels[1], double_kernels[2], kernel_size=kernel_size)
        

    def forward(self, x):
        
        ### Pool Branch
        branch_pool = self.pad(x)
        branch_pool = F.avg_pool1d(branch_pool, kernel_size=self.kernel_size, stride=1)
        branch_pool = self.branch_pool(branch_pool)
        
        ### 1x1 Branch
        branch1x1 = self.branch1x1(x)
        
        ### Single Branch
        branchsingle = self.branchsingle_1(x)
        branchsingle = self.branchsingle_2(branchsingle)
        
        ### Double Branch
        branchdouble = self.branchdouble_1(x)
        branchdouble = self.branchdouble_2(branchdouble)
        branchdouble = self.branchdouble_3(branchdouble)
        
        ### Concat
        outputs = [branch_pool, branch1x1, branchsingle, branchdouble]
        return torch.cat(outputs, 1)
        


class ReductionModule(nn.Module):

    def __init__(self, in_channels, kernel_size, single_kernels=96, double_kernels=[64, 96, 96]):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        ### Single Branch
        self.branchsingle_2 = BasicConv(in_channels, single_kernels, kernel_size=kernel_size, padding='valid', stride=2)
        
        ### Double Branch
        self.branchdouble_1 = BasicConv(in_channels, double_kernels[0], kernel_size=1)
        self.branchdouble_2 = BasicConv(double_kernels[0], double_kernels[1], kernel_size=kernel_size)
        self.branchdouble_3 = BasicConv(double_kernels[1], double_kernels[2], kernel_size=kernel_size, padding='valid', stride=2)
        

    def forward(self, x):
        
        ### Pool Branch
        branch_pool = F.max_pool1d(x, kernel_size=self.kernel_size, stride=2)
                
        ### Single Branch
        branchsingle = self.branchsingle_2(x)
        
        ### Double Branch
        branchdouble = self.branchdouble_1(x)
        branchdouble = self.branchdouble_2(branchdouble)
        branchdouble = self.branchdouble_3(branchdouble)
        
        ### Concat
        outputs = [branch_pool, branchsingle, branchdouble]
        return torch.cat(outputs, 1)



class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding='same', **kwargs):
        super().__init__()
        
        self.padding = padding
        
        ### Padding (same)
        if kernel_size == 1:
            lpad, rpad = (0,0)
        else:
            lpad  = int(kernel_size/2)
            rpad  = int(np.ceil(kernel_size/2))-1
#        self.pad = nn.ReplicationPad1d((lpad, rpad))
        self.pad = nn.ConstantPad1d((lpad, rpad), 0)
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.padding == 'same':
            x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__=='__main__':
    
    net = Inception(16,4)
    net.print_net_parameters_num()
    1/0
    from torch.autograd import Variable
    x = Variable(torch.rand(50,1,512))
    out = net(x)
    
    print(x.shape)
    print(out.shape)