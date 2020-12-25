#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:13:27 2018

@author: Matteo Gadaleta
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetv2(nn.Module):
    def __init__(self, kernel_size=16, dropout_prob=0, output_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        
        ### Input block definitions
        # Padding (same)
        lpad  = int(kernel_size/2)
        rpad  = int(np.ceil(kernel_size/2))-1
#        self.pad = nn.ReplicationPad1d((lpad, rpad))
        self.pad = nn.ConstantPad1d((lpad, rpad), 0)
        self.conv_in = nn.Conv1d(1, 64, kernel_size=kernel_size, stride=2)
        
        ### RESIDUAL BLOCKS definitions
        res_layers = [
                    ResidualBlock(64, 64, kernel_size, dropout_prob=dropout_prob, stride=1),
                    ResidualBlock(64, 64, kernel_size, dropout_prob=dropout_prob, stride=2),
                    ResidualBlock(64, 64, kernel_size, dropout_prob=dropout_prob, stride=1),
                    ResidualBlock(64, 128, kernel_size, dropout_prob=dropout_prob, stride=2),
                    ResidualBlock(128, 128, kernel_size, dropout_prob=dropout_prob, stride=1),
                    ResidualBlock(128, 128, kernel_size, dropout_prob=dropout_prob, stride=2),
                    ResidualBlock(128, 128, kernel_size, dropout_prob=dropout_prob, stride=1),
                    ResidualBlock(128, 256, kernel_size, dropout_prob=dropout_prob, stride=2),
                    ResidualBlock(256, 256, kernel_size, dropout_prob=dropout_prob, stride=1),
                    ResidualBlock(256, 256, kernel_size, dropout_prob=dropout_prob, stride=2),
                    ResidualBlock(256, 256, kernel_size, dropout_prob=dropout_prob, stride=1),
                    ResidualBlock(256, 512, kernel_size, dropout_prob=dropout_prob, stride=2),
                    ResidualBlock(512, 512, kernel_size, dropout_prob=dropout_prob, stride=1),
                    ResidualBlock(512, 512, kernel_size, dropout_prob=dropout_prob, stride=2),
                    ResidualBlock(512, 512, kernel_size, dropout_prob=dropout_prob, stride=1),
                    ]
                
        self.res_blocks = nn.Sequential(*res_layers)
        
        # OUT
        self.bn_out = nn.BatchNorm1d(512)
        self.act_out = nn.ReLU(inplace=True)
        self.denselayer = nn.Linear(512, output_size)
        
        ### Weigths initialization
        self.apply(self.init_weights)
        
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal(m.weight.data)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
        
        
        
    def forward(self, x):
        
        ### Input block
        x = self.conv_in(self.pad(x))
        ### RESIDUAL BLOCKS
        x = self.res_blocks(x)
        
        ### Output block
        x = self.act_out(self.bn_out(x))
        x = F.avg_pool1d(x, x.shape[2])
        x = x.view(x.size(0), -1) # Flatten
        
        x = self.denselayer(x)
        
        return x

    def print_net_parameters_num(self):
        
        tot_params = 0
        for name, p in self.named_parameters():
            print('%s \t Num params: %d \t Shape: %s' % (name, np.prod(p.data.shape), str(p.data.shape)))
            tot_params += np.prod(p.data.shape)
        print('TOTAL NUMBER OF PARAMETERS: ', tot_params)



# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout_prob=0):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.dropout_prob = dropout_prob
        
        ### Padding (same)
        if kernel_size == 1:
            lpad, rpad = (0,0)
        else:
            lpad  = int(kernel_size/2)
            rpad  = int(np.ceil(kernel_size/2))-1
#        self.pad = nn.ReplicationPad1d((lpad, rpad))
        self.pad = nn.ConstantPad1d((lpad, rpad), 0)
        
        # BN 1
        self.bn_res1 = nn.BatchNorm1d(in_channels)
        # Act 1
        self.act_res1 = nn.ReLU(inplace=True)
        # Conv 1
        self.conv_res1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        # BN 2
        self.bn_res2 = nn.BatchNorm1d(out_channels)
        # Act 2
        self.act_res2 = nn.ReLU(inplace=True)
        # Conv 2
        self.conv_res2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, bias=False)
        
        # Residual projection
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn_1x1 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        if self.in_channel != self.out_channel:
            x = self.act_res1(self.bn_res1(x))
            out = self.act_res2(self.bn_res2(self.conv_res1(self.pad(x))))
        else:
            out = self.act_res2(self.bn_res2(self.conv_res1(self.pad(self.act_res1(self.bn_res1(x))))))
        
        if self.dropout_prob > 0:
            out = F.dropout(out, p=self.dropout_prob, training=self.training)
        
        out = self.conv_res2(self.pad(out))
        
        if (self.in_channel != self.out_channel) or self.stride > 1:
            return torch.add(self.conv_1x1(x), out)
        else:
            return torch.add(x, out)


if __name__=='__main__':
  
    net = ResNetv2(kernel_size=16, dropout_prob=0.3, output_size=4)
    net.print_net_parameters_num()
    
    from torch.autograd import Variable
    x = Variable(torch.rand(2,1,9000))
    out = net(x)
    
    print(x.shape)
    print(out.shape)
