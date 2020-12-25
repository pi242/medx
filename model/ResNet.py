#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:06:12 2017

@author: Matteo Gadaleta
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, kernel_size=16, dropout_prob=0, output_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        
        ### Input block definitions
        self.conv_in = BasicConv(1, 64, kernel_size=kernel_size, stride = 2)
        
        ### RESIDUAL BLOCKS definitions
        res_layers = [
                    ResidualBlock(64, 64, kernel_size, dropout_prob=0, stride=1),
                    ResidualBlock(64, 64, kernel_size, dropout_prob=0, stride=2),
                    ResidualBlock(64, 64, kernel_size, dropout_prob=0, stride=1),
                    ResidualBlock(64, 128, kernel_size, dropout_prob=0, stride=2),
                    ResidualBlock(128, 128, kernel_size, dropout_prob=0, stride=1),
                    ResidualBlock(128, 128, kernel_size, dropout_prob=0, stride=2),
                    ResidualBlock(128, 128, kernel_size, dropout_prob=0, stride=1),
                    ResidualBlock(128, 256, kernel_size, dropout_prob=0, stride=2),
                    ResidualBlock(256, 256, kernel_size, dropout_prob=0, stride=1),
                    ResidualBlock(256, 256, kernel_size, dropout_prob=0, stride=2),
                    ResidualBlock(256, 256, kernel_size, dropout_prob=0, stride=1),
                    ResidualBlock(256, 512, kernel_size, dropout_prob=0, stride=2),
                    ResidualBlock(512, 512, kernel_size, dropout_prob=0, stride=1),
                    ResidualBlock(512, 512, kernel_size, dropout_prob=0, stride=2),
                    ResidualBlock(512, 512, kernel_size, dropout_prob=0, stride=1),
                    ]
                
        self.res_blocks = nn.Sequential(*res_layers)
        
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
        x = self.conv_in(x)
        ### RESIDUAL BLOCKS
        x = self.res_blocks(x)
        
        ### Output block
        x = F.avg_pool1d(x, x.shape[2])
        
        x = x.view(x.size(0), -1) # Flatten
        
        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob
        
        self.conv1 = BasicConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = BasicConv(out_channels, out_channels, kernel_size=kernel_size, en_act=False)
                
        # Residual projection
        if (self.in_channels != self.out_channels) or (self.stride > 1):
            self.convproj = BasicConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, en_act=False)
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.dropout_prob > 0:
            out = F.dropout(out, p=self.dropout_prob, training=self.training)
            
        if (self.in_channels != self.out_channels) or (self.stride > 1):
            x = self.convproj(x)
            
        out += x
        out = F.relu(out, inplace=True)
        
        return out
        


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding='same', en_bn=True, en_act=True, **kwargs):
        super().__init__()
        
        self.padding = padding
        self.en_bn = en_bn
        self.en_act = en_act
        
        ### Padding (same)
        if kernel_size == 1:
            lpad, rpad = (0,0)
        else:
            lpad  = int(kernel_size/2)
            rpad  = int(np.ceil(kernel_size/2))-1
#        self.pad = nn.ReplicationPad1d((lpad, rpad))
        self.pad = nn.ConstantPad1d((lpad, rpad), 0)
        
        bias = False if en_bn else True
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, **kwargs)
        
        if self.en_bn:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.padding == 'same':
            x = self.pad(x)
        x = self.conv(x)
        
        if self.en_bn:
            x = self.bn(x)
            
        if self.en_act:
            return F.relu(x, inplace=True)
        else:
            return x



if __name__=='__main__':
  
    net = ResNet(kernel_size=16, dropout_prob=0.3, output_size=4)
    net.print_net_parameters_num()
    
    from torch.autograd import Variable
    x = Variable(torch.rand(50,1,512))
    out = net(x)
    
    print(x.shape)
    print(out.shape)
