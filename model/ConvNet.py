#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:53:14 2018

@author: Matteo Gadaleta
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, kernel_size=16, dropout_prob=0, output_size=2, en_conv1x1=False):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        
        ### Layers definitions
        # Conv layers
        self.conv1_1 = BasicConv(1, 64, kernel_size)
        self.conv1_2 = BasicConv(64, 64, kernel_size)
        
        self.conv2_1 = BasicConv(64, 128, kernel_size)
        self.conv2_2 = BasicConv(128, 128, kernel_size)
        
        self.conv3_1 = BasicConv(128, 256, kernel_size)
        self.conv3_2 = BasicConv(256, 256, kernel_size)
        self.conv3_3 = BasicConv(256, 256, 1 if en_conv1x1 else kernel_size)
        
        self.conv4_1 = BasicConv(256, 512, kernel_size)
        self.conv4_2 = BasicConv(512, 512, kernel_size)
        self.conv4_3 = BasicConv(512, 512, 1 if en_conv1x1 else kernel_size)
        
        self.conv5_1 = BasicConv(512, 512, kernel_size)
        self.conv5_2 = BasicConv(512, 512, kernel_size)
        self.conv5_3 = BasicConv(512, 512, 1 if en_conv1x1 else kernel_size)
        
        # Dense layers
        self.dense_1 = nn.Linear(512*35, 4096)
        self.dense_2 = nn.Linear(4096, 4096)
        
        # Output layer
        self.output_layer = nn.Linear(4096, output_size)
        
        ### Weigths initialization
        self.apply(self.init_weights)
        
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal(m.weight.data)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.bias.data.zero_()
        
        
        
    def forward(self, x):
        
        ### Convolutional blocks
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = F.max_pool1d(x, kernel_size=4, stride=4)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = F.max_pool1d(x, kernel_size=4, stride=4)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = F.max_pool1d(x, kernel_size=4, stride=4)
        
        ### Flatten
        x = x.view(x.size(0), -1) # Flatten
        
        ### FC blocks
        x = self.dense_1(x)
        x = F.relu(x, inplace=True)
        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            
        x = self.dense_2(x)
        x = F.relu(x, inplace=True)
        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        ### Output block
        x = self.output_layer(x)
        
        return x

    def print_net_parameters_num(self):
        
        all_params = 0
        for name, p in self.named_parameters():
            print('%s \t Num params: %d \t Shape: %s' % (name, np.prod(p.data.shape), str(p.data.shape)))
            all_params += np.prod(p.data.shape)
        print('Total parameters: ', all_params)

      

class AlexNet(nn.Module):
    def __init__(self, kernel_size=16, dropout_prob=0, output_size=2):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        
        # Conv layers
        self.conv1 = BasicConv(1, 96, kernel_size*5, stride=4, en_bn=False)
        self.conv2 = BasicConv(96, 256, kernel_size*3, en_bn=False)
        self.conv3 = BasicConv(256, 384, kernel_size, en_bn=False)
        self.conv4 = BasicConv(384, 384, kernel_size, en_bn=False)
        self.conv5 = BasicConv(384, 256, kernel_size, en_bn=False)
        
        # Dense layers
        self.dense_1 = nn.Linear(256*35, 4096)
        self.dense_2 = nn.Linear(4096, 4096)
        
        # Output layer
        self.output_layer = nn.Linear(4096, output_size)
        
        ### Weigths initialization
        self.apply(self.init_weights)
        
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal(m.weight.data)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.bias.data.zero_()
        
        
        
    def forward(self, x):
        
        ### Convolutional blocks
        x = self.conv1(x)
        x = F.max_pool1d(x, 4, stride=4)
        
        x = self.conv2(x)
        x = F.max_pool1d(x, 4, stride=4)
        
        x = self.conv3(x)
        
        x = self.conv4(x)
        
        x = self.conv5(x)
        x = F.max_pool1d(x, 4, stride=4)
        
        ### Flatten
        x = x.view(x.size(0), -1) # Flatten
        
        ### FC blocks
        x = self.dense_1(x)
        x = F.relu(x, inplace=True)
        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            
        x = self.dense_2(x)
        x = F.relu(x, inplace=True)
        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        ### Output block
        x = self.output_layer(x)
        
        return x

    def print_net_parameters_num(self):
        
        all_params = 0
        for name, p in self.named_parameters():
            print('%s \t Num params: %d \t Shape: %s' % (name, np.prod(p.data.shape), str(p.data.shape)))
            all_params += np.prod(p.data.shape)
        print('Total parameters: ', all_params)


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

#    net = VGG(kernel_size=16, dropout_prob=0.3, output_size=4, en_conv1x1=True)
    net = VGG(kernel_size=16, dropout_prob=0.3, output_size=4)
    net.print_net_parameters_num()
    
    from torch.autograd import Variable
    x = Variable(torch.rand(1,1,9000))
    out = net(x)
    
    print(x.shape)
    print(out.shape)




