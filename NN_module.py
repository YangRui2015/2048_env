# -*- coding: utf-8 -*-
# NN.py
# author: yangrui

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

# CNN网络
class CNN_Net(nn.Module):
    def __init__(self, input_len, output_num, conv_size=(32, 64), fc_size=(1024, 128), out_softmax=False):
        super(CNN_Net, self).__init__()
        self.input_len = input_len
        self.output_num = output_num
        self.out_softmax = out_softmax 

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_size[0], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_size[0], conv_size[1], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Linear(conv_size[1] * self.input_len * self.input_len, fc_size[0])
        self.fc2 = nn.Linear(fc_size[0], fc_size[1])
        self.head = nn.Linear(fc_size[1], self.output_num)

    def forward(self, x):
        x = x.reshape(-1,1,self.input_len, self.input_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        output = self.head(x)
        if self.out_softmax:
            output = F.softmax(output, dim=1)   #值函数估计不应该有softmax
        return output


# 全连接网络
class FC_Net(nn.Module):
    def __init__(self, input_num, output_num, fc_size=(1024, 128), out_softmax=False):
        super(FC_Net, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        self.out_softmax = out_softmax 

        self.fc1 = nn.Linear(self.input_num, fc_size[0])
        self.fc2 = nn.Linear(fc_size[0], fc_size[1])
        self.head = nn.Linear(fc_size[1], self.output_num)

    def forward(self, x):
        x = x.reshape(-1, self.input_num)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        output = self.head(x)
        if self.out_softmax:
            output = F.softmax(output, dim=1)   #值函数估计不应该有softmax
        return output
