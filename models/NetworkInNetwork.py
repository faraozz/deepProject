#
# Network in network implementation
# original paper: https://arxiv.org/pdf/1312.4400.pdf%20http://arxiv.org/abs/1312.4400.pdf
# helpful description: https://d2l.ai/chapter_convolutional-modern/nin.html
#

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NiN_Block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel=3, stride=2, pad=0):
        super(NiN_Block, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 1)
        self.conv3 = nn.Conv2d(output_channels, output_channels, 1)

    def forward(self, input):
        x = F.relu(self.conv1(input), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        return x

class NiN(nn.Module):
    def __init__(self, input_channels=1, output_categories=10):
        super(NiN, self).__init__()

        self.nin1 = NiN_Block(input_channels, 96, kernel=5, stride=1, pad=2)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.nin2 = NiN_Block(96, 256, kernel=5, pad=2)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.nin3 = NiN_Block(256, 384, kernel=3, pad=1)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.nin4 = NiN_Block(384, output_categories, kernel=3, pad=1)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # this computes the average for each filter layer - and since
        # there are output_categories filter layers, this results in one response per target category
        self.flatten = nn.Flatten()

    def forward(self, input):
        x = self.nin1(input)
        x = self.maxpool1(x)
        x = self.nin2(x)
        x = self.maxpool2(x)
        x = self.nin3(x)
        x = self.maxpool3(x)
        x = self.nin4(x)
        x = self.avgpool1(x)
        x = self.flatten(x)
        x = nn.functional.softmax(x)
        return x

