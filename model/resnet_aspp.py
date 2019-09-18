import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.SPP import ASPP_simple, ASPP
from model.ResNet import ResNet101, ResNet18, ResNet34, ResNet50

import time
INPUT_SIZE = 512

class ResNet_ASPP(nn.Module):
    def __init__(self, nInputChannels, n_classes, os, backbone_type):
        super(ResNet_ASPP, self).__init__()

        self.os = os
        self.backbone_type = backbone_type
        
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if backbone_type == 'resnet18':
            self.backbone_features = ResNet18(nInputChannels, os, pretrained=False)
        elif backbone_type == 'resnet34':
            self.backbone_features = ResNet34(nInputChannels, os, pretrained=False)
        elif backbone_type == 'resnet50':
            self.backbone_features = ResNet50(nInputChannels, os, pretrained=True)
        else:
            raise NotImplementedError

        asppInputChannels = 512
        asppOutputChannels = 256
        if backbone_type == 'resnet50': asppInputChannels = 2048
        
        self.aspp = ASPP(asppInputChannels, asppOutputChannels, rates)
        self.last_conv = nn.Sequential(
                nn.Conv2d(asppOutputChannels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
            ) 

    def forward(self, input):
        x, low_level_features, conv1_feat, layer2_feat, layer3_feat = self.backbone_features(input)
        layer4_feat = x
        if self.os == 32:
            x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.aspp(x)
        aspp_x = x
        x = self.last_conv(x)
        x = F.upsample(x, input.size()[2:], mode='bilinear', align_corners=True)

        return x, conv1_feat, low_level_features, layer2_feat, layer3_feat, layer4_feat, aspp_x
