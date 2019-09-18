import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.SPP import ASPP_simple, ASPP
from model.ResNet import ResNet101, ResNet18, ResNet34, ResNet50
from model.resnet_aspp import ResNet_ASPP 

import time
INPUT_SIZE = 512

def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)

class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MGA_Network(nn.Module):
    def __init__(self, nInputChannels, n_classes, os, img_backbone_type, flow_backbone_type):
        super(MGA_Network, self).__init__()

        self.inplanes = 64
        self.os = os
        
        if os == 16:
            aspp_rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            aspp_rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        elif os == 32:
            strides = [1, 2, 2, 2]
            rates = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        assert img_backbone_type == 'resnet101'

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = [3, 4, 23, 3]

        self.layer1 = self._make_layer( 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer( 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer( 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_layer( 512, layers[3], stride=strides[3], rate=rates[3])
        
        asppInputChannels = 2048
        asppOutputChannels = 256
        lowInputChannels =  256
        lowOutputChannels = 48

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, aspp_rates)

        self.last_conv = nn.Sequential(
                nn.Conv2d(asppOutputChannels+lowOutputChannels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
            )

        # low_level_features to 48 channels
        self.conv2 = nn.Conv2d(lowInputChannels, lowOutputChannels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(lowOutputChannels)

        self.resnet_aspp = ResNet_ASPP(nInputChannels, n_classes, os, flow_backbone_type)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # # for MGA_tmc
        if flow_backbone_type == 'resnet34':
            self.conv1x1_conv1_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
            self.conv1x1_conv1_spatial = nn.Conv2d(64, 1, 1, bias=True)

            self.conv1x1_layer1_channel_wise = nn.Conv2d(64*4, 64*4, 1, bias=True)
            self.conv1x1_layer1_spatial = nn.Conv2d(64, 1, 1, bias=True)

            self.conv1x1_layer2_channel_wise  = nn.Conv2d(128*4, 128*4, 1, bias=True)
            self.conv1x1_layer2_spatial = nn.Conv2d(128, 1, 1, bias=True)

            self.conv1x1_layer3_channel_wise = nn.Conv2d(256*4, 256*4, 1, bias=True)
            self.conv1x1_layer3_spatial = nn.Conv2d(256, 1, 1, bias=True)

            self.conv1x1_layer4_channel_wise = nn.Conv2d(512*4, 512*4, 1, bias=True)
            self.conv1x1_layer4_spatial = nn.Conv2d(512, 1, 1, bias=True)
        else:
            raise NotImplementedError

    def _make_layer(self, planes, blocks, stride=1, rate=1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encoder_attention_module_MGA_tmc(self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial):
        # spatial attention
        flow_feat_map = conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)

        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat

    def decoder_attention_module_MGA_t(self, img_feat, flow_map):
        final_feat = img_feat * flow_map + img_feat
        return final_feat, flow_map

    def forward(self, img, flow): 
        flow_map, flow_conv1_feat, flow_layer1_feat, flow_layer2_feat, flow_layer3_feat, flow_layer4_feat, flow_aspp_feat = self.resnet_aspp(flow)
        flow_feat_lst = [flow_conv1_feat, flow_layer1_feat, flow_layer2_feat, flow_layer3_feat, flow_layer4_feat, flow_aspp_feat]

        flow_map = torch.nn.Sigmoid()(flow_map) 

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat = x 
        x = self.encoder_attention_module_MGA_tmc(x, flow_conv1_feat, self.conv1x1_conv1_channel_wise, self.conv1x1_conv1_spatial)
        after_conv1_feat = x

        x = self.maxpool(x)
        x = self.layer1(x)
        layer1_feat = x
        x = self.encoder_attention_module_MGA_tmc(x, flow_layer1_feat, self.conv1x1_layer1_channel_wise, self.conv1x1_layer1_spatial)
        after_layer1_feat = x
        low_level_features = x

        x = self.layer2(x)
        layer2_feat = x 
        x = self.encoder_attention_module_MGA_tmc(x, flow_layer2_feat, self.conv1x1_layer2_channel_wise, self.conv1x1_layer2_spatial)
        after_layer2_feat = x 

        x = self.layer3(x)
        layer3_feat = x 
        x = self.encoder_attention_module_MGA_tmc(x, flow_layer3_feat, self.conv1x1_layer3_channel_wise, self.conv1x1_layer3_spatial)
        after_layer3_feat = x

        x = self.layer4(x)
        layer4_feat = x 
        x = self.encoder_attention_module_MGA_tmc(x, flow_layer4_feat, self.conv1x1_layer4_channel_wise, self.conv1x1_layer4_spatial)
        after_layer4_feat = x

        img_feat_lst = [conv1_feat, layer1_feat, layer2_feat, layer3_feat, layer4_feat]
        img_feat_attentioned_lst = [after_conv1_feat, after_layer1_feat, after_layer2_feat, after_layer3_feat, after_layer4_feat]

        if self.os == 32:
            x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        x = self.aspp(x)
        img_feat_lst.append(x)
        
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        
        x = F.upsample(x, low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        img_feat_lst.append(x)

        before_attention_x = x

        flow_map = F.upsample(flow_map, x.size()[2:], mode='bilinear', align_corners=True)
        x, flow_map = self.decoder_attention_module_MGA_t(x, flow_map)
        after_attention_x = x

        x = self.last_conv(x)
        x = F.upsample(x, img.size()[2:], mode='bilinear', align_corners=True)

        return x, flow_map, flow_feat_lst, img_feat_lst, img_feat_attentioned_lst

def init_conv1x1(net):
    for k, v in net.state_dict().items():
        if 'conv1x1' in k:
            if 'weight' in k:
                nn.init.kaiming_normal_(v)
            elif 'bias' in k:
                nn.init.constant_(v, 0)
    return net

def get_params(model, lr):
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv1x1' in key:
            params += [{'params':[value], 'lr':lr*10}]
        else:
            params += [{'params':[value], 'lr':lr}]
    return params