# encoding: utf-8
import math

import torch
from torch import nn
from .AIBN import AIBNorm2d
from .TNorm import TNorm
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, with_permute_adain=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.with_permute_adain = with_permute_adain
        if self.with_permute_adain:
            self.permute_adain = PermuteAdaptiveInstanceNorm2d()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.with_permute_adain:
            out = self.permute_adain(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_permute_adain:
            out = self.permute_adain(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.with_permute_adain:
            out = self.permute_adain(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AIBNBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 adaptive_weight=None,
                 generate_weight=True,
                 init_weight=0.1,
                 with_permute_adain=False):
        super(AIBNBottleneck, self).__init__()
        if adaptive_weight is None:
            self.adaptive_weight = nn.Parameter(torch.ones(1) * init_weight)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = AIBNorm2d(planes,
                             adaptive_weight=self.adaptive_weight,
                             generate_weight=generate_weight)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = AIBNorm2d(planes,
                             adaptive_weight=self.adaptive_weight,
                             generate_weight=generate_weight)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = AIBNorm2d(planes * 4,
                             adaptive_weight=self.adaptive_weight,
                             generate_weight=generate_weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample[1].adaptive_weight = self.adaptive_weight
        self.stride = stride
        self.with_permute_adain = with_permute_adain
        if self.with_permute_adain:
            self.permute_adain = PermuteAdaptiveInstanceNorm2d()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.with_permute_adain:
            out = self.permute_adain(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_permute_adain:
            out = self.permute_adain(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.with_permute_adain:
            out = self.permute_adain(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AIBNResNet(nn.Module):
    def __init__(self,
                 last_stride=2,
                 block=AIBNBottleneck,
                 layers=[3, 4, 6, 3],
                 init_weight=0.1,
                 ):
        self.inplanes = 64
        super(AIBNResNet, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_normal(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer_normal(Bottleneck,
                                              128,
                                              layers[1],
                                              stride=2)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       adaptive_weight=None,
                                       init_weight=init_weight)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=last_stride,
                                       adaptive_weight=None,
                                       init_weight=init_weight)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    adaptive_weight=None,
                    init_weight=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                AIBNorm2d(planes * block.expansion,
                          adaptive_weight=adaptive_weight,
                          generate_weight=True,
                          init_weight=init_weight),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  adaptive_weight=adaptive_weight,
                  generate_weight=True,
                  init_weight=init_weight,
                  with_permute_adain=self.with_permute_adain))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(self.inplanes,
                          planes,
                          adaptive_weight=adaptive_weight,
                          generate_weight=True,
                          init_weight=init_weight,
                          with_permute_adain=self.with_permute_adain))
            else:
                layers.append(
                    block(self.inplanes,
                          planes,
                          adaptive_weight=adaptive_weight,
                          generate_weight=True,
                          init_weight=init_weight,
                          with_permute_adain=self.with_permute_adain))

        return nn.Sequential(*layers)

    def _make_layer_normal(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, with_permute_adain=self.with_permute_adain))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_permute_adain=self.with_permute_adain))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TNormResNet(nn.Module):
    def __init__(self,
                 domain_number=1,
                 last_stride=2,
                 block=AIBNBottleneck,
                 layers=[3, 4, 6, 3],
                 init_weight=0.1):
        self.inplanes = 64
        super(TNormResNet, self).__init__()
        self.domain_number = domain_number
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # add missed relu
        self.relu_after_tnorm = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_normal(Bottleneck, 64, layers[0])
        self.tnorm1 = TNorm(256, domain_number)
        self.layer2 = self._make_layer_normal(Bottleneck,
                                              128,
                                              layers[1],
                                              stride=2)
        self.tnorm2 = TNorm(512, domain_number)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       adaptive_weight=None,
                                       init_weight=init_weight)

        self.tnorm3 = TNorm(1024, domain_number)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=last_stride,
                                       adaptive_weight=None,
                                       init_weight=init_weight)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    adaptive_weight=None,
                    init_weight=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                AIBNorm2d(
                    planes * block.expansion,
                    adaptive_weight=adaptive_weight,
                    generate_weight=True,
                    init_weight=init_weight
                ),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  adaptive_weight=adaptive_weight,
                  generate_weight=True,
                  init_weight=init_weight))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(self.inplanes,
                          planes,
                          adaptive_weight=adaptive_weight,
                          generate_weight=True,
                          init_weight=init_weight))
            else:
                layers.append(
                    block(self.inplanes,
                          planes,
                          adaptive_weight=adaptive_weight,
                          generate_weight=True,
                          init_weight=init_weight))

        return nn.Sequential(*layers)

    def _make_layer_normal(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, domain_index=None, convert=False):
        if convert:
            selected_domain = np.random.randint(0,
                                                self.domain_number,
                                                size=(x.size(0)))
        else:
            selected_domain = None
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # add missed relu
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.tnorm1(x, domain_index, convert, selected_domain)
        x = self.layer2(x)
        x = self.tnorm2(x, domain_index, convert, selected_domain)
        x = self.layer3(x)
        x = self.tnorm3(x, domain_index, convert, selected_domain)
        x = self.layer4(x)

        x = self.adaptive_pool(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for k, v in param_dict.items():
            if k in self.state_dict().keys():
                self.state_dict()[k].copy_(v)

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
