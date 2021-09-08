
"""
vgg model 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HWconv2d_Torch(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(HWconv2d_Torch, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_man = 2
        self.w_exp = 5
        
        self.a_man = 2
        self.a_exp = 5

        self.c_tc = 8   # tensor core channels
        self.c_mem = 16 # maximum input channel chunk for memory

        self.y_exp = 5
        self.y_man = 10

    def forward(self, input: Tensor) -> Tensor:
        N, C, H, W = input.size()
        c_out, c_in, k, _ = self.weight.size()

        if c_in <= 3:
            self.c_tc = c_in
        
        c_iter = c_in // self.c_tc

        # compute the original output
        y_org = F.conv2d(input, self.weight.data, self.bias, self.stride, self.padding, self.dilation, self.groups)
        y_hw = torch.zeros_like(y_org.data)

        for ii in range(c_iter):
            maskc = torch.zeros_like(self.weight.data)
            maskc[:, ii*self.c_tc:(ii+1)*self.c_tc, :, :] = 1 # select 8 input channels
            for ih in range(k):
                for iw in range(k):
                    maskk = torch.zeros_like(self.weight.data)
                    maskk[:,:,ih,iw] = 1
                    mask = maskc * maskk    # combined mask
                    y = F.conv2d(input, self.weight*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                    y_hw += y   # high precision accumulation
        return y_hw



def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = HWconv2d_Torch(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                # layers += [conv2d]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    7: [32, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        if depth == 7:
            self.classifier = nn.Sequential(
                nn.Linear(256, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class vgg7_torch:
    base = VGG
    args = list()
    kwargs={'depth':7, 'batch_norm':False}