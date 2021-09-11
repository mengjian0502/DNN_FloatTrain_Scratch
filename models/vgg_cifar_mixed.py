r"""

VGG on CIFAR-10

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules_mixed import *


def make_layers(cfg, lr=0.1, momentum=0.9):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1, lr=lr, momentum=momentum, use_relu=True, relu_inplace=True)
            layers += [conv2d]
            in_channels = v
    return layers

cfg = {
    7: [32, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}

class VGGMixed(nn.Module):
    def __init__(self, num_classes=10, depth=16, lr=0.1, momentum=0.9):
        super(VGGMixed, self).__init__()
        self.features = make_layers(cfg[depth], lr, momentum)
        self.layers = len(self.features)

        self.model = nn.Sequential(*self.features)
        
        if depth == 7:
            self.fc = Linear(in_features=256, out_features=num_classes, bias=True, lr=lr, momentum=momentum)
        
        self.loss = MSE(num_classes)

    def feed_forward(self, x, target):
        for ii in range(self.layers):
            x = self.features[ii].forward(x)
        x = x.view(x.size(0), -1)
        self.pred = self.fc.forward(x)
        self.out, self.err = self.loss.feed_forward(self.pred, target)
        return self.out, self.err
    
    def zero_grad(self):
        self.fc.zero_grad()
        for jj in range(self.layers):
            module = self.features[self.layers - 1 - jj]
            if isinstance(module, Conv2d):
                module.zero_grad()
    
    def feed_backward(self):
        dloss = self.loss.feed_backward()
        dfc = self.fc.feed_backward(dloss)
        grad = dfc
        for jj in range(self.layers):
            module = self.features[self.layers - 1 - jj]
            dout = module.feed_backward(grad)
            grad = dout
    
    def weight_update(self):
        self.loss.apply_weight_grad()
        self.fc.weight_update()
        for jj in range(self.layers):
            module = self.features[self.layers - 1 - jj]
            if isinstance(module, Conv2d):
                module.weight_update()


class vgg7mixed:
    base = VGGMixed
    args = list()
    kwargs={'depth':7}
    
if __name__ == '__main__':
    model_cfg = vgg7mixed
    net = vgg7mixed.base(*model_cfg.args, **model_cfg.kwargs).cuda()
    net = net.cuda()
    test_x = torch.randn(1,3,32,32).cuda()
    test_y = torch.randn(1,10).cuda()
    out, err = net.feed_forward(test_x, test_y)
    print(out.size())

    net.feed_backward()
    