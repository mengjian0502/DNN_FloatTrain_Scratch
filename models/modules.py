"""
Forward pass modules
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from qtorch.quant import float_quantize
from torch.nn import init
from .function import *

class HWconv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, 
                dilation=1, groups=1, bias=False, lr=0.1, momentum=0.9, use_relu=True, relu_inplace=True, low_precision=True):
        super(HWconv2d, self).__init__()
        self.lp = low_precision
        if self.lp:
            # low precision values
            self.w_man = 2
            self.w_exp = 5

            self.g_man = 2
            self.g_exp = 5
            
            self.x_man = 2
            self.x_exp = 5
            
            # accumulation precisions
            self.y_exp = 5
            self.y_man = 10

            self.c_tc = 8   # tensor core channels

        # initialize
        fan_in = out_channels * kernel_size * kernel_size
        weight_std = np.sqrt(2. / fan_in)
        self.weight= torch.empty(out_channels, in_channels, kernel_size, kernel_size).normal_(mean=0.0, std=weight_std).cuda()
        if bias:
            self.bias = torch.empty(out_channels).normal_(mean=0.0, std=weight_std).cuda()
        else:
            self.bias = torch.zeros(out_channels).cuda()

        # convolution
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # relu
        self.use_relu = use_relu
        self.relu_inplace = relu_inplace

        # gradient
        self.w_grad = torch.zeros_like(self.weight).cuda()
        
        # SGD with momentum
        self.momentum = momentum
        self.lr = lr
        self.w_vel = torch.zeros_like(self.weight).cuda()
    
    def relu(self, input, inplace=False, bp=False):
        if bp:
            out = torch.ceil(torch.clamp(input, min=0, max=1))
            return out

        out = F.relu(input, inplace=inplace)
        return out

    def conv(self, input):
        c_out, c_in, k, _ = self.weight.size()
        
        if c_in <= 3:
            self.c_tc = c_in
        
        c_iter = c_in // self.c_tc

        # compute the original output
        y_org = F.conv2d(input, self.weight.data, self.bias, self.stride, self.padding, self.dilation, self.groups)
        y_total = torch.zeros_like(y_org.data)

        # low precision floating point
        if self.lp:
            self.wlp = float_quantize(self.weight.data, exp=self.w_exp, man=self.w_man, rounding="nearest")
            if c_in == 3:
                self.xlp = float_quantize(self.input, exp=self.x_exp, man=self.x_man, rounding="nearest")
            else:
                self.xlp = self.input 


        for ii in range(c_iter):
            maskc = torch.zeros_like(self.weight.data)
            maskc[:, ii*self.c_tc:(ii+1)*self.c_tc, :, :] = 1 # select 8 input channels
            for ih in range(k):
                for iw in range(k):
                    maskk = torch.zeros_like(self.weight.data)
                    maskk[:,:,ih,iw] = 1
                    mask = maskc * maskk    # combined mask

                    # low precision output
                    y = F.conv2d(self.xlp, self.wlp.data*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                    
                    # high precision accumulation
                    y_total += y
        return y_total
    
    def forward(self, input: Tensor):
        self.input = input.cuda()

        # convolution
        y = self.conv(self.input)
        
        if self.use_relu:
            self.out = self.relu(y, inplace=self.relu_inplace, bp=False)
        else:
            self.out = y

        return self.out
    
    def zero_grad(self):
        self.w_grad.fill_(0.)

    def feed_backward(self, output_grad, batch_idx):
        r"""
        Gradient computation based on 1 image
        """
        if len(output_grad.size()) < 4:
            output_grad.unsqueeze(0)
        
        if self.use_relu:
            try:
                relu_mask = torch.ceil(torch.clamp(self.out[batch_idx].unsqueeze(0), min=0, max=1))
            except:
                import pdb;pdb.set_trace()
        
            try:
                output_grad = output_grad * relu_mask
            except:
                import pdb;pdb.set_trace()

        output_grad_t = output_grad.transpose(0,1)
        input_i = self.input[batch_idx].unsqueeze(0)
        input_i_t = input_i.transpose(0,1)
        
        # flip the weight
        weight_flip = torch.flip(self.weight, [2,3])
        weight_t = weight_flip.transpose(0,1)

        
        dw = F.conv2d(input_i_t, output_grad_t, stride=self.stride, padding=self.padding)
        dout = F.conv2d(output_grad, weight_t, stride=self.stride, padding=self.padding)

        # low precision gradient
        if self.lp:
            dw = float_quantize(dw, exp=self.g_exp, man=self.g_man, rounding="nearest")
            dout = float_quantize(dout, exp=self.g_exp, man=self.g_man, rounding="nearest")

        self.w_grad += dw.transpose(0,1)
        self.w_grad = float_quantize(self.w_grad, exp=self.y_exp, man=self.y_man, rounding="nearest")
        return dout

    def weight_update(self):
        self.w_vel = self.momentum * self.w_vel + self.w_grad
        
        # low precision velocity
        if self.lp:
            self.w_vel = float_quantize(self.w_vel, exp=self.g_exp, man=self.g_man, rounding="nearest")
        self.weight -= self.lr * self.w_vel
        


class FC(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=0.1, momentum=0.9):
        super(FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
    
        weight_bound = np.sqrt(6. / (in_features + out_features))
        self.weight = torch.empty(out_features, in_features).uniform_(-weight_bound, weight_bound).cuda()

        if bias:
            self.bias = torch.empty(out_features).uniform_(-weight_bound, weight_bound).cuda()
        else:
            self.bias = torch.zeros(out_features).cuda()

        # Gradient
        self.w_grad = torch.zeros_like(self.weight).cuda()
        self.b_grad = torch.zeros_like(self.bias).cuda()
        
        # SGD with momentum
        self.momentum = momentum
        self.lr = lr
        self.w_vel = torch.zeros_like(self.weight).cuda()
        self.b_vel = torch.zeros_like(self.bias).cuda()
    
    def zero_grad(self):
        self.w_grad.fill_(0.)
        self.b_grad.fill_(0.)

    def forward(self, input):
        self.input = input.cuda()
        self.out  = F.linear(self.input, self.weight, self.bias)
        return self.out
    
    def feed_backward(self, out_gradient, batch_idx):
        r"""
        Gradient computation based on 1 image

        dw = out_grad.T @ input
        db = out_grad.sum()

        out_grad: (1, out_features)
        input: (1, in_features)
        """
        y_i = self.input[batch_idx, :].view(1, -1)
        out_grad_transpose = out_gradient.transpose(0,1)

        self.w_grad += torch.matmul(out_grad_transpose, y_i)
        self.b_grad += out_gradient.view(self.bias.size())
        dout = torch.matmul(out_gradient, self.weight.data)
        return dout
    
    def weight_update(self):
        r"""
        Update the weight after the gradient accumulation
        """
        self.w_vel = self.momentum * self.w_vel + self.w_grad
        self.b_vel = self.momentum * self.b_vel + self.b_grad
        
        self.weight -= self.lr * self.w_vel
        self.bias -= self.lr * self.b_vel
        

class MaxPooling(nn.Module):
    r"""
    Implementing max pooling with pytorch function
    """
    def __init__(self, kernel_size, stride):
        super(MaxPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.name = 'MaxPool2d'
        self.type = 'pool'
    
    def forward(self, input):
        self.input = input
        self.out = maxpool_2d(input, f=self.kernel_size, s=self.stride)
        self.N, self.C, self.H, self.W = self.out.size()
        return self.out
    
    def feed_backward(self, out_gradient, batch_idx):
        r"""
        Gradient computation based on 1 image
        """
        if len(out_gradient.size()) == 2:
            out_gradient = out_gradient.view(1, self.C, self.H, self.W)   # if the gradient flow back from the flatten

        input_i = self.input[batch_idx].unsqueeze(0)

        dpool = maxpoolBackward(out_gradient, input_i, f=self.kernel_size, s=self.stride)
        return dpool

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.training = True

        # running statistics
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        # affine transformation
        self.weight = torch.Tensor(num_features)
        self.bias = torch.Tensor(num_features)

        # initialize the weights and bias
        init.ones_(self.weight)
        init.zeros_(self.bias)

        # gradient accumulation
        self.w_grad = torch.zeros_like(self.weight).cuda()
        self.b_grad = torch.zeros_like(self.bias).cuda()
    
    def forward(self, input:Tensor):        
        self.input = input.cuda()
        
        if self.training:
            self.mean = self.input.mean([0,2,3])
            self.var = self.input.var([0,2,3])
            self.std = torch.sqrt(self.var + self.eps)

            # update running statistics
            self.running_mean = self.momentum * self.mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * self.std  + (1 - self.momentum) * self.running_var
        else:
            self.mean = self.running_mean
            self.var = self.running_var
            self.std = torch.sqrt(self.var + self.eps)

        self.xmu = self.input - self.mean[None, :, None, None]
        self.inv_std = 1 / (self.std[None, :, None, None])
        self.output = self.xmu.mul(self.inv_std)
        
        if self.affine:
            self.prod = self.output * self.weight[None, :, None, None]
            self.output = self.prod + self.bias[None, :, None, None]
        return self.output
    
    def feed_backward(self, output_grad, batch_idx):
        prod_i = self.prod[batch_idx, :].view(1, -1)
        
        # accumulate the gradient along batch dim
        self.b_grad += output_grad.view(self.bias.size())
        self.w_grad += output_grad.mul(prod_i)

        



class MSELoss(nn.Module):
    def __init__(self, num_classes):
        super(MSELoss, self).__init__()
        self.num_classes = int(num_classes)
        self.exp = 5
        self.man = 2

        self.name = 'MSELoss'
        self.type = 'lossFunc'
    
    def feed_forward(self, output, target):
        self.batch = output.size(0)
        self.output = output
        self.target = target

        assert self.num_classes == output.size(1), "Number of classes and the output dim must be identical"

        loss = F.mse_loss(output, target)
        loss = float_quantize(loss, exp=self.exp, man=self.man, rounding="nearest")
        return output, loss

    def feed_backward(self):
        """
        evaluate the gradient w.r.t output
        """ 
        rough_grad = 2 * (self.output - self.target) / (self.output.size(0)*self.output.size(1))
        return rough_grad
        
    def weight_grad(self, groups=0):
        pass

    def apply_weight_grad(self, learning_rate=1.0, momentum=0.5,
                        batch_size=100, last_group=False):
        pass


