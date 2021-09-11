"""
DNN Modules

The feed backward will be completed in the batch-wise operation
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from qtorch.quant import float_quantize
from torch.nn import init
from .function import *

class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, 
                dilation=1, groups=1, bias=False, lr=0.1, momentum=0.9, use_relu=True, relu_inplace=True, low_precision=False):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

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

        # gradient
        self.w_grad = torch.zeros_like(self.weight).cuda()
        
        # SGD with momentum
        self.momentum = momentum
        self.lr = lr
        self.w_vel = torch.zeros_like(self.weight).cuda()

    def conv(self, input):
        c_out, c_in, k, _ = self.weight.size()
        
        if c_in <= 3:
            self.c_tc = c_in
        
        c_iter = c_in // self.c_tc

        # compute the original output
        odim = math.floor((input.size(2) + 2*self.padding - self.dilation * (self.weight.size(2)-1)-1)/self.stride + 1)
        y_total = torch.zeros((input.size(0), self.weight.size(0), odim, odim)).cuda()

        # low precision floating point
        if self.lp:
            self.weight = float_quantize(self.weight.data, exp=self.w_exp, man=self.w_man, rounding="nearest")
            if c_in != 3:
                self.input = float_quantize(self.input, exp=self.x_exp, man=self.x_man, rounding="nearest")
            else:
                self.input = self.input 

        for ii in range(c_iter):
            maskc = torch.zeros_like(self.weight.data)
            maskc[:, ii*self.c_tc:(ii+1)*self.c_tc, :, :] = 1 # select 8 input channels
            for ih in range(k):
                for iw in range(k):
                    maskk = torch.zeros_like(self.weight.data)
                    maskk[:,:,ih,iw] = 1
                    mask = maskc * maskk    # combined mask

                    # low precision output
                    y = F.conv2d(self.input, self.weight.data*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                    
                    # high precision accumulation
                    y_total += y
                    if self.lp:
                        y_total = float_quantize(y_total, exp=self.y_exp, man=self.y_man, rounding="nearest")
        return y_total
    
    def forward(self, input: Tensor):
        self.input = input.cuda()
        # convolution
        self.out = self.conv(self.input)
        return self.out
    
    def zero_grad(self):
        self.w_grad.fill_(0.)

    def feed_backward(self, output_grad):
        r"""
        Gradient computation based on 1 image
        """
        if len(output_grad.size()) < 4:
            output_grad.unsqueeze(0)

        output_grad_t = output_grad.transpose(0,1)
        input_i = self.input
        input_i_t = input_i.transpose(0,1)
        
        # flip the weight
        weight_flip = torch.flip(self.weight, [2,3])
        weight_t = weight_flip.transpose(0,1)

        dout = F.conv2d(output_grad, weight_t, stride=self.stride, padding=self.padding)
        
        # output gradient
        if self.lp:
            dout = float_quantize(dout, exp=self.g_exp, man=self.g_man, rounding="nearest")
        
        # weight gradient accumulation
        if self.lp:
            dw = torch.zeros_like(self.weight).transpose(0,1)
            for batch_idx in range(output_grad.size(0)):
                input_i = self.input[batch_idx].unsqueeze(0)
                output_grad_i = output_grad[batch_idx].unsqueeze(0)

                input_i_t = input_i.transpose(0,1)
                output_grad_i_t = output_grad_i.transpose(0,1)

                dwi = F.conv2d(input_i_t, output_grad_i_t, stride=self.stride, padding=self.padding)
                dw = float_quantize(dw, exp=self.y_exp, man=self.y_man, rounding="nearest")
                dw += dwi
        else:
            dw = F.conv2d(input_i_t, output_grad_t, stride=self.stride, padding=self.padding)
        
        self.w_grad = dw.transpose(0,1)
        return dout

    def weight_update(self):
        self.w_vel = self.momentum * self.w_vel + self.w_grad
        
        # low precision velocity
        if self.lp:
            self.w_vel = float_quantize(self.w_vel, exp=self.g_exp, man=self.g_man, rounding="nearest")
        self.weight -= self.lr * self.w_vel
    
    def extra_repr(self):
        return super(Conv2d, self).extra_repr() + 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, lp={}'.format(self.in_channels, 
                self.out_channels, self.kernel_size, self.stride, self.padding, self.lp)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=0.1, momentum=0.9, low_precision=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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
        
        # low precision floating point
        if self.lp:
            self.weight = float_quantize(self.weight.data, exp=self.w_exp, man=self.w_man, rounding="nearest")
            self.input = float_quantize(self.input, exp=self.x_exp, man=self.x_man, rounding="nearest")


        self.out  = F.linear(self.input, self.weight, self.bias)
        return self.out
    
    def feed_backward(self, out_gradient):
        r"""
        Gradient computation based on 1 image

        dw = out_grad.T @ input
        db = out_grad.sum()

        out_grad: (1, out_features)
        input: (1, in_features)
        """
        out_grad_transpose = out_gradient.transpose(0,1)
        self.w_grad = torch.matmul(out_grad_transpose, self.input)
        self.b_grad = out_gradient.sum(dim=0).view(self.bias.size())
        
        # output gradient
        dout = torch.matmul(out_gradient, self.weight.data)
        if self.lp:
            dout = float_quantize(dout, exp=self.g_exp, man=self.g_man, rounding="nearest")
        return dout
    
    def weight_update(self):
        r"""
        Update the weight after the gradient accumulation
        """
        self.w_vel = self.momentum * self.w_vel + self.w_grad
        self.b_vel = self.momentum * self.b_vel + self.b_grad
        
        self.weight -= self.lr * self.w_vel
        self.bias -= self.lr * self.b_vel

    def extra_repr(self):
        return super(Linear, self).extra_repr() + 'in_features={}, out_features={}, lp={}'.format(self.in_features, self.out_features, self.lp)
        

class MaxPool2d(nn.Module):
    r"""
    Implementing max pooling with pytorch function
    """
    def __init__(self, kernel_size, stride, low_precision=False):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.name = 'MaxPool2d'
        self.type = 'pool'

        self.lp = low_precision
        if self.lp:
            self.g_man = 2
            self.g_exp = 5
            
    def forward(self, input):
        self.input = input
        self.out = maxpool_2d(input, f=self.kernel_size, s=self.stride)
        self.N, self.C, self.H, self.W = self.out.size()
        return self.out
    
    def feed_backward(self, out_gradient):
        r"""
        Gradient computation based on 1 image
        """
        if len(out_gradient.size()) == 2:
            out_gradient = out_gradient.view(-1, self.C, self.H, self.W)   # if the gradient flow back from the flatten

        dout = maxpoolBackward(out_gradient, self.input, f=self.kernel_size, s=self.stride)
        if self.lp:
            dout = float_quantize(dout, exp=self.g_exp, man=self.g_man, rounding="nearest")
        return dout

    def extra_repr(self):
        return super(MaxPool2d, self).extra_repr() + 'kernel_size={}, stride={}, lp={}'.format(self.kernel_size, self.stride, self.lp)

class BatchNorm(nn.Module):
    def __init__(self, num_features, batch_size=128, eps=1e-5, m=0.1, lr=0.1, momentum=0.9, affine=True, use_relu=True, relu_inplace=True):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.m = m
        self.affine = affine
        self.training = True
        self.batch_size = batch_size

        # running statistics
        self.running_mean = torch.zeros(num_features).cuda()
        self.running_var = torch.ones(num_features).cuda()

        # affine transformation
        self.weight = torch.Tensor(num_features)
        self.bias = torch.Tensor(num_features)

        # initialize the weights and bias
        init.ones_(self.weight)
        init.zeros_(self.bias)

        # gradient accumulation
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


    def forward(self, input:Tensor):        
        self.input = input.cuda()
        
        # if self.training:
        #     self.mean = self.input.mean([0,2,3])
        #     self.var = self.input.var([0,2,3])
        #     self.std = torch.sqrt(self.var + self.eps)

        #     # update running statistics
        #     self.running_mean = self.momentum * self.mean + (1 - self.m) * self.running_mean
        #     self.running_var = self.momentum * self.std  + (1 - self.m) * self.running_var
        # else:
        #     self.mean = self.running_mean
        #     self.var = self.running_var
        #     self.std = torch.sqrt(self.var + self.eps)
        
        self.mean = self.input.mean([0,2,3])
        self.var = self.input.var([0,2,3])
        self.std = torch.sqrt(self.var + self.eps)

        # update running statistics
        self.running_mean = self.momentum * self.mean + (1 - self.m) * self.running_mean
        self.running_var = self.momentum * self.std  + (1 - self.m) * self.running_var

        self.inv_std = 1 / (self.std[None, :, None, None])
        self.xmu = self.input - self.mean[None, :, None, None]
        self.xhat = self.xmu.mul(self.inv_std)
        
        if self.affine: 
            self.output = self.xhat * self.weight[None, :, None, None] + self.bias[None, :, None, None]    
        return self.output
    
    def feed_backward(self, output_grad):
        self.b_grad = output_grad.sum(dim=0).sum([1,2])
        self.w_grad = output_grad.mul(self.xhat).sum(dim=0).sum([1,2])

        ppc = self.input.size(2) * self.input.size(3)
        dinv_std = self.xmu
        
        dinvvar = (1.0 / (2.0 * torch.sqrt(1/self.var[None, :, None, None]))) * dinv_std
        dvar = (-1.0 / (self.var[None, :, None, None] ** 2)) * dinvvar
        ddenominator = (self.input - self.mean[None, :, None, None]) * (2 * (ppc - 1) / ppc ** 2) * dvar
        
        dcentered = torch.sqrt(1/self.var)
        dnumerator = (1.0 - 1.0 / ppc) * dcentered[None, :, None, None]
        
        dX = ddenominator + dnumerator
        dout = dX * output_grad
        return dout

    def weight_update(self):
        self.w_vel = self.momentum * self.w_vel + self.w_grad
        self.b_vel = self.momentum * self.b_vel + self.b_grad

        self.weight -= self.lr * self.w_vel
        self.bias -= self.lr * self.b_vel

    def extra_repr(self):
        return super(BatchNorm, self).extra_repr() + 'num_features={}, eps={}'.format(self.num_features, self.eps)

class ReLU(nn.Module):
    def __init__(self, inplace=True):
        super(ReLU, self).__init__()
        self.inplace = inplace
    
    def forward(self, input):
        self.output = F.relu(input, inplace=self.inplace)
        return self.output
    
    def feed_backward(self, output_grad):
        relu_mask = torch.ceil(torch.clamp(self.output, min=0, max=1))
        dout = output_grad * relu_mask
        return dout


class MSE(nn.Module):
    def __init__(self, num_classes, low_precision=False):
        super(MSE, self).__init__()
        self.num_classes = int(num_classes)
        self.exp = 5
        self.man = 2

        self.lp = low_precision

        self.name = 'MSELoss'
        self.type = 'lossFunc'
    
    def feed_forward(self, output, target):
        self.batch = output.size(0)
        self.output = output
        self.target = target

        assert self.num_classes == output.size(1), "Number of classes and the output dim must be identical"

        loss = F.mse_loss(output, target)
        if self.lp:
            loss = float_quantize(loss, exp=self.exp, man=self.man, rounding="nearest")
        
        return output, loss

    def feed_backward(self):
        """
        evaluate the gradient w.r.t output
        """ 
        dout = 2 * (self.output - self.target) / (self.output.size(0)*self.output.size(1))
        
        # output gradient
        if self.lp:
            dout = float_quantize(dout, exp=self.exp, man=self.man, rounding="nearest")
        return dout
        
    def weight_grad(self, groups=0):
        pass

    def apply_weight_grad(self, learning_rate=1.0, momentum=0.5,
                        batch_size=100, last_group=False):
        pass

    def extra_repr(self):
        return super(MSE, self).extra_repr() + 'lp={}'.format(self.lp)



