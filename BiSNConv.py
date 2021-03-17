import torch
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair, _single, _triple
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


__all__ = ['citation', 'spectral_norm', 'BiSNConv1d', 'BiSNConv2d', 'BiSNConv3d']

'''
Spectral Normalization for Generative Adversarial Networks
Generalizable Adversarial Training via Spectral Normalization
Mean Spectral Normalization of Deep Neural Networks for Embedded Automation
Why Spectral Normalization Stabilizes GANs: Analysis and Improvements
'''
citation = OrderedDict({'Bidirectional Spectral Norm': {'Title': 'Mean Spectral Normalization of Deep Neural Networks for Embedded Automation',
                                                        'Authors': 'Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon',
                                                        'Year': '2018',
                                                        'Journal': 'ECCV',
                                                        'Institution': 'Korea Advanced Institute of Science and Technology, Lunit Inc., and Adobe Research',
                                                        'URL': 'https://arxiv.org/pdf/1807.06521.pdf',
                                                        'Notes': 'Added the possiblity to switch from SE to eCA in the ChannelGate and updated deprecated sigmoid',
                                                        'Source Code': 'Modified from: https://github.com/Jongchan/attention-module'
                                                        'Modified by': 'Ing. John LaMaster'}})


'''
Code modified from: https://github.com/AntixK/mean-spectral-norm
By: Anand Krishnamoorthy
09.03.2020
'''


class BiSNConv1d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(BSNConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, 'reflect')
        self.u0, self.u1 = None, None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda(), requires_grad=True)

    def forward(self, input):
        w_mat0 = self.weight.view(self.weight.size(0), -1)
        w_mat1 = self.weight.transpose(1, 0, 2).view(self.weight.size(0), -1)
        sigma0, _u0 = spectral_norm(w_mat0, self.u0)
        sigma1, _u1 = spectral_norm(w_mat1, self.u1)
        self.u0, self.u1 = _u0, _u1
        sigma = (sigma0 + sigma1) / 2
        self.weight.data = self.renorm.data * self.weight.data / sigma
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class BiSNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BSNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'reflect')
        self.u0, self.u1 = None, None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda(), requires_grad=True)

    def forward(self, input):
        w_mat0 = self.weight.view(self.weight.size(0), -1)
        w_mat1 = self.weight.transpose(1, 0, 2, 3).view(self.weight.size(0), -1)
        sigma0, _u0 = spectral_norm(w_mat0, self.u0)
        sigma1, _u1 = spectral_norm(w_mat1, self.u1)
        self.u0, self.u1 = _u0, _u1
        sigma = (sigma0 + sigma1) / 2
        self.weight.data = self.renorm.data * self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class BiSNConv3d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(BSNConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, 'reflect')
        self.u0, self.u1 = None, None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda(), requires_grad=True)

    def forward(self, input):
        w_mat0 = self.weight.view(self.weight.size(0), -1)
        w_mat1 = self.weight.transpose(1, 0, 2, 3).view(self.weight.size(0), -1)
        sigma0, _u0 = spectral_norm(w_mat0, self.u0)
        sigma1, _u1 = spectral_norm(w_mat1, self.u1)
        self.u0, self.u1 = _u0, _u1
        sigma = (sigma0 + sigma1) / 2
        self.weight.data = self.renorm.data * self.weight.data / sigma
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def _L2Norm(v, eps=1e-12):
    return v/(torch.norm(v) + eps)

def spectral_norm(W, u=None, Num_iter=100):
    '''
    Spectral Norm of a Matrix is its maximum singular value.
    This function employs the Power iteration procedure to
    compute the maximum singular value.

    :param W: Input(weight) matrix - autograd.variable
    :param u: Some initial random vector - FloatTensor
    :param Num_iter: Number of Power Iterations
    :return: Spectral Norm of W, orthogonal vector _u
    '''
    if not Num_iter >= 1:
        raise ValueError("Power iteration must be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0,1).to(W.device) #.cuda()
    _u = u
    for _ in range(Num_iter):
        _v = _L2Norm(torch.matmul(_u, W.data))
        _u = _L2Norm(torch.matmul(_v, torch.transpose(W.data,0, 1)))
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0,1)) * _v)
    return sigma, _u
      
