import torch 
import torch.nn as nn
from bidirectional_spectral_norm import bidirectional_spectral_norm
from modules.norm.SNlayers import SNLinear
import copy


__all__ = ['citation', 'spectral_norm', 'SNConv1d', 'SNConv2d', 'SNLinear', 'MeanSpectralNorm1d', 'MeanSpectralNorm2d', 'MeanSpectralNorm3d', 'add_BSN']

'''
Spectral Normalization for Generative Adversarial Networks
Generalizable Adversarial Training via Spectral Normalization
Mean Spectral Normalization of Deep Neural Networks for Embedded Automation
Why Spectral Normalization Stabilizes GANs: Analysis and Improvements
'''
citation = OrderedDict({'Spectral Normalization': {'Title': 'Mean Spectral Normalization of Deep Neural Networks for Embedded Automation',
                                                   'Authors': 'Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon',
                                                   'Year': '2018',
                                                   'Journal': 'ECCV',
                                                   'Institution': 'Korea Advanced Institute of Science and Technology, Lunit Inc., and Adobe Research',
                                                   'URL': 'https://arxiv.org/pdf/1807.06521.pdf',
                                                   'Notes': 'Added the possiblity to switch from SE to eCA in the ChannelGate and updated deprecated sigmoid',
                                                   'Source Code': 'Modified from: https://github.com/Jongchan/attention-module'},
                        'Mean Spectral Normalization': {'Title': 'Mean Spectral Normalization of Deep Neural Networks for Embedded Automation',
                                                   'Authors': 'Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon',
                                                   'Year': '2018',
                                                   'Journal': 'ECCV',
                                                   'Institution': 'Korea Advanced Institute of Science and Technology, Lunit Inc., and Adobe Research',
                                                   'URL': 'https://arxiv.org/pdf/1807.06521.pdf',
                                                   'Notes': 'Added the possiblity to switch from SE to eCA in the ChannelGate and updated deprecated sigmoid',
                                                   'Source Code': 'Modified from: https://github.com/Jongchan/attention-module'},
                        'Bidirectional Spectral Norm': {'Title': 'Mean Spectral Normalization of Deep Neural Networks for Embedded Automation',
                                                   'Authors': 'Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon',
                                                   'Year': '2018',
                                                   'Journal': 'ECCV',
                                                   'Institution': 'Korea Advanced Institute of Science and Technology, Lunit Inc., and Adobe Research',
                                                   'URL': 'https://arxiv.org/pdf/1807.06521.pdf',
                                                   'Notes': 'Added the possiblity to switch from SE to eCA in the ChannelGate and updated deprecated sigmoid',
                                                   'Source Code': 'Modified from: https://github.com/Jongchan/attention-module'}})




'''
Code modified from: https://github.com/AntixK/mean-spectral-norm
By: Anand Krishnamoorthy
09.03.2020
'''

def _L2Norm(v, eps=torch.tensor(1e-12)):
    return v/(torch.norm(v) + eps)

def spectral_norm(W, u=None, Num_iter=1):
    '''
    Spectral Norm of a Matrix is its maximum singular value.
    This function employs the Power iteration procedure to
    compute the maximum singular value.

    JTL: 10.15.2020
        Original implementation set Num_iter=100 for every single call. This
        causes it to take 2.5-3x as long to train. PyTorch's torch.nn.utils.spectral_norm
        is MUCH faster. They set Num_iter=1. There is an assumed smoothness between
        iterations, which should be why PyTorch uses the smaller value. Therefore,
        100 iterations will be used initially to set the stage, and from then on it
        will use only 1 iteration per call

    :param W: Input(weight) matrix - autograd.variable
    :param u: Some initial random vector - FloatTensor
    :param Num_iter: Number of Power Iterations
    :return: Spectral Norm of W, orthogonal vector _u
    '''
    if not Num_iter >= 1:
        raise ValueError("Power iteration must be a positive integer")

    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0.,1.).to(W.device) #.cuda()
        Num_iter = 100

    _u = u
    _v = torch.tensor(0)
    for _ in range(Num_iter):
        _v = _L2Norm(torch.matmul(_u, W.data))
        _u = _L2Norm(torch.matmul(_v, torch.transpose(W.data, 0, 1)))
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0,1)) * _v)
    # sigma = torch.dot(_u, torch.mv(torch.transpose(W.data, 0,1), _v))
    return sigma, _u


class mSNConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanSpectralNormConv1d, self).__init__()

        self.conv = spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.momentum = 0.1

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        x = self.conv(input)

        # Recenter the pre-activations using running mean
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
        else:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            y = y - mu.view(-1, 1)

        y = y + self.bias
        return y.view(return_shape).transpose(0, 1)

class mSNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanSpectralNormConv2d, self).__init__()

        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.momentum = 0.1

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        x = self.conv(input)

        # Recenter the pre-activations using running mean
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
        else:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            y = y - mu.view(-1, 1)

        y = y + self.bias
        return y.view(return_shape).transpose(0, 1)

class mSNConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanSpectralNormConv3d, self).__init__()

        self.conv = spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.momentum = 0.1

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        x = self.conv(input)

        # Recenter the pre-activations using running mean
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
        else:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            y = y - mu.view(-1, 1)

        y = y + self.bias
        return y.view(return_shape).transpose(0, 1)

class SNConv1d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(SNConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, 'reflect')
        self.u = None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda(), requires_grad=True)

    def forward(self, input):
        #print("renorm:",self.renorm.data)
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = spectral_norm(w_mat, self.u)
        self.u = _u
        self.weight.data = self.renorm.data * self.weight.data / sigma
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'reflect')
        self.u = None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda(), requires_grad=True)

    def forward(self, input):
        #print("renorm:",self.renorm.data)
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = spectral_norm(w_mat, self.u)
        self.u = _u
        self.weight.data = self.renorm.data * self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNConv3d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(SNConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, 'reflect')
        self.u = None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda(), requires_grad=True)

    def forward(self, input):
        #print("renorm:",self.renorm.data)
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = spectral_norm(w_mat, self.u)
        self.u = _u
        self.weight.data = self.renorm.data * self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = None#torch.tensor(-1) # None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda(), requires_grad=True)
        
    def forward(self, input):
        w_mat = self.weight
        sigma, _u = spectral_norm(w_mat, self.u)
        self.u = _u
        self.weight.data = (self.weight.data / sigma) * self.renorm.data
        return F.linear(input, self.weight, self.bias)
