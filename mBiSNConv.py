import torch 
import torch.nn as nn
from bidirectional_spectral_norm import bidirectional_spectral_norm
from spectral_norm_modules import SNLinear
import copy


def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_num_gen(gen):
    return sum(1 for x in gen)

def add_mBiSN(model):
    modules = copy.deepcopy(model._modules)
    for m in modules:
        child = modules[m]
        if is_leaf(child):
            if isinstance(child, nn.Conv1d):
                model._modules[m] = mBiSNConv1dmodule(child)
                del child
            elif isinstance(child, nn.Conv2d):
                model._modules[m] = mBiSNConv2dmodule(child)
                del child
            elif isinstance(child, nn.Conv3d):
                model._modules[m] = mBiSNConv3dmodule(child)
                del child
            elif isinstance(child, nn.Linear):
                try:
                    bias = True if not child.bias==False else False
                except RuntimeError:
                    bias = False
                model._modules[m] = SNLinear(child.in_features, child.out_features, True)
            elif isinstance(child, nn.BatchNorm1d):
                model._modules[m] = Empty()
        else:
            add_mBiSN(model._modules[m])

class Empty(nn.Module):
    def __init__(self):
        super(Empty, self).__init__()

    def __repr__(self):
        return ''

    def forward(self, input):
        return input

class mBiSNConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='reflect'):
        super(mBiSNConv1d, self).__init__()

        self.dim = 1
        self.conv = bidirectional_spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode))

        self.bias = nn.Parameter(torch.zeros(self.out_channels,1), requires_grad=True)
        self.register_buffer('running_mean', torch.zeros(self.out_channels))
        self.momentum = 0.1

    def __repr__(self):
        s = 'mBiSNConv{}d({}, {}, kernel_size={}, stride={}'.format(self.dim, self.conv.in_channels, self.conv.out_channels,
                                                                    self.conv.kernel_size, self.conv.stride)
        if self.padding[0] > 0: s += ', padding={}'.format(self.conv.padding)
        if self.dilation[0] > 1: s += ', dilation={}'.format(self.conv.dilation)
        if self.groups > 1: s += ', groups={}'.format(self.conv.groups)
        try: self.cbias = True if not self.conv.bias==False else False
        except RuntimeError: self.cbias = False
        if not self.cbias: s += ', bias={}'.format(self.cbias)
        return s + ', track_running_stats=True)'
        # s += '\neps={}, momentum={}, track_running_stats={})'.format()

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

class mBiSNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(mBiSNConv2d, self).__init__()

        self.dim=2
        self.conv = bidirectional_spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
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

class mBiSNConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(mBiSNConv3d, self).__init__()

        self.dim = 3
        self.conv = bidirectional_spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
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


class mBiSNConv1dmodule(nn.Module):
    def __init__(self, model):
        self.in_channels = int(model.in_channels)
        self.out_channels = int(model.out_channels)
        self.kernel_size = model.kernel_size
        self.stride = model.stride
        self.padding = model.padding
        self.dilation  = model.dilation
        self.groups = int(model.groups)
        try: self.cbias = True if not model.bias==False else False
        except RuntimeError: self.cbias = False
        self.padding_mode = model.padding_mode
        self.dim = 1
        super(mBiSNConv1dmodule, self).__init__()

        self.conv = bidirectional_spectral_norm(nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
                              bias=self.cbias, padding_mode=self.padding_mode))

        self.bias = nn.Parameter(torch.zeros(self.out_channels,1), requires_grad=True)
        self.register_buffer('running_mean', torch.zeros(self.out_channels))
        self.momentum = 0.1

    def __repr__(self):
        s = 'mBiSNConv{}d({}, {}, kernel_size={}, stride={}'.format(self.dim, self.in_channels, self.out_channels, self.kernel_size, self.stride)
        if self.padding[0] > 0: s += ', padding={}'.format(self.padding)
        if self.dilation[0] > 1: s += ', dilation={}'.format(self.dilation)
        if self.groups > 1: s += ', groups={}'.format(self.groups)
        if not self.cbias: s += ', bias={}'.format(self.cbias)
        return s + ', track_running_stats=True)'
        # s += '\neps={}, momentum={}, track_running_stats={})'.format()

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

class mBiSNConv2dmodule(nn.Module):
    def __init__(self, model):
        self.in_channels = int(model.in_channels)
        self.out_channels = int(model.out_channels)
        self.kernel_size = model.kernel_size
        self.stride = model.stride
        self.padding = model.padding
        self.dilation  = model.dilation
        self.groups = int(model.groups)
        try: self.cbias = True if not model.bias==False else False
        except RuntimeError: self.cbias = False
        self.padding_mode = model.padding_mode
        self.dim = 2        
        super(mBiSNConv2dmodule, self).__init__()

        self.conv = bidirectional_spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
                              bias=self.cbias, padding_mode=self.padding_mode))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.momentum = 0.1

    def __repr__(self):
        s = 'mBiSNConv{}d({}, {}, kernel_size={}, stride={}'.format(self.dim, self.in_channels, self.out_channels, self.kernel_size, self.stride)
        if self.padding[0] > 0: s += ', padding={}'.format(self.padding)
        if self.dilation[0] > 1: s += ', dilation={}'.format(self.dilation)
        if self.groups > 1: s += ', groups={}'.format(self.groups)
        if not self.cbias: s += ', bias={}'.format(self.cbias)
        return s + ', track_running_stats=True)'

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

class mBiSNConv3dmodule(nn.Module):
    def __init__(self, model):
        self.in_channels = int(model.in_channels)
        self.out_channels = int(model.out_channels)
        self.kernel_size = model.kernel_size
        self.stride = model.stride
        self.padding = model.padding
        self.dilation  = model.dilation
        self.groups = int(model.groups)
        try: self.cbias = True if not model.bias==False else False
        except RuntimeError: self.cbias = False
        self.padding_mode = model.padding_mode
        self.dim = 1        
        super(mBiSNConv3dmodule, self).__init__()

        self.conv = bidirectional_spectral_norm(nn.Conv3d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
                              bias=self.cbias, padding_mode=self.padding_mode))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.momentum = 0.1

    def __repr__(self):
        s = 'mBiSNConv{}d({}, {}, kernel_size={}, stride={}'.format(self.dim, self.in_channels, self.out_channels, self.kernel_size, self.stride)
        if self.padding[0] > 0: s += ', padding={}'.format(self.padding)
        if self.dilation[0] > 1: s += ', dilation={}'.format(self.dilation)
        if self.groups > 1: s += ', groups={}'.format(self.groups)
        if not self.cbias: s += ', bias={}'.format(self.cbias)
        return s + ', track_running_stats=True)'

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
