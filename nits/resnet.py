import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = [self.kernel_size[0] // 2]  # dynamic add padding based on the kernel_size

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
        self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

    from collections import OrderedDict
class ResNet1DResidualBlock(ResidualBlock):
        conv3 = partial(Conv1dAuto, kernel_size=3, bias=False)

        def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3, *args, **kwargs):
            super().__init__(in_channels, out_channels)
            self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
            self.shortcut = nn.Sequential(OrderedDict(
                {
                    'conv': nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
                                      stride=self.downsampling, bias=False),
                    'bn': nn.BatchNorm1d(self.expanded_channels)

                })) if self.should_apply_shortcut else None

        @property
        def expanded_channels(self):
            return self.out_channels * self.expansion

        @property
        def should_apply_shortcut(self):
            return self.in_channels != self.expanded_channels


class ResNetResidualBlock(ResidualBlock):
        conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
        def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
            super().__init__(in_channels, out_channels)
            self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
            self.shortcut = nn.Sequential(OrderedDict(
                {
                    'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                      stride=self.downsampling, bias=False),
                    'bn': nn.BatchNorm2d(self.expanded_channels)

                })) if self.should_apply_shortcut else None

        @property
        def expanded_channels(self):
            return self.out_channels * self.expansion

        @property
        def should_apply_shortcut(self):
            return self.in_channels != self.expanded_channels

from collections import OrderedDict
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
            return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                              'bn': nn.BatchNorm2d(out_channels)}))


def conv1D_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm1d(out_channels)}))

class ResNet1DBasicBlock(ResNet1DResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv1D_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv1D_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
if __name__ == '__main__':
    dummy = torch.ones((2, 4, 224))

    block = ResNet1DBasicBlock(4, 16)
    print(block(dummy).shape)
    print(block)