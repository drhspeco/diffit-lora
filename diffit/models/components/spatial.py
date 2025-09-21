"""
Spatial Operations for DiffiT

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import torch.nn as nn


class Tokenizer(nn.Module):
    """
    Convert image patches to tokens
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, out_channels: int, in_channels=3):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv3x3(x)


class Head(nn.Module):
    """
    Output head to convert tokens back to image
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, in_channels: int, out_channels=3):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=in_channels // 4, num_channels=in_channels)
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv3x3(self.group_norm(x))


class Downsample(nn.Module):
    """
    Downsampling layer
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1, output_padding: int = 1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, output_padding=output_padding)

    def forward(self, x):
        return self.conv(x)
