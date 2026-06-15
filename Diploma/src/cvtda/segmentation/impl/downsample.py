import typing

import torch

from .conv import make_conv


class DownsampleBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = make_conv(in_channels, out_channels)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, input: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(input)
        output = self.pool(features)
        return features, output
