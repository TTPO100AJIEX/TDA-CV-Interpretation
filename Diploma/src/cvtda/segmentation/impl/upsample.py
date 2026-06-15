import torch

from .conv import make_conv


class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = make_conv(2 * out_channels, out_channels)

    def forward(self, input: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        features = torch.cat([features, self.up_conv(input)], dim=1)
        return self.conv(features)
