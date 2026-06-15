import torch

from .downsample import DownsampleBlock
from .upsample import UpsampleBlock
from .bottom import Bottom


class MiniUnetModule(torch.nn.Module):
    def __init__(
        self,
        images_example: torch.Tensor,
        features_example: torch.Tensor,
        with_images: bool = True,
        with_features: bool = True,
        layers: int = 4,
        start_channels: int = 8,
        remove_cross_maps: bool = False,
    ):
        super().__init__()
        self.remove_cross_maps_ = remove_cross_maps
        start_channels = max(images_example.shape[1], start_channels)
        channels = [images_example.shape[1], *[(2**i) * start_channels for i in range(layers + 1)]]

        self.downsample_blocks = torch.nn.ModuleList([DownsampleBlock(*c) for c in zip(channels[:-2], channels[1:-1])])
        for downsample_block in self.downsample_blocks:
            _, images_example = downsample_block(images_example)

        self.bottom = Bottom(images_example, features_example, with_images, with_features, channels[-2], channels[-1])

        channels.reverse()
        self.upsample_blocks = torch.nn.ModuleList([UpsampleBlock(*c) for c in zip(channels[:-2], channels[1:-1])])
        self.output = torch.nn.Sequential(torch.nn.Conv2d(channels[-2], 1, kernel_size=1), torch.nn.Sigmoid())

    def forward(self, images: torch.Tensor, features: torch.Tensor):
        feature_maps = []
        for downsample_block in self.downsample_blocks:
            feature_map, images = downsample_block(images)
            feature_maps.insert(0, feature_map)

        images = self.bottom(images, features)

        for upsample_block, feature_map in zip(self.upsample_blocks, feature_maps):
            if self.remove_cross_maps_:
                feature_map = torch.zeros_like(feature_map)
            images = upsample_block(images, feature_map)
        return self.output(images)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
