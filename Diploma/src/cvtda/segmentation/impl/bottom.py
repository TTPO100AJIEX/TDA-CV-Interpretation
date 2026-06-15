import torch

from .conv import make_conv


class Bottom(torch.nn.Module):
    def __init__(
        self,
        images_example: torch.Tensor,
        features_example: torch.Tensor,
        with_images: bool,
        with_features: bool,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.with_images = with_images
        self.with_features = with_features
        self.in_shape = images_example.shape[1:]

        if with_images and with_features:
            in_channels *= 2
        self.conv = make_conv(in_channels, out_channels)

        flat_shape = torch.nn.Flatten()(images_example).shape[1]
        self.features = torch.nn.Sequential(
            torch.nn.Linear(features_example.shape[1], 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, flat_shape),
            torch.nn.BatchNorm1d(flat_shape),
            torch.nn.GELU(),
        )

    def forward(self, images: torch.Tensor, features: torch.Tensor):
        if not self.with_features and not self.with_images:
            return self.conv(torch.zeros_like(images))

        if self.with_features:
            target_shape = (features.shape[0], *self.in_shape)
            features = self.features(features).reshape(*target_shape)

        data_in = []
        if self.with_images:
            data_in.append(images)
        if self.with_features:
            data_in.append(features)
        return self.conv(torch.cat(data_in, dim=1))
