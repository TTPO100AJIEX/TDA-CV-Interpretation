import torch


def make_conv(in_channels: int, out_channels: int, convolutions: int = 2):
    model = [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    ]
    for _ in range(convolutions - 1):
        model.extend(
            [
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
            ]
        )
    return torch.nn.Sequential(*model)
