import torch
import torchph.nn.slayer
import torchvision.models


class Slayer(torch.nn.Module):
    """
    Wrapper around :class:`torchph.nn.slayer.SLayerExponential` to alleviate
    a bug in the library that breaks if a batch contains only one element.
    """

    def __init__(self, n_elements: int):
        super().__init__()
        self.slayer = torchph.nn.slayer.SLayerExponential(n_elements)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.slayer(input)
        # There as a squeeze() at the end of SLayerExponential.forward()
        # that breaks everything if n_elements=1 or batch_size=1
        if len(output.shape) != 1:
            return output
        if self.slayer.n_elements == 1:
            return output.unsqueeze(1)
        return output.unsqueeze(0)


class NNBase(torch.nn.Module):
    """
    Base module for neural networks used in cvtda. Combines three branches:
    - Baseline convolutional neural network. ResNet34 is used a default.
    - FC layer over topological features.
    - Processing persistence diagrams directly. See :class:`Slayer` for details.
    """

    def __init__(
        self,
        num_diagrams: int,
        skip_diagrams: bool = False,
        features_per_diagram: int = 8,
        skip_images: bool = False,
        images_n_channels: int = 3,
        images_output: int = 512,
        skip_features: bool = False,
        features_output: int = 512,
        base=torchvision.models.resnet34,
    ):
        super().__init__()

        # Create torchph slayes
        def make_slayer():
            n = features_per_diagram
            return torch.nn.Sequential(
                Slayer(n),
                torch.nn.Linear(n, n),
                torch.nn.BatchNorm1d(n),
                torch.nn.GELU(),
            )

        self.slayers_ = None if skip_diagrams else torch.nn.ModuleList([make_slayer() for _ in range(num_diagrams)])
        self.slayers_compressor_ = torch.nn.Linear(num_diagrams * features_per_diagram, features_output)

        # Create baseline CNN
        self.images_ = None
        if not skip_images:
            self.images_ = base(num_classes=images_output)
            self.images_.conv1 = torch.nn.Conv2d(images_n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Create FC for topological features.
        self.features_ = None if skip_features else torch.nn.LazyLinear(features_output)

    def forward(self, images, features, *diagrams):
        result = []

        # CNN
        if self.images_ is not None:
            result.append(self.images_(images))

        # Diagrams
        if self.slayers_ is not None:
            result.append(
                self.slayers_compressor_(
                    torch.hstack(
                        [
                            self.slayers_[i](
                                (diagrams[2 * i], diagrams[2 * i + 1], diagrams[2 * i].shape[1], len(images))
                            )
                            for i in range(len(self.slayers_))
                        ]
                    )
                )
            )

        # Topological features
        if self.features_ is not None:
            result.append(self.features_(features))

        # Concatentate everything and return
        return torch.cat(result, dim=1)
