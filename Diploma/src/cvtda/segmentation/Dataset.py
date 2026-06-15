import torch


class Dataset:
    """
    A single structure representing the dataset for segmentation with topological features.

    Attributes
    ----------
    images : ``list[torch.Tensor]``
        (size `num_items x num_channels x width x height`) Sets of images.
    labels : ``torch.Tensor``
        (size `num_items x width x height`) Target segmentation masks.
    features : ``torch.Tensor``
        (size `num_items x num_features`) Topological features for each image.
    """

    def __init__(self, images, features, masks):
        # Save images in a torch-compatible format.
        self.images = torch.tensor(images, dtype=torch.float32)
        if len(self.images.shape) == 4:
            if self.images.shape[-1] == 3:
                self.images = self.images.permute((0, 3, 1, 2))
        else:
            assert len(self.images.shape) == 3
            self.images = self.images.unsqueeze(1)

        # Save everything else.
        self.masks = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)
        self.features = torch.tensor(features, dtype=torch.float32)
