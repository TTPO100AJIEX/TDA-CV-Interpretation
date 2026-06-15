import typing

import numpy
import torch
import torch.utils.data
import torchph.nn.slayer

import cvtda.utils
import cvtda.logging

from .device import default_device


def transform(diagram: torch.Tensor, dim: int):
    """
    Extracts a subdiagram corresponding to one homology dimnesion from the given persistence diagram
    and performs a rotation of the persistence diagrams into birth-persistence coordinates.
    See :class:`torchph.nn.slayer.LogStretchedBirthLifeTimeCoordinateTransform` for details.

    Parameters
    ----------
    diagram : ``torch.Tensor``
        (size `num_points x 3`) A persistence diagram.
    dim : ``int``
        Homology dimension to extract.

    Returns
    -------
    ``torch.Tensor``
        (size `num_points x 3`) Resulting subdiagram.
    """
    dim_filter = diagram[:, 2] == dim
    non_degenerate_filter = diagram[:, 0] < diagram[:, 1]
    rotation = torchph.nn.slayer.LogStretchedBirthLifeTimeCoordinateTransform(0.01)
    return rotation(diagram[dim_filter & non_degenerate_filter][:, 0:2])


def process_diagram(diags: torch.Tensor):
    """
    Preprocess a set of diagrams for convenient use with :mod:`torchph`.
    - Partitions the diagrams by homology dimensions.
    - Performs a rotation into birth-persistence coordinates. See :func:`transform` for details.
    - Prepares the diagrams for use in :mod:`torchph`. See :func:`torchph.nn.slayer.prepare_batch` for details.

    Parameters
    ----------
    diags : ``torch.Tensor``
        (size `num_diagrams x num_points x 3`) A batch of persistence diagrams.

    Returns
    -------
    ``tuple[torch.Tensor, torch.Tensor]``
        1. (size `num_diagrams x num_points x 3`) Preprocessed diagrams.
        2. (size `num_diagrams x num_points x 3`) Mask for non-dummy points on persistence diagrams.
        See :func:`torchph.nn.slayer.prepare_batch` for details.
    """
    diagrams, non_dummy_points = [], []
    for dim in diags[:, :, 2].unique(sorted=True):
        diags_dim = [transform(diag, dim) for diag in diags]
        processed = torchph.nn.slayer.prepare_batch(diags_dim)
        diagrams.append(processed[0].cpu())
        non_dummy_points.append(processed[1].cpu())
    return diagrams, non_dummy_points


class Dataset(torch.utils.data.Dataset):
    """
    A single structure representing the dataset with topological information.

    Attributes
    ----------
    images : ``torch.Tensor``
        (size `num_items x num_channels x width x height`) Raw images.
    labels : ``torch.Tensor``
        (size `num_items x ...`) The target variable.
        May be of any shape, if the first dimension is the number of items.
    features : ``torch.Tensor``
        (size `num_items x num_features`) Topological features for each image.
    raw_diagrams : ``list[numpy.ndarray]``, optional
        (size `n_items x n_diagrams x n_points x 3`) Raw persistence diagrams, if provided.
    diagrams : ``list[torch.Tensor]``, optional
        (size `n_items x n_diagrams x n_points x 3`) Processed persistence diagrams as single tensors.
    non_dummy_points : ``list[torch.Tensor]``, optional
        (size `n_items x n_diagrams x n_points`) Mask for non-dummy points on persistence diagrams.
        See :func:`transform` for details.
    """

    def __init__(
        self,
        images: numpy.ndarray,
        diagrams: typing.Optional[typing.List[numpy.ndarray]],  # n_items x n_diagrams x n_points x 3
        features: numpy.ndarray,
        labels: typing.Optional[numpy.ndarray],
        n_jobs: int = -1,
        device: torch.device = default_device,
    ):
        self.n_jobs_ = n_jobs
        self.device_ = device

        # Save images in a torch-compatible format.
        if images is not None:
            self.images = torch.tensor(images, dtype=torch.float32)
            if len(self.images.shape) == 4:
                self.images = self.images.permute((0, 3, 1, 2))
            else:
                assert len(self.images.shape) == 3
                self.images = self.images.unsqueeze(1)

        # Save everything else.
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.raw_diagrams = diagrams

        if diagrams is None:
            return

        # Reorder the diagrams
        diagrams = [
            torch.tensor(numpy.array([item[num_diagram] for item in diagrams]), dtype=torch.float32)
            for num_diagram in range(len(diagrams[0]))
        ]

        # Process diagrams.
        pbar = cvtda.logging.logger().pbar(diagrams, desc="Dataset: processing diagrams")
        diagrams = cvtda.utils.parallel(process_diagram, pbar, n_jobs=self.n_jobs_)

        # Store
        self.diagrams, self.non_dummy_points = [], []
        for diag, ndp in diagrams:
            self.diagrams.extend(diag)
            self.non_dummy_points.extend(ndp)

        cvtda.logging.logger().print(
            f"Constructed a dataset of {len(self.images)} images of shape {self.images[0].shape} "
            + f"with {len(self.diagrams)} diagrams and {self.features.shape[1]} features"
        )

    def __len__(self):
        return len(self.images)

    def get_labels(self, idxs):
        """
        Gets the target variable on the required torch device for a batch of elements.

        Parameters
        ----------
        idxs : ``Iterable[int]``
            (size `batch_size`) The indexes of objects in the batch.

        Returns
        -------
        ``torch.Tensor``
            (size `batch_size x ...`) The target variable for the requested objects.
        """
        return self.labels[idxs].to(self.device_)

    def get_diagrams(self, idxs):
        """
        Gets the processed persistence diagrams on the required torch device for a batch of elements.

        Parameters
        ----------
        idxs : ``Iterable[int]``
            (size `batch_size`) The indexes of objects in the batch.

        Returns
        -------
        ``tuple[list[torch.Tensor], list[torch.Tensor]]``
            1. (size `batch_size x num_diagrams x num_points x 3`) Preprocessed diagrams.
            2. (size `batch_size x num_diagrams x num_points x 3`) Mask for non-dummy points on persistence diagrams.
        """
        output = []
        for diag, ndp in zip(self.diagrams, self.non_dummy_points):
            output.append(diag[idxs].to(self.device_))
            output.append(ndp[idxs].to(self.device_))
        return output
