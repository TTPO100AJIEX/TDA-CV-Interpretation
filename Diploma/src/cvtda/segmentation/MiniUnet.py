import typing

import numpy
import torch
import torch.utils.data
import segmentation_models_pytorch

import cvtda.utils
import cvtda.logging

from .impl import MiniUnetModule
from .estimate_quality import estimate_quality
from .Dataset import Dataset


class MiniUnet:
    """
    A miniature UNet network following [1].

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"
            https://doi.org/10.48550/arXiv.1505.04597
    """

    def __init__(
        self,
        random_state: int = 42,
        device: torch.device = torch.device("cuda"),
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        remove_cross_maps: bool = False,
        with_images: bool = False,
        with_features: bool = False,
    ):
        self.random_state_ = random_state

        self.device_ = device
        self.batch_size_ = batch_size
        self.learning_rate_ = learning_rate
        self.n_epochs_ = n_epochs
        self.remove_cross_maps_ = remove_cross_maps

        self.with_images_ = with_images
        self.with_features_ = with_features

    def fit(self, train: Dataset, val: typing.Optional[Dataset]):
        """
        Trains the model on the given dataset.

        Parameters
        ----------
        train : ``Dataset``
            Training dataset.
        val : ``Dataset``, optional
            Validation dataset. If specified, the quality metrics on this dataset will be printed after every epoch.
        """

        # Set seed
        cvtda.utils.set_random_seed(self.random_state_)

        # Initialize dataloader
        train_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train.images, train.features, train.masks),
            batch_size=self.batch_size_,
            shuffle=True,
        )

        # Initialize model
        self.init_(*next(iter(train_dl)))

        pbar = cvtda.logging.logger().pbar(range(self.n_epochs_))
        for epoch in pbar:
            sum_loss = 0

            self.model_.train()
            for images, features, masks in train_dl:
                self.optimizer_.zero_grad()
                pred = self.forward_(images, features)  # Forward pass
                loss = self.loss_(pred, masks.to(self.device_))  # Loss
                loss.backward()
                self.optimizer_.step()  # Optimization step
                sum_loss += loss.item()
            postfix = {"loss": sum_loss, "lr": self.optimizer_.param_groups[0]["lr"]}
            self.scheduler_.step()

            # Print metrics on validation, if needed
            if val is not None:
                val_proba = self.predict_proba(val)
                postfix = {**postfix, **estimate_quality(val_proba, val.masks.cpu().numpy())}
            cvtda.logging.logger().set_pbar_postfix(pbar, postfix)

        self.fitted_ = True
        return self

    def predict_proba(self, dataset: Dataset) -> numpy.ndarray:
        """
        Calculate predictions for all images in the dataset.

        Parameters
        ----------
        dataset : ``cvtda.neural_network.Dataset``
            Dataset.

        Returns
        -------
        ``torch.Tensor``
            (size `num_items x width x height`) Predicted probability estimates for each pixel.
        """
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dataset.images, dataset.features), batch_size=self.batch_size_, shuffle=False
        )

        y_pred = []
        self.model_.eval()
        with torch.no_grad():
            for images, features in dl:
                y_pred.append(self.forward_(images, features))
        return torch.vstack(y_pred).cpu().numpy()

    def init_(self, images: torch.Tensor, features: torch.Tensor, masks: torch.Tensor):
        # Create model
        self.model_ = MiniUnetModule(
            images,
            features,
            with_images=self.with_images_,
            with_features=self.with_features_,
            remove_cross_maps=self.remove_cross_maps_,
        ).to(self.device_)

        # Create optimizer and scheduler
        self.optimizer_ = torch.optim.AdamW(params=self.model_.parameters(), lr=self.learning_rate_)

        def lr_scheduler_lambda(epoch):
            if epoch < self.n_epochs_ // 10:
                return 1
            if epoch < self.n_epochs_ // 4:
                return 0.1
            if epoch < self.n_epochs_ // 2:
                return 0.01
            if epoch < 3 * self.n_epochs_ // 4:
                return 0.001
            return 0.0001

        self.scheduler_ = torch.optim.lr_scheduler.LambdaLR(self.optimizer_, lr_scheduler_lambda)

        # Ensure everything is initialized correctly
        assert self.forward_(images, features).shape == masks.shape
        cvtda.logging.logger().print(f"Parameters: {sum(p.numel() for p in self.model_.parameters())}")

    def forward_(self, images: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.model_(images.to(self.device_), features.to(self.device_))

    def loss_(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy(input, target, reduction="mean")
        iou = segmentation_models_pytorch.losses.JaccardLoss("binary", from_logits=False)(input, target)
        return bce + iou
