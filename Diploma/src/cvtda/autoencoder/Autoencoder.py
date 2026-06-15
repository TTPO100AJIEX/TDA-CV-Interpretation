import typing

import numpy
import torch
import torchvision
import torch.utils.data

import cvtda.utils
import cvtda.logging
import cvtda.neural_network

from .estimate_quality import estimate_quality


class Autoencoder:
    """
    A simple opinionated autoencoder implementation following [1].

    References
    ----------
    .. [1] Mark A. Kramer "Nonlinear principal component analysis using autoassociative neural networks"
            https://doi.org/10.1002/aic.690370209
    """

    def __init__(
        self,
        random_state: int = 42,
        device: torch.device = torch.device("cuda"),
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        n_epochs: int = 20,
        latent_dim: int = 256,
        skip_diagrams: bool = False,
        skip_images: bool = False,
        skip_features: bool = False,
        base=torchvision.models.resnet34,
    ):
        self.random_state_ = random_state

        self.device_ = device
        self.batch_size_ = batch_size
        self.learning_rate_ = learning_rate
        self.n_epochs_ = n_epochs
        self.latent_dim_ = latent_dim

        self.skip_diagrams_ = skip_diagrams
        self.skip_images_ = skip_images
        self.skip_features_ = skip_features
        self.base_ = base

    def fit(self, train: cvtda.neural_network.Dataset, val: typing.Optional[cvtda.neural_network.Dataset]):
        """
        Trains the model on the given dataset.

        Parameters
        ----------
        train : ``cvtda.neural_network.Dataset``
            Training dataset.
        val : ``cvtda.neural_network.Dataset``, optional
            Validation dataset. If specified, the quality metrics on this dataset will be printed after every epoch.
        """

        # Set seed
        cvtda.utils.set_random_seed(self.random_state_)

        # Initialize dataloader
        train_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train.images, train.features, torch.arange(len(train))),
            batch_size=self.batch_size_,
            shuffle=True,
        )

        # Initialize model
        self.init_(*next(iter(train_dl)), train)

        for epoch in range(self.n_epochs_):
            sum_loss = 0

            self.model_list_.train()
            for images, features, diagram_idxs in cvtda.logging.logger().pbar(train_dl, desc=f"Epoch {epoch}"):
                self.optimizer_.zero_grad()
                pred = self.decode_(self.encode_(images, features, diagram_idxs, train))  # Forward pass
                loss = torch.nn.functional.mse_loss(pred, images.to(self.device_), reduction="mean")  # Loss
                loss.backward()
                self.optimizer_.step()  # Optimization step
                sum_loss += loss.item()
            postfix = {"loss": sum_loss, "lr": self.optimizer_.param_groups[0]["lr"]}
            self.scheduler_.step()

            # Print metrics on validation, if needed
            if val is not None:
                decoded = self.decode(self.encode(val))
                postfix = {**postfix, **estimate_quality(decoded, val.images.permute((0, 2, 3, 1)).cpu().numpy())}

            cvtda.logging.logger().print(f"Epoch {epoch}:", postfix)

        self.fitted_ = True
        return self

    def encode(self, dataset: cvtda.neural_network.Dataset) -> numpy.ndarray:
        """
        Runs the encoder on a dataset.

        Parameters
        ----------
        dataset : ``cvtda.neural_network.Dataset``
            Dataset.

        Returns
        -------
        ``torch.Tensor``
            (size `num_items x latent_dim`) Encoded images in the latent space.
        """

        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dataset.images, dataset.features, torch.arange(len(dataset))),
            batch_size=self.batch_size_,
            shuffle=False,
        )

        encoded = []
        self.model_list_.eval()
        with torch.no_grad():
            for images, features, diagram_idxs in dl:
                encoded.append(self.encode_(images, features, diagram_idxs, dataset))
        return torch.vstack(encoded).cpu().numpy()

    def decode(self, latent_vectors: numpy.ndarray) -> numpy.ndarray:
        """
        Runs the decoder on a dataset.

        Parameters
        ----------
        latent_vectors : ``torch.Tensor``
            Encoded images in the latent space.

        Returns
        -------
        ``torch.Tensor``
            (size `num_items x num_channels x width x height`) Decoded images.
        """

        dl = torch.utils.data.DataLoader(latent_vectors, batch_size=self.batch_size_, shuffle=False)

        decoded = []
        self.model_list_.eval()
        with torch.no_grad():
            for vecs in dl:
                decoded.append(self.decode_(vecs))
        return torch.vstack(decoded).permute((0, 2, 3, 1)).squeeze().cpu().numpy()

    def init_(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        diagram_idxs: torch.Tensor,
        dataset: cvtda.neural_network.Dataset,
    ):
        # Create encoder
        diagrams = [] if self.skip_diagrams_ else dataset.get_diagrams(diagram_idxs)
        self.encoder_base_ = (
            cvtda.neural_network.NNBase(
                num_diagrams=len(diagrams) // 2,
                skip_diagrams=self.skip_diagrams_,
                skip_images=self.skip_images_,
                skip_features=self.skip_features_,
                images_n_channels=images.shape[1],
                base=self.base_,
            )
            .to(self.device_)
            .train()
        )
        self.encoder_ = torch.nn.LazyLinear(self.latent_dim_).to(self.device_).train()

        # Create decoder
        w = images.shape[2] // 4
        h = images.shape[3] // 4
        self.decoder_ = (
            torch.nn.Sequential(
                torch.nn.InstanceNorm1d(self.latent_dim_),
                torch.nn.Linear(self.latent_dim_, 128 * w * h),
                torch.nn.InstanceNorm1d(128 * w * h),
                torch.nn.Unflatten(1, (128, w, h)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1),
                torch.nn.InstanceNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.InstanceNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1),
                torch.nn.InstanceNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
                torch.nn.InstanceNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, images.shape[1], kernel_size=3, padding=1),
                torch.nn.Sigmoid(),
            )
            .to(self.device_)
            .train()
        )

        self.model_list_ = torch.nn.ModuleList([self.encoder_base_, self.encoder_, self.decoder_])

        # Create optimizer and scheduler
        self.optimizer_ = torch.optim.AdamW(params=self.model_list_.parameters(), lr=self.learning_rate_)

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
        encoded = self.encode_(images, features, diagram_idxs, dataset)
        assert encoded.shape == (images.shape[0], self.latent_dim_), encoded.shape

        decoded = self.decode(encoded)
        assert decoded.shape == images.permute((0, 2, 3, 1)).squeeze().shape, f"{decoded.shape} != {images.shape}"

        cvtda.logging.logger().print(
            f"Encoder Base parameters: {sum(p.numel() for p in self.encoder_base_.parameters())}"
        )
        cvtda.logging.logger().print(f"Decoder parameters: {sum(p.numel() for p in self.decoder_.parameters())}")

    def encode_(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        diagram_idxs: torch.Tensor,
        dataset: cvtda.neural_network.Dataset,
    ) -> torch.Tensor:
        images = images.to(self.device_)
        features = features.to(self.device_)
        if self.skip_diagrams_:
            return self.encoder_(self.encoder_base_(images, features))
        return self.encoder_(self.encoder_base_(images, features, *dataset.get_diagrams(diagram_idxs)))

    def decode_(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder_(latent_vector.to(self.device_))
