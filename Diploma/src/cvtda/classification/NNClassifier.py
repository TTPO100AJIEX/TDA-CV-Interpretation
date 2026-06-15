import typing

import numpy
import torch
import torchvision
import sklearn.base
import sklearn.metrics
import torch.utils.data

import cvtda.utils
import cvtda.logging
import cvtda.neural_network


class NNClassifier(sklearn.base.ClassifierMixin):
    """
    A simple opinionated classifier based on neural networks.
    """

    def __init__(
        self,
        random_state: int = 42,
        device: torch.device = torch.device("cuda"),
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        n_epochs: int = 20,
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
            torch.utils.data.TensorDataset(train.images, train.features, train.labels, torch.arange(len(train))),
            batch_size=self.batch_size_,
            shuffle=True,
        )

        # Initialize model
        self.init_(len(torch.unique(train.labels)), *next(iter(train_dl)), train)

        for epoch in range(self.n_epochs_):
            sum_loss = 0

            self.model_list_.train()
            for images, features, labels, diagram_idxs in cvtda.logging.logger().pbar(train_dl, desc=f"Epoch {epoch}"):
                self.optimizer_.zero_grad()
                pred = self.forward_(images, features, diagram_idxs, train)  # Forward pass
                loss = torch.nn.functional.cross_entropy(pred, labels.to(self.device_), reduction="mean")  # Loss
                loss.backward()
                self.optimizer_.step()  # Optimization step
                sum_loss += loss.item()
            postfix = {"loss": sum_loss, "lr": self.optimizer_.param_groups[0]["lr"]}
            self.scheduler_.step()

            # Print metrics on validation, if needed
            if val is not None:
                val_proba = self.predict_proba_(val)
                val_pred = numpy.argmax(val_proba, axis=1)
                postfix["val_acc"] = sklearn.metrics.accuracy_score(val.labels, val_pred)

            cvtda.logging.logger().print(f"Epoch {epoch}:", postfix)

        self.fitted_ = True
        return self

    def predict_proba(self, dataset: cvtda.neural_network.Dataset) -> numpy.ndarray:
        """
        Calculate predictions for all images in the dataset.

        Parameters
        ----------
        dataset : ``cvtda.neural_network.Dataset``
            Dataset.

        Returns
        -------
        ``torch.Tensor``
            (size `num_items x num_classes`) Probabilities of each object belonging to each class.
        """
        assert self.fitted_ is True, "fit() must be called before predict_proba()"
        cvtda.utils.set_random_seed(self.random_state_)
        return self.predict_proba_(dataset)

    def init_(
        self,
        num_classes: int,
        images: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        diagram_idxs: torch.Tensor,
        dataset: cvtda.neural_network.Dataset,
    ):
        diagrams = [] if self.skip_diagrams_ else dataset.get_diagrams(diagram_idxs)

        is_no_topology = self.skip_diagrams_ and self.skip_features_ and not self.skip_images_
        cvtda.logging.logger().print("Topology: ", not is_no_topology)
        cvtda.logging.logger().print("Images: ", images.shape)
        cvtda.logging.logger().print("Features: ", features.shape)

        if is_no_topology:
            images_output = num_classes
        else:
            images_output = 1024

        # Create model
        self.nn_base_ = (
            cvtda.neural_network.NNBase(
                num_diagrams=len(diagrams) // 2,
                skip_diagrams=self.skip_diagrams_,
                skip_images=self.skip_images_,
                skip_features=self.skip_features_,
                images_n_channels=images.shape[1],
                images_output=images_output,
                base=self.base_,
            )
            .to(self.device_)
            .train()
        )

        self.model_ = (
            torch.nn.Sequential(
                torch.nn.Dropout(0.4),
                torch.nn.LazyLinear(256),
                torch.nn.BatchNorm1d(256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.GELU(),
                torch.nn.Linear(32, num_classes),
            )
            .to(self.device_)
            .train()
        )

        if is_no_topology:
            self.model_ = torch.nn.Identity()

        self.model_list_ = torch.nn.ModuleList([self.nn_base_, self.model_])

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
        self.forward_(images, features, diagram_idxs, dataset)
        if not is_no_topology:
            cvtda.logging.logger().print(f"Input to LazyLinear: {self.model_[1].in_features}")
        cvtda.logging.logger().print(f"Parameters: {sum(p.numel() for p in self.model_list_.parameters())}")

    def forward_(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        diagram_idxs: torch.Tensor,
        dataset: cvtda.neural_network.Dataset,
    ) -> torch.Tensor:
        images = images.to(self.device_)
        features = features.to(self.device_)
        if self.skip_diagrams_:
            return self.model_(self.nn_base_(images, features))
        return self.model_(self.nn_base_(images, features, *dataset.get_diagrams(diagram_idxs)))

    def predict_proba_(self, dataset: cvtda.neural_network.Dataset) -> numpy.ndarray:
        # Initialize dataloader
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dataset.images, dataset.features, torch.arange(len(dataset))),
            batch_size=self.batch_size_,
            shuffle=False,
        )

        # Make predictions
        y_pred_proba = []
        self.model_list_.eval()
        with torch.no_grad():
            for images, features, diagram_idxs in dl:
                y_pred_proba.append(self.forward_(images, features, diagram_idxs, dataset))
        y_pred_proba = torch.vstack(y_pred_proba)
        # Apply softmax to make probabilities from logits
        return torch.nn.functional.softmax(y_pred_proba, dim=1).cpu().numpy()
