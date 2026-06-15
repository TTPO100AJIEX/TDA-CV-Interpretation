import typing

import torch
import torchvision
import torch.utils.data
import pytorch_metric_learning.miners
import pytorch_metric_learning.losses
import pytorch_metric_learning.samplers
import pytorch_metric_learning.utils.accuracy_calculator

import cvtda.utils
import cvtda.logging
import cvtda.neural_network

from .BaseLearner import BaseLearner


class NNLearner(BaseLearner):
    """
    A neural network-based face recognition model trained for the triplet loss [1].

    References
    ----------
    .. [1] Hideki Oki, Motoshi Abe, Junichi Miyao, Takio Kurita
            "Triplet Loss for Knowledge Distillation"
            https://doi.org/10.48550/arXiv.2004.08116
    """

    def __init__(
        self,
        n_jobs: int = -1,
        random_state: int = 42,
        device: torch.device = torch.device("cuda"),
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        n_epochs: int = 25,
        length_before_new_iter: typing.Optional[int] = None,
        margin: int = 0.1,
        latent_dim: int = 256,
        skip_diagrams: bool = False,
        skip_images: bool = False,
        skip_features: bool = False,
        base=torchvision.models.resnet34,
    ):
        super().__init__(n_jobs)
        self.random_state_ = random_state

        self.device_ = device
        self.batch_size_ = batch_size
        self.learning_rate_ = learning_rate
        self.n_epochs_ = n_epochs
        self.length_before_new_iter_ = length_before_new_iter or (self.batch_size_ * 20)

        self.margin_ = margin
        self.latent_dim_ = latent_dim

        self.skip_diagrams_ = skip_diagrams
        self.skip_images_ = skip_images
        self.skip_features_ = skip_features
        self.base_ = base

    def fit(self, train: cvtda.neural_network.Dataset, val: typing.Optional[cvtda.neural_network.Dataset]):
        # Set seed
        cvtda.utils.set_random_seed(self.random_state_)

        # Initialize sampler and dataloader
        train_mpc_sampler = pytorch_metric_learning.samplers.MPerClassSampler(
            m=4, labels=train.labels, length_before_new_iter=self.length_before_new_iter_
        )
        train_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train.images, train.features, train.labels, torch.arange(len(train))),
            batch_size=self.batch_size_,
            sampler=train_mpc_sampler,
        )

        # Initialize model
        self.init_(*next(iter(train_dl)), train)

        # Initialize triplet loss and relevant utilities
        train_miner = pytorch_metric_learning.miners.TripletMarginMiner(margin=self.margin_, type_of_triplets="all")
        train_loss = pytorch_metric_learning.losses.TripletMarginLoss(margin=self.margin_)
        metrics = pytorch_metric_learning.utils.accuracy_calculator.AccuracyCalculator()

        # Initialize validation dataloader
        if val is not None:
            val_dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(val.images, val.features, val.labels, torch.arange(len(val))),
                batch_size=self.batch_size_,
            )

        pbar = cvtda.logging.logger().pbar(range(self.n_epochs_), desc="Train")
        for _ in pbar:
            sum_loss = 0

            self.model_list_.train()
            for images, features, labels, diagram_idxs in train_dl:
                self.optimizer_.zero_grad()
                embeddings = self.forward_(images, features, diagram_idxs, train)  # Forward pass
                indices = train_miner(embeddings, labels)
                loss = train_loss(embeddings, labels, indices)  # Loss
                loss.backward()
                self.optimizer_.step()  # Optimization step
                sum_loss += loss.item()
            postfix = {"loss": sum_loss}
            self.scheduler_.step()

            # Print metrics on validation, if needed
            if val is not None:
                self.model_list_.eval()
                all_embeddings, all_targets = [], []
                for images, features, labels, diagram_idxs in val_dl:
                    with torch.no_grad():
                        all_targets.append(labels)
                        all_embeddings.append(self.forward_(images, features, diagram_idxs, val))
                result = metrics.get_accuracy(torch.cat(all_embeddings, dim=0), torch.cat(all_targets, dim=0))
                postfix = {**postfix, **result}

            cvtda.logging.logger().set_pbar_postfix(pbar, postfix)

        self.model_list_.eval()
        self.fitted_ = True
        return self

    def calculate_distance_(self, first: int, second: int, dataset: cvtda.neural_network.Dataset):
        self.model_list_.eval()
        idx = [first, second]
        embeddings = self.forward_(dataset.images[idx], dataset.features[idx], idx, dataset)
        return torch.sqrt(torch.sum((embeddings[0] - embeddings[1]) ** 2)).item()

    def init_(
        self,
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
            images_output = len(torch.unique(labels))
        else:
            images_output = self.latent_dim_

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
                torch.nn.Dropout(0.3),
                torch.nn.LazyLinear(1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(1024, 768),
                torch.nn.BatchNorm1d(768),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(768, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.GELU(),
                torch.nn.Linear(512, self.latent_dim_),
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
