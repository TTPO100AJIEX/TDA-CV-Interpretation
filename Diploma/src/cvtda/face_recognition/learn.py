import os
import typing

import numpy
import torch
import torchvision
import matplotlib.pyplot as plt

import cvtda.logging
import cvtda.dumping
import cvtda.neural_network

from .BaseLearner import BaseLearner
from .DiagramsLearner import DiagramsLearner
from .NNLearner import NNLearner
from .SimpleTopologicalLearner import SimpleTopologicalLearner


def learn(
    # Train
    train_images: numpy.ndarray,
    train_features: numpy.ndarray,
    train_labels: numpy.ndarray,
    train_diagrams: typing.List[numpy.ndarray],
    # Test
    test_images: numpy.ndarray,
    test_features: numpy.ndarray,
    test_labels: numpy.ndarray,
    test_diagrams: typing.List[numpy.ndarray],
    # General
    n_jobs: int = 1,
    random_state: int = 42,
    dump_name: typing.Optional[str] = None,
    # Neural networks
    nn_device: torch.device = cvtda.neural_network.default_device,
    nn_batch_size: int = 32,
    nn_learning_rate: float = 1e-3,
    nn_epochs: int = 25,
    nn_margin: float = 0.1,
    nn_latent_dim: int = 256,
    nn_length_before_new_iter: typing.Optional[int] = None,
    nn_base=torchvision.models.resnet34,
):
    """
    Tries 6 face recognition models on the given dataset.
    (1) Uses topological feature vectors with Euclidean distance as the latent space.
    (2) Uses persistence diagrams with the bottleneck distance as the latent space.
    (3) Neural network with only traditional CNN on raw images.
    (4) Neural network with only topological features.
    (5) Neural network with both topological features and traditional CNN.
    (6) Neural network with only persistence diagrams.

    Parameters
    ----------
    train_images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Images of the training set.
    train_features : ``numpy.ndarray``
        (size `num_items x num_features`) Topological features of the training set.
    train_labels : ``numpy.ndarray``
        (size `num_items`) Target class labels for each object of the training set.
    train_diagrams : ``numpy.ndarray``
        (size `num_items x num_diagrams x num_points x 3`) Persistence diagrams of the training set.

    test_images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Images of the test set.
    test_features : ``numpy.ndarray``
        (size `num_items x num_features`) Topological features of the test set.
    test_labels : ``numpy.ndarray``
        (size `num_items`) Target class labels for each object of the test set.
    test_diagrams : ``numpy.ndarray``, optional
        (size `num_items x num_diagrams x num_points x 3`) Persistence diagrams of the test set.

    n_jobs : ``int``, default: ``-1``
        The number of jobs to use for the computation. See :mod:`joblib` for details.
    random_state : ``int``, default ``42``
        The seed to initialize the pseudo random generator.
    dump_name : ``str``, optional, default ``None``
        The root to dump the results in, if dumping is enabled.

    nn_device : ``torch.device``, default `cvtda.neural_network.default_device`
        A :mod:`torch` device to perform the computations on.
    nn_batch_size : ``int``, default `32`
        Batch size to train the neural networks with.
    nn_learning_rate : ``float``, default `1e-3`
        Learning rate to train the neural networks with.
    nn_epochs : ``int``, default `25`
        Number of epochs to train the neural networks for.
    nn_margin : ``float``, default `0.1`
        Margin for the triplet loss function used to train neural networks.
    nn_latent_dim : ``int``, default `256`
        Dimensionality of the latent space produces by the autoencoder.
    nn_length_before_new_iter : ``int``, optional, default None
        Number of batches sampled at each epoch. Defaults to 20 * batch_size if not specified.
    nn_base : ``torch.nn.Module``, default `torchvision.models.resnet34`
        Constructor of the CNN model used as a baseline for comparison.

    Returns
    -------
    ``matplotlib.pyplot.Figure``
        A :mod:`matplotlib` figure with distributions of distances between
        photos of same person and different people for analysis.
    """

    # Create datasets
    nn_train = cvtda.neural_network.Dataset(
        train_images, train_diagrams, train_features, train_labels, n_jobs=n_jobs, device=nn_device
    )
    nn_test = cvtda.neural_network.Dataset(
        test_images, test_diagrams, test_features, test_labels, n_jobs=n_jobs, device=nn_device
    )

    def classify_one(learner: BaseLearner, name: str, display_name: str, ax: plt.Axes):
        cvtda.logging.logger().print(f"Trying {name} - {learner}")
        learner.fit(nn_train, nn_test)  # Train
        ax.set_title(display_name)
        learner.estimate_quality(nn_test, ax)  # Test

    # Initialize models.
    nn_kwargs = dict(
        n_jobs=n_jobs,
        random_state=random_state,
        device=nn_device,
        batch_size=nn_batch_size,
        learning_rate=nn_learning_rate,
        margin=nn_margin,
        latent_dim=nn_latent_dim,
        length_before_new_iter=nn_length_before_new_iter,
    )
    classifiers = [
        SimpleTopologicalLearner(n_jobs=n_jobs),
        DiagramsLearner(n_jobs=n_jobs),
        NNLearner(
            **nn_kwargs, n_epochs=nn_epochs, skip_diagrams=True, skip_images=False, skip_features=True, base=nn_base
        ),
        NNLearner(
            **nn_kwargs, n_epochs=nn_epochs * 2, skip_diagrams=True, skip_images=True, skip_features=False, base=nn_base
        ),
        NNLearner(
            **nn_kwargs, n_epochs=nn_epochs, skip_diagrams=True, skip_images=False, skip_features=False, base=nn_base
        ),
        NNLearner(
            **nn_kwargs,
            n_epochs=nn_epochs // 2,
            skip_diagrams=False,
            skip_images=True,
            skip_features=True,
            base=nn_base,
        ),
    ]
    names = [
        "SimpleTopologicalLearner",
        "DiagramsLearner",
        "NNLearner_images",
        "NNLearner_features",
        "NNLearner_features_images",
        "NNLearner_diagrams",
    ]
    display_names = [
        "Topological features",
        "Persistence diagrams",
        "Baseline model",
        "FC over topological features",
        "Combined neural network",
        "Trainable vectorization",
    ]

    # Run models
    figure, axes = plt.subplots(2, 3, figsize=(12, 5))
    for args in zip(classifiers, names, display_names, axes.flat):
        classify_one(*args)

    # Improve presentation
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc=(0.35, 0.75))
    figure.tight_layout()

    # Dump results, if needed
    dumper = cvtda.dumping.dumper()
    if (dump_name is not None) and isinstance(dumper, cvtda.dumping.NumpyDumper):
        file = dumper.get_file_name_(cvtda.dumping.dump_name_concat(dump_name, "distributions"))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        figure.savefig(file[:-4] + ".svg")
        figure.savefig(file[:-4] + ".png")
    return figure
