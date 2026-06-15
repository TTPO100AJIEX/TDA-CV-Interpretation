import typing

import numpy
import torch
import pandas
import torchvision

import cvtda.logging
import cvtda.dumping
import cvtda.neural_network

from .Autoencoder import Autoencoder
from .estimate_quality import estimate_quality


def try_autoencoders(
    # Train
    train_images: numpy.ndarray,
    train_features: numpy.ndarray,
    train_diagrams: typing.Optional[typing.List[numpy.ndarray]],
    # Test
    test_images: numpy.ndarray,
    test_features: numpy.ndarray,
    test_diagrams: typing.Optional[typing.List[numpy.ndarray]],
    # General
    n_jobs: int = -1,
    random_state: int = 42,
    dump_name: typing.Optional[str] = None,
    only_get_from_dump: bool = False,
    # Neural networks
    nn_device: torch.device = cvtda.neural_network.default_device,
    nn_batch_size: int = 128,
    nn_learning_rate: float = 1e-3,
    nn_epochs: int = 20,
    nn_latent_dim: int = 256,
    nn_base=torchvision.models.resnet34,
):
    """
    Tries 4 autoencoder models solving compression on the given dataset.
    (1) Using only topological features.
    (2) Using only traditional CNN on raw images.
    (3) Using both topological features and traditional CNN.
    (4) Using only persistence diagrams.

    Parameters
    ----------
    train_images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Images of the training set.
    train_features : ``numpy.ndarray``
        (size `num_items x num_features`) Topological features of the training set.
    train_diagrams : ``numpy.ndarray``, optional
        (size `num_items x num_diagrams x num_points x 3`) Persistence diagrams of the training set.
        If not provided, model (4) is excluded from the analysis.

    test_images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Images of the test set.
    test_features : ``numpy.ndarray``
        (size `num_items x num_features`) Topological features of the test set.
    test_diagrams : ``numpy.ndarray``, optional
        (size `num_items x num_diagrams x num_points x 3`) Persistence diagrams of the test set.
        If not provided, model (4) is excluded from the analysis.

    n_jobs : ``int``, default: ``-1``
        The number of jobs to use for the computation. See :mod:`joblib` for details.
    random_state : ``int``, default ``42``
        The seed to initialize the pseudo random generator.
    dump_name : ``str``, optional, default ``None``
        The root to dump the results in, if dumping is enabled.
    only_get_from_dump : ``bool``, default `False`
        If true, all results will be obtained from dump, and no computations will be performed.

    nn_device : ``torch.device``, default `cvtda.neural_network.default_device`
        A :mod:`torch` device to perform the computations on.
    nn_batch_size : ``int``, default `128`
        Batch size to train the neural networks with.
    nn_learning_rate : ``float``, default `1e-3`
        Learning rate to train the neural networks with.
    nn_epochs : ``int``, default `20`
        Number of epochs to train the neural networks for.
    nn_latent_dim : ``int``, default `256`
        Dimensionality of the latent space produces by the autoencoder.
    nn_base : ``torch.nn.Module``, default `torchvision.models.resnet34`
        Constructor of the CNN model used as a baseline for comparison.

    Returns
    -------
    ``pandas.DataFrame``
        A data frame with quality metrics of each model. See :func:`estimate_quality` for the list of metrics.
    """

    without_diagrams = (train_diagrams is None) and (test_diagrams is None)

    # Create datasets
    if (train_images is not None) and (not only_get_from_dump):
        nn_train = cvtda.neural_network.Dataset(
            train_images, train_diagrams, train_features, None, n_jobs=n_jobs, device=nn_device
        )
        nn_test = cvtda.neural_network.Dataset(
            test_images, test_diagrams, test_features, None, n_jobs=n_jobs, device=nn_device
        )

    def try_one(model: Autoencoder, name: str, display_name: str):
        if without_diagrams and name == "diagrams":
            cvtda.logging.logger().print(f"Skipping {name} - {model}")
            return {}

        cvtda.logging.logger().print(f"Trying {name} - {model}")

        dumper = cvtda.dumping.dumper()
        encoded_dump_name = cvtda.dumping.dump_name_concat(dump_name, f"{name}_encoded")
        decoded_dump_name = cvtda.dumping.dump_name_concat(dump_name, f"{name}_decoded")
        if only_get_from_dump or dumper.has_dump(decoded_dump_name):
            # Get from dump, if available
            decoded = dumper.get_dump(decoded_dump_name)
        else:
            # Train
            model.fit(nn_train, nn_test)

            # Test
            encoded = model.encode(nn_test)
            decoded = model.decode(encoded)

            # Dump if requested
            if encoded_dump_name is not None:
                dumper.save_dump(encoded, encoded_dump_name)
            if decoded_dump_name is not None:
                dumper.save_dump(decoded, decoded_dump_name)

        # Calculate quality and return
        result = {"model": display_name, **estimate_quality(decoded, test_images)}
        cvtda.logging.logger().print(result)
        return result

    # Initialize models.
    models = [
        Autoencoder(
            random_state=random_state,
            device=nn_device,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate * 10,
            n_epochs=nn_epochs * 2,
            latent_dim=nn_latent_dim,
            skip_diagrams=True,
            skip_images=True,
            skip_features=False,
            base=nn_base,
        ),
        Autoencoder(
            random_state=random_state,
            device=nn_device,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate,
            n_epochs=nn_epochs,
            latent_dim=nn_latent_dim,
            skip_diagrams=True,
            skip_images=False,
            skip_features=True,
            base=nn_base,
        ),
        Autoencoder(
            random_state=random_state,
            device=nn_device,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate,
            n_epochs=nn_epochs,
            latent_dim=nn_latent_dim,
            skip_diagrams=True,
            skip_images=False,
            skip_features=False,
            base=nn_base,
        ),
        Autoencoder(
            random_state=random_state,
            device=nn_device,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate,
            n_epochs=nn_epochs // 2,
            latent_dim=nn_latent_dim,
            skip_diagrams=False,
            skip_images=True,
            skip_features=True,
            base=nn_base,
        ),
    ]

    names = ["features", "images", "features_images", "diagrams"]
    display_names = [
        "FC over topological features",
        "Baseline model",
        "Combined neural network",
        "Trainable vectorization",
    ]

    # Run and return
    return pandas.DataFrame([try_one(*args) for args in zip(models, names, display_names)])
