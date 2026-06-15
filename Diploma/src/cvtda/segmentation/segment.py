import typing

import numpy
import torch
import pandas

import cvtda.logging
import cvtda.dumping
import cvtda.neural_network

from .Dataset import Dataset
from .MiniUnet import MiniUnet
from .estimate_quality import estimate_quality


def segment(
    # Train
    train_images: numpy.ndarray,
    train_features: numpy.ndarray,
    train_masks: numpy.ndarray,
    # Test
    test_images: numpy.ndarray,
    test_features: numpy.ndarray,
    test_masks: numpy.ndarray,
    # General
    random_state: int = 42,
    dump_name: typing.Optional[str] = None,
    only_get_from_dump: bool = False,
    # Neural networks
    device: torch.device = cvtda.neural_network.default_device,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    n_epochs: int = 100,
    remove_cross_maps: bool = False,
):
    """
    Tries 4 UNet-based segmentation models on the given dataset.
    (1) Random predictions for baseline.
    (2) Only topological features.
    (3) Only traditional CNN on raw images.
    (4) Both topological features and traditional CNN.

    Parameters
    ----------
    train_images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Images of the training set.
    train_features : ``numpy.ndarray``
        (size `num_items x num_features`) Topological features of the training set.
    train_masks : ``numpy.ndarray``
        (size `num_items`) Target segmentation masks of the training set.

    test_images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Images of the test set.
    test_features : ``numpy.ndarray``
        (size `num_items x num_features`) Topological features of the test set.
    test_masks : ``numpy.ndarray``
        (size `num_items`) Target segmentation masks of the test set.

    random_state : ``int``, default ``42``
        The seed to initialize the pseudo random generator.
    dump_name : ``str``, optional, default ``None``
        The root to dump the results in, if dumping is enabled.
    only_get_from_dump : ``bool``, default `False`
        If true, all results will be obtained from dump, and no computations will be performed.

    device : ``torch.device``, default `cvtda.neural_network.default_device`
        A :mod:`torch` device to perform the computations on.
    batch_size : ``int``, default `64`
        Batch size to train the neural networks with.
    learning_rate : ``float``, default `1e-3`
        Learning rate to train the neural networks with.
    n_epochs : ``int``, default `100`
        Number of epochs to train the neural networks for.

    Returns
    -------
    ``pandas.DataFrame``
        A data frame with quality metrics of each model. See :func:`estimate_quality` for details.
    """
    nn_train = Dataset(train_images, train_features, train_masks)
    nn_test = Dataset(test_images, test_features, test_masks)

    def try_one(model: MiniUnet, name: str, display_name: str):
        cvtda.logging.logger().print(f"Trying {name} - {model}")

        dumper = cvtda.dumping.dumper()
        model_dump_name = cvtda.dumping.dump_name_concat(dump_name, name)
        if only_get_from_dump or dumper.has_dump(model_dump_name):
            # Get from dump, if available
            y_pred_proba = dumper.get_dump(model_dump_name)
        else:
            # Train
            model.fit(nn_train, nn_test)

            # Test
            y_pred_proba = model.predict_proba(nn_test)

            # Dump if requested
            if model_dump_name is not None:
                dumper.save_dump(y_pred_proba, model_dump_name)

        # Calculate quality and return
        result = {"model": display_name, **estimate_quality(y_pred_proba, test_masks)}
        cvtda.logging.logger().print(result)
        return result

    # Initialize models.
    unet_kwargs = dict(
        random_state=random_state,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        remove_cross_maps=remove_cross_maps,
    )
    models = [
        MiniUnet(**unet_kwargs, with_images=False, with_features=False),
        MiniUnet(**unet_kwargs, with_images=True, with_features=False),
        MiniUnet(**unet_kwargs, with_images=False, with_features=True),
        MiniUnet(**unet_kwargs, with_images=True, with_features=True),
    ]

    names = ["no", "images", "topological", "combined"]
    display_names = ["No features", "Without topological features", "Only topological features", "Combined features"]

    # Run and return
    return pandas.DataFrame([try_one(*args) for args in zip(models, names, display_names)])
