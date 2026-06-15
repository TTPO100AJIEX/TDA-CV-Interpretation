import os
import typing

import numpy
import torch
import pandas
import xgboost
import catboost
import torchvision
import sklearn.base
import sklearn.ensemble
import sklearn.neighbors
import matplotlib.pyplot as plt

import cvtda.logging
import cvtda.dumping
import cvtda.neural_network

from .NNClassifier import NNClassifier
from .estimate_quality import estimate_quality


def classify(
    # Train
    train_images: numpy.ndarray,
    train_features: numpy.ndarray,
    train_labels: numpy.ndarray,
    train_diagrams: typing.Optional[typing.List[numpy.ndarray]],
    # Test
    test_images: numpy.ndarray,
    test_features: numpy.ndarray,
    test_labels: numpy.ndarray,
    test_diagrams: typing.Optional[typing.List[numpy.ndarray]],
    # General
    label_names: typing.Optional[typing.List[str]] = None,
    confusion_matrix_include_values: bool = True,
    n_jobs: int = -1,
    random_state: int = 42,
    dump_name: typing.Optional[str] = None,
    only_get_from_dump: bool = False,
    # KNN
    knn_neighbours: int = 50,
    # Random forest
    random_forest_estimators: int = 100,
    # Neural networks
    nn_device: torch.device = cvtda.neural_network.default_device,
    nn_batch_size: int = 128,
    nn_learning_rate: float = 1e-3,
    nn_epochs: int = 20,
    nn_base=torchvision.models.resnet34,
    # Gradient boosting
    grad_boost_max_iter: int = 20,
    grad_boost_max_depth: int = 4,
    grad_boost_max_features: float = 0.1,
    # XGBoost
    xgboost_n_classifiers: int = 25,
    xgboost_max_depth: int = 4,
    xgboost_device: str = "gpu",
    # CatBoost
    catboost_iterations: int = 600,
    catboost_depth: int = 4,
    catboost_device: str = ("GPU" if torch.cuda.is_available() else "CPU"),
):
    """
    Tries 9 classification models on the given dataset.
    (1) KNN. See :class:`sklearn.neighbors.KNeighborsClassifier`.
    (2) Random forest. See :class:`sklearn.ensemble.RandomForestClassifier`.
    (3) Gradient boosting. See :class:`sklearn.ensemble.HistGradientBoostingClassifier`.
    (4) CatBoost. See :class:`catboost.CatBoostClassifier`.
    (5) XGBoost. See :class:`xgboost.XGBClassifier`.
    (6) Neural network with only topological features.
    (7) Neural network with only persistence diagrams.
    (8) Neural network with only traditional CNN on raw images.
    (9) Neural network with both topological features and traditional CNN.

    Parameters
    ----------
    train_images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Images of the training set.
    train_features : ``numpy.ndarray``
        (size `num_items x num_features`) Topological features of the training set.
    train_labels : ``numpy.ndarray``
        (size `num_items`) Target class labels for each object of the training set.
    train_diagrams : ``numpy.ndarray``, optional
        (size `num_items x num_diagrams x num_points x 3`) Persistence diagrams of the training set.
        If not provided, model (4) is excluded from the analysis.

    test_images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Images of the test set.
    test_features : ``numpy.ndarray``
        (size `num_items x num_features`) Topological features of the test set.
    test_labels : ``numpy.ndarray``
        (size `num_items`) Target class labels for each object of the test set.
    test_diagrams : ``numpy.ndarray``, optional
        (size `num_items x num_diagrams x num_points x 3`) Persistence diagrams of the test set.
        If not provided, model (4) is excluded from the analysis.

    label_names : ``list[str]``, optional, default None
        Class label names to print in the confusion matrices.
        If not specified, the classes will be labeled sequentially with numbers.
    confusion_matrix_include_values : ``bool``, default True
        Includes values in confusion matrices.
    n_jobs : ``int``, default: ``-1``
        The number of jobs to use for the computation. See :mod:`joblib` for details.
    random_state : ``int``, default ``42``
        The seed to initialize the pseudo random generator.
    dump_name : ``str``, optional, default ``None``
        The root to dump the results in, if dumping is enabled.
    only_get_from_dump : ``bool``, default `False`
        If true, all results will be obtained from dump, and no computations will be performed.

    knn_neighbours : ``int``, default `50`
        Number of neighbors to use in KNN.

    random_forest_estimators : ``int``, default `100`
        The number of trees in the random forest.

    nn_device : ``torch.device``, default `cvtda.neural_network.default_device`
        A :mod:`torch` device to perform the computations on.
    nn_batch_size : ``int``, default `128`
        Batch size to train the neural networks with.
    nn_learning_rate : ``float``, default `1e-3`
        Learning rate to train the neural networks with.
    nn_epochs : ``int``, default `20`
        Number of epochs to train the neural networks for.
    nn_base : ``torch.nn.Module``, default `torchvision.models.resnet34`
        Constructor of the CNN model used as a baseline for comparison.

    grad_boost_max_iter : ``int``, default `20`
        The maximum number of iterations of the boosting process.
    grad_boost_max_depth : ``int``, default `4`
        The maximum depth of each tree.
    grad_boost_max_features : ``float``, default `0.1`
        Proportion of randomly chosen features in each node split.

    xgboost_n_classifiers : ``int``, default `25`
        The number of trees in XGBoost.
    xgboost_max_depth : ``int``, default `4`
        The maximum depth of each tree.
    xgboost_device : ``str``, default `gpu`
        The device to train XGBoost on.

    catboost_iterations : ``int``, default `600`
        The number of trees in CatBoost.
    catboost_depth : ``int``, default `4`
        Depth of a tree in CatBoost.
    catboost_device : ``str``, default `GPU` if available, else `CPU`
        The device to train CatBoost on.

    Returns
    -------
    ``tuple(pandas.DataFrame, matplotlib.pyplot.Figure)``
        (1) A data frame with quality metrics of each model.
        (2) A matplotlib figure with confusion matrices for each model.
        See :func:`estimate_quality` for details.
    """

    without_diagrams = (train_diagrams is None) and (test_diagrams is None)

    # Create datasets
    if (train_images is not None) and (not only_get_from_dump):
        nn_train = cvtda.neural_network.Dataset(
            train_images, train_diagrams, train_features, train_labels, n_jobs=n_jobs, device=nn_device
        )
        nn_test = cvtda.neural_network.Dataset(
            test_images, test_diagrams, test_features, test_labels, n_jobs=n_jobs, device=nn_device
        )

    def classify_one(classifier: sklearn.base.ClassifierMixin, name: str, display_name: str, ax: plt.Axes):
        if without_diagrams and name == "NNClassifier_diagrams":
            cvtda.logging.logger().print(f"Skipping {name} - {classifier}")
            return {}

        cvtda.logging.logger().print(f"Trying {name} - {classifier}")

        dumper = cvtda.dumping.dumper()
        model_dump_name = cvtda.dumping.dump_name_concat(dump_name, name)
        if only_get_from_dump or dumper.has_dump(model_dump_name):
            # Get from dump, if available
            y_pred_proba = dumper.get_dump(model_dump_name)
        else:
            # Train and test
            if isinstance(classifier, NNClassifier):
                # Different inputs for neural networks
                classifier.fit(nn_train, nn_test)
                y_pred_proba = classifier.predict_proba(nn_test)
            else:
                classifier.fit(train_features, train_labels)
                y_pred_proba = classifier.predict_proba(test_features)

            # Dump if requested
            if model_dump_name is not None:
                dumper.save_dump(y_pred_proba, model_dump_name)

        # Calculate quality and return
        ax.set_title(display_name)
        result = {
            "classifier": display_name,
            **estimate_quality(
                y_pred_proba,
                test_labels,
                ax,
                label_names=label_names,
                confusion_matrix_include_values=confusion_matrix_include_values,
            ),
        }
        cvtda.logging.logger().print(result)
        return result

    # Initialize models.
    classifiers = [
        sklearn.neighbors.KNeighborsClassifier(n_jobs=n_jobs, n_neighbors=knn_neighbours),
        sklearn.ensemble.RandomForestClassifier(
            n_estimators=random_forest_estimators, random_state=random_state, n_jobs=n_jobs
        ),
        sklearn.ensemble.HistGradientBoostingClassifier(
            random_state=random_state,
            max_iter=grad_boost_max_iter,
            max_depth=grad_boost_max_depth,
            max_features=grad_boost_max_features,
            verbose=cvtda.logging.logger().verbosity(),
        ),
        catboost.CatBoostClassifier(
            iterations=catboost_iterations,
            depth=catboost_depth,
            random_seed=random_state,
            loss_function="MultiClass",
            devices="0-3",
            task_type=catboost_device,
            verbose=(cvtda.logging.logger().verbosity() != 0),
        ),
        xgboost.XGBClassifier(
            n_jobs=n_jobs, n_estimators=xgboost_n_classifiers, max_depth=xgboost_max_depth, device=xgboost_device
        ),
        NNClassifier(
            random_state=random_state,
            device=nn_device,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate * 10,
            n_epochs=nn_epochs * 2,
            skip_diagrams=True,
            skip_images=True,
            skip_features=False,
            base=nn_base,
        ),
        NNClassifier(
            random_state=random_state,
            device=nn_device,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate,
            n_epochs=nn_epochs // 2,
            skip_diagrams=False,
            skip_images=True,
            skip_features=True,
            base=nn_base,
        ),
        NNClassifier(
            random_state=random_state,
            device=nn_device,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate,
            n_epochs=nn_epochs,
            skip_diagrams=True,
            skip_images=False,
            skip_features=True,
            base=nn_base,
        ),
        NNClassifier(
            random_state=random_state,
            device=nn_device,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate,
            n_epochs=nn_epochs,
            skip_diagrams=True,
            skip_images=False,
            skip_features=False,
            base=nn_base,
        ),
    ]
    names = [
        "KNeighborsClassifier",
        "RandomForestClassifier",
        "HistGradientBoostingClassifier",
        "CatBoostClassifier",
        "XGBClassifier",
        "NNClassifier_features",
        "NNClassifier_diagrams",
        "NNClassifier_images",
        "NNClassifier_features_images",
    ]
    display_names = [
        "KNN",
        "Random forest",
        "Histogram-based boosting",
        "CatBoost",
        "XGBoost",
        "FC over topological features",
        "Trainable vectorization",
        "Baseline model",
        "Combined neural network",
    ]

    # Run models
    figure, axes = plt.subplots(3, 3, figsize=(15, 15))
    df = pandas.DataFrame([classify_one(*args) for args in zip(classifiers, names, display_names, axes.flat)])
    figure.tight_layout()

    # Dump confusion matrices, if needed
    dumper = cvtda.dumping.dumper()
    if (dump_name is not None) and isinstance(dumper, cvtda.dumping.NumpyDumper):
        file = dumper.get_file_name_(cvtda.dumping.dump_name_concat(dump_name, "confusion_matrixes"))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        figure.savefig(file[:-4] + ".svg")
        figure.savefig(file[:-4] + ".png")
        df.to_csv(dumper.get_file_name_(cvtda.dumping.dump_name_concat(dump_name, "quality_metrics.csv"))[:-4])
    return df, figure
