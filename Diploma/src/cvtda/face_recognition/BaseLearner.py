import abc
import typing

import numpy
import itertools
import matplotlib.pyplot as plt

import cvtda.utils
import cvtda.logging
import cvtda.neural_network


class BaseLearner(abc.ABC):
    """
    Base class of a face recognition model.
    """

    def __init__(self, n_jobs: int = -1):
        self.n_jobs_ = n_jobs

    @abc.abstractmethod
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
        pass

    def estimate_quality(self, dataset: cvtda.neural_network.Dataset, ax: typing.Optional[plt.Axes] = None):
        """
        Estimates the quality of the model on a given dataset:
        creates a :mod:`matplotlib` figure in a given axis showing the
        distributions of distances between photos of same person and different people.

        Parameters
        ----------
        dataset : ``cvtda.neural_network.Dataset``
            Dataset.
        ax : ``matplotlib.pyplot.Axes``, optional, default None
            A matplotlib axis to draw the result in.
        """

        def calculate_distance_(i: int, j: int):
            return (i, j, self.calculate_distance_(i, j, dataset))

        # Calculate distances for all pairs of objects in parallel
        idxs = list(itertools.product(range(len(dataset)), range(len(dataset))))
        pbar = cvtda.logging.logger().pbar(idxs, desc="Calculating pairwise distances")
        distances_flat = cvtda.utils.parallel(calculate_distance_, pbar, n_jobs=self.n_jobs_)

        # Partition pairs of photos of the same person and different people
        correct_dists, incorrect_dists = {}, {}
        for i, j, distance in cvtda.logging.logger().pbar(distances_flat, desc="Analyzing distances"):
            label1, label2 = dataset.get_labels([i, j])
            if label1 == label2:
                correct_dists[(i, j)] = distance
            else:
                incorrect_dists[(i, j)] = distance
        correct_dists_values = list(correct_dists.values())
        incorrect_dists_values = list(incorrect_dists.values())

        # Draw the results
        if ax is not None:
            ax.set_ylim(0, 1)
            ax.get_yaxis().set_ticks([])
            ax.plot(correct_dists_values, numpy.ones_like(correct_dists_values) * 0.35, "x", label="Same person")
            ax.plot(
                incorrect_dists_values, numpy.ones_like(incorrect_dists_values) * 0.65, "x", label="Different people"
            )
        return correct_dists, incorrect_dists

    @abc.abstractmethod
    def calculate_distance_(self, first: int, second: int, dataset: cvtda.neural_network.Dataset) -> float:
        """
        Calculate the latent distance between two objects in the dataset.

        Parameters
        ----------
        first : ``int``
            Index of the first object in the dataset.
        second : ``int``
            Index of the second object in the dataset.
        dataset : ``cvtda.neural_network.Dataset``
            Dataset.

        Returns
        -------
        ``float``
            Distance between two objects in the latent space.
        """
        pass
