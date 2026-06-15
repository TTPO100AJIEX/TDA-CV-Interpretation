import typing

import numpy
import gudhi.hera

import cvtda.neural_network
from .BaseLearner import BaseLearner


class DiagramsLearner(BaseLearner):
    """
    Face recognition model that uses persistence diagrams
    with the bottleneck distance as the latent space.
    """

    def fit(self, train: cvtda.neural_network.Dataset, val: typing.Optional[cvtda.neural_network.Dataset]):
        pass

    def prepare_diagram_(self, diagram: numpy.ndarray, dimension: int) -> numpy.ndarray:
        dim_filter = diagram[:, 2] == dimension
        non_degenerate_filter = diagram[:, 0] < diagram[:, 1]
        return diagram[dim_filter & non_degenerate_filter][:, 0:2]

    def calculate_distance_(self, first: int, second: int, dataset: cvtda.neural_network.Dataset):
        diagrams1 = dataset.raw_diagrams[first]
        diagrams2 = dataset.raw_diagrams[second]
        assert len(diagrams1) == len(diagrams2)

        distance_vector = []
        for i in range(0, len(diagrams1), 2):
            diag1 = diagrams1[i]
            diag2 = diagrams2[i]
            for dim in numpy.unique(diag1[:, 2]):
                d1 = self.prepare_diagram_(diag1, dim)
                d2 = self.prepare_diagram_(diag2, dim)
                distance_vector.append(gudhi.hera.bottleneck_distance(d1, d2))

        return numpy.sqrt(numpy.sum(numpy.array(distance_vector) ** 2))
