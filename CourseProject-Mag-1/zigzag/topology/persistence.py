# adapted from https://github.com/RitAreaSciencePark/ZigZagLLMs/

import itertools
import typing

import numpy
import gudhi
import dionysus
import scipy.sparse

import cvtda.logging


def _ranges(indices: numpy.ndarray) -> typing.Iterator[int]:
    for _, group in itertools.groupby(enumerate(indices), lambda pair: pair[1] - pair[0]):
        group = list(group)
        yield group[0][1] + 1
        yield group[-1][1] + 2


def generate_simplex_tree(
    knn_graphs: typing.List[scipy.sparse.csr_matrix], dimension: int
) -> typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]]]:
    simplices: typing.List[typing.List[int]] = []
    simplices_padded: typing.List[typing.List[int]] = []
    simplex_id: typing.Dict[typing.Tuple[int, ...], int] = {}

    for knn_graph in cvtda.logging.logger().pbar(knn_graphs, desc="Generate simplex tree"):
        S = gudhi.SimplexTree()
        for point in range(knn_graph.shape[0]):
            S.insert([point])
        for line in numpy.array(knn_graph.nonzero()).T:
            S.insert(list(line))
        S.expansion(dimension)

        simplices_padded.append([])
        for s in S.get_skeleton(dimension):
            key = tuple(s[0])
            if key not in simplex_id:
                simplex_id[key] = len(simplices)
                simplices.append(s[0])
            simplices_padded[-1].append(simplex_id[key])

    cvtda.logging.logger().print(f"simplices: {len(simplices)}, simplices_padded: {len(simplices_padded)}")
    return simplices, simplices_padded


def compute_layers_with_intersection(simplices_padded: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]:
    layers: typing.List[typing.List[int]] = []
    for i in cvtda.logging.logger().pbar(list(range(2 * len(simplices_padded) - 1)), desc="Intersection layers"):
        layers.append([])
        if i % 2 == 1:
            layers[i] = list(set(simplices_padded[i // 2]).intersection(simplices_padded[(i + 1) // 2]))
        else:
            layers[i] = simplices_padded[i // 2]
    return layers


def compute_filtration_times(
    simplices: typing.List[typing.List[int]], layers: typing.List[typing.List[int]]
) -> typing.Tuple[dionysus.Filtration, typing.List[typing.List[int]]]:
    appearance_matrix = numpy.zeros((len(layers), len(simplices)), dtype=int)
    for k in cvtda.logging.logger().pbar(list(range(len(layers))), desc="Appearance matrix"):
        appearance_matrix[k, layers[k]] = 1
    times = [
        list(_ranges(numpy.where(appearance_matrix[:, i] == 1)[0]))
        for i in cvtda.logging.logger().pbar(list(range(appearance_matrix.shape[1])), desc="Filtration times")
    ]
    return dionysus.Filtration(simplices), times


def compute_zigzag_persistence(
    filtration: dionysus.Filtration, times: typing.List[typing.List[int]]
) -> typing.List[dionysus.Diagram]:
    cone = dionysus.fast_zigzag(filtration, times)
    reduced_matrix, _ = dionysus.homology_persistence(
        cone, method="matrix_v", progress=(cvtda.logging.logger().verbosity() != 0)
    )
    diagrams = []
    for diagram_map in dionysus.init_zigzag_diagrams(reduced_matrix, cone):
        merged_diagram = dionysus.Diagram()
        for _, diagram in diagram_map.items():
            for point in diagram:
                merged_diagram.append(point)
        diagrams.append(merged_diagram)
    return diagrams


def convert_diagrams_to_numpy(diagrams: typing.List[dionysus.Diagram]) -> typing.List[numpy.ndarray]:
    return [
        numpy.array([[int(interval.birth) - 1, int(interval.death) - 1] for interval in diag])
        for diag in cvtda.logging.logger().pbar(diagrams, desc="To numpy")
    ]


def compute_zigzag_barcodes(knn_graphs: typing.List[numpy.ndarray], dimension: int) -> typing.List[numpy.ndarray]:
    simplices, simplices_padded = generate_simplex_tree(knn_graphs=knn_graphs, dimension=dimension)
    layers = compute_layers_with_intersection(simplices_padded)
    filtration, times = compute_filtration_times(simplices, layers)
    diagrams = compute_zigzag_persistence(filtration, times)
    return convert_diagrams_to_numpy(diagrams)
