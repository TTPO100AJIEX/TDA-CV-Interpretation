# ported from https://github.com/RitAreaSciencePark/ZigZagLLMs/

import typing

import numpy


def persistence_image(diagram: numpy.ndarray, num_layers: int) -> numpy.ndarray:
    freq_array = numpy.zeros((num_layers * 2 + 1, num_layers * 2 + 1))
    for birth, death in diagram:
        freq_array[death, birth] += 1
    return freq_array


def effective_persistence_image(diagram: numpy.ndarray, num_layers: int) -> numpy.ndarray:
    pi = persistence_image(diagram, num_layers)
    result = numpy.zeros((pi.shape[0] // 2 + 1, pi.shape[0] // 2 + 1))
    for i in range(len(pi)):
        for j in range(len(pi[i])):
            eff_d = i // 2 + i % 2
            eff_b = j // 2 + j % 2
            if eff_d != eff_b:
                result[eff_d, eff_b] += pi[i, j]
    return result


def betti_curve(diagram: numpy.ndarray, num_layers: int) -> numpy.ndarray:
    betti = numpy.zeros(num_layers * 2, dtype=int)
    for birth, death in diagram:
        betti[birth:death] += 1
    return betti


def betti_layer(diagram: numpy.ndarray, num_layers: int) -> numpy.ndarray:
    return betti_curve(diagram, num_layers)[::2]


def _ph_similarity(pi: numpy.ndarray, num_layers: int) -> numpy.ndarray:
    result = numpy.zeros((num_layers, num_layers))
    for i in range(num_layers):
        result[i, i] = numpy.sum(pi[i + 1 :, : i + 1])
        for j in range(i + 1, num_layers):
            result[i, j] = result[i, j - 1] - numpy.sum(pi[j, : i + 1])
            result[j, i] = result[i, j]
    return result


def _weights(num_layers: int, alpha: float) -> numpy.ndarray:
    weights = numpy.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            if alpha < 0 and i == j:
                weights[i, j] = 1e-10
            else:
                weights[i, j] = numpy.abs(i - j) ** alpha
    return weights


def inter_layer_persistence(diagram: numpy.ndarray, num_layers: int) -> numpy.ndarray:
    psim = _ph_similarity(effective_persistence_image(diagram, num_layers), num_layers)
    betti = betti_layer(diagram, num_layers)

    result = numpy.zeros((num_layers, num_layers))
    for i in range(num_layers):
        if betti[i] != 0:
            result[i] = psim[i] / betti[i]
    return result


def weighed_inter_layer_persistence(diagrams: numpy.ndarray, num_layers: int, alpha: float) -> numpy.ndarray:
    weights = _weights(num_layers, alpha)
    layer_pers = inter_layer_persistence(diagrams, num_layers)
    return numpy.sum(layer_pers * weights, axis=1) / numpy.sum(weights, axis=1)


def _relative_frequency(diagram: typing.List[numpy.ndarray], num_layers: int, alpha: float, axis: int) -> numpy.ndarray:
    weights = _weights(num_layers, alpha)
    pi = effective_persistence_image(diagram, num_layers)[:-1, :-1]
    result = numpy.sum(pi * weights, axis=axis) / numpy.sum(weights, axis=axis)
    if numpy.sum(result) == 0:
        return result
    return result / numpy.sum(result)


def births_relative_frequency(diagram: numpy.ndarray, num_layers: int, alpha: float) -> numpy.ndarray:
    return _relative_frequency(diagram, num_layers, alpha, axis=1)


def deaths_relative_frequency(diagram: numpy.ndarray, num_layers: int, alpha: float) -> numpy.ndarray:
    return _relative_frequency(diagram, num_layers, alpha, axis=0)
