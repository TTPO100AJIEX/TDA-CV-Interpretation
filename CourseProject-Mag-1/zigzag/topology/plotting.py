# ported from https://github.com/RitAreaSciencePark/ZigZagLLMs/

import typing

import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import zigzag.topology.metrics as metrics

_CB_PALETTE = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
    "#FE6100",
    "#FFB000",
    "#3CAB20",
    "#6B750C",
    "#A6761D",
    "#D8A21E",
    "#F0E442",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]


def plot_persistence_image(diagram: numpy.ndarray, num_layers: int, ax=None):
    CMAP = mcolors.LinearSegmentedColormap.from_list("", [(0, "white"), (1, mcolors.to_rgb("#DC267F"))])

    pis = metrics.effective_persistence_image(diagram, num_layers)
    pis_pers = numpy.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            if i - j >= 0:
                pis_pers[i - j, j] = pis[i, j]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(numpy.log10(pis_pers + 1e-8), vmin=0, cmap=CMAP, origin="lower")
    ax.set_xlabel("Слой появления $(\\ell_{\\rm birth})$")
    ax.set_ylabel("Устойчивости $(\\ell_{\\rm death} - \\ell_{\\rm birth})$")
    ax.get_figure().colorbar(im, ax=ax).set_label("Log10 количества симплексов")
    return ax.get_figure().tight_layout()


def plot_weighted_inter_layer_persistence(
    diagram: numpy.ndarray, num_layers: int, alphas: typing.Tuple[float] = (-1.0, 0.0, 0.5, 1, 2), ax=None
) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for alpha, color in zip(alphas, _CB_PALETTE):
        w_pers = metrics.weighed_inter_layer_persistence(diagram, num_layers, alpha)
        ax.plot(w_pers, color=color, label="$\\alpha=%.2f$" % alpha)
    ax.set_xlabel("Номер слоя")
    ax.set_ylabel("Межслойная устойчивость")
    ax.legend()
    return ax.get_figure().tight_layout()


def plot_births_relative_frequency(
    diagram: numpy.ndarray, num_layers: int, alphas: typing.Tuple[float] = (-1.0, 0.0, 0.5, 1, 2), ax=None
) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for alpha, color in zip(alphas, _CB_PALETTE):
        freq = metrics.births_relative_frequency(diagram, num_layers, alpha)
        ax.plot(freq, color=color, label="$\\alpha=%.1f$" % alpha)
    ax.axhline(y=1 / num_layers, label="Равномерное распр.", color="black", linestyle="--", lw=1)
    ax.set_xlabel("Номер слоя")
    ax.set_ylabel("Относительная частота рождений")
    ax.legend()
    return ax.get_figure().tight_layout()
