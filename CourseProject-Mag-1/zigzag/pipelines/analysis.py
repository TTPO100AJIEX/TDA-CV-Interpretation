import typing

import numpy
import torch
import joblib
import cvtda.logging
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import zigzag.utils
import zigzag.topology

from .utils import Params
from .utils import Verbosity


def analyze_knn_graphs(knn_graphs: typing.List[csr_matrix], params: Params, dumper: zigzag.utils.UniversalDumper):
    diagrams = dumper.execute(zigzag.topology.compute_zigzag_barcodes, "diagrams", knn_graphs, params.dimension)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for diagram, ax in zip(diagrams, axes.flat):
        zigzag.topology.plot_persistence_image(diagram, params.num_layers, ax=ax)
    fig.savefig(f"{dumper.directory_}/persistence_image.png")
    fig.savefig(f"{dumper.directory_}/persistence_image.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for diagram, ax in zip(diagrams, axes.flat):
        zigzag.topology.plot_weighted_inter_layer_persistence(diagram, params.num_layers, ax=ax)
    fig.savefig(f"{dumper.directory_}/inter_layer_persistence.png")
    fig.savefig(f"{dumper.directory_}/inter_layer_persistence.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for diagram, ax in zip(diagrams, axes.flat):
        zigzag.topology.plot_births_relative_frequency(diagram, params.num_layers, ax=ax)
    fig.savefig(f"{dumper.directory_}/births_relative_frequency.png")
    fig.savefig(f"{dumper.directory_}/births_relative_frequency.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    zigzag.topology.plot_persistence_image(diagrams[1], params.num_layers, ax=axes[0])
    zigzag.topology.plot_weighted_inter_layer_persistence(diagrams[1], params.num_layers, ax=axes[1])
    zigzag.topology.plot_births_relative_frequency(diagrams[1], params.num_layers, ax=axes[2])
    fig.savefig(f"{dumper.directory_}/dimension_1.png")
    fig.savefig(f"{dumper.directory_}/dimension_1.svg")
    plt.close(fig)


def analyze_vector(
    hidden_states: typing.List[typing.Union[numpy.ndarray, torch.Tensor]],
    params: Params,
    dumper: zigzag.utils.UniversalDumper,
):
    cvtda.logging.logger().print("Analyzing as vectors")
    subdumper = dumper.make_subdumper(f"{params.k_neighbors}_neighbors")
    knn_graphs = subdumper.execute(
        zigzag.topology.make_knn_graphs_vector, "knn_graphs", hidden_states, params.k_neighbors
    )
    analyze_knn_graphs(knn_graphs, params, subdumper)


def analyze_cubical(
    persistence_diagrams: typing.List[numpy.ndarray], metric: str, params: Params, dumper: zigzag.utils.UniversalDumper
):
    cvtda.logging.logger().print(f"Analyzing as persistence diagrams with metric {metric}")
    subdumper = dumper.make_subdumper(f"{metric}/{params.k_neighbors}_neighbors")
    knn_graphs = subdumper.execute(
        zigzag.topology.make_knn_graphs_pds, "knn_graphs", persistence_diagrams, params.k_neighbors, metric
    )
    analyze_knn_graphs(knn_graphs, params, subdumper)


def analyze_vectorizer(features: typing.List[numpy.ndarray], params: Params, dumper: zigzag.utils.UniversalDumper):
    cvtda.logging.logger().print("Analyzing using vectorization")
    analyze_vector(features, params, dumper)


def analyze_impl(
    hidden_states: typing.List[torch.Tensor],
    params: typing.List[Params],
    dumper: zigzag.utils.UniversalDumper,
    class_labels: typing.Optional[torch.Tensor] = None,
):
    if len(hidden_states[0].shape) == 4:
        features = dumper.execute(
            zigzag.topology.make_features, "features", hidden_states, dump_name=f"{dumper.directory_}/features"
        )
        analyzers = ["vectorizer"]

        # persistence_diagrams = dumper.execute(
        #     zigzag.topology.make_cubical_persistence, "persistence_diagrams", hidden_states
        # )
        # analyzers.append("cubical_landscape")
        # analyzers.append("cubical_persistence_image")
        # analyzers.append("cubical_bottleneck")
    else:
        analyzers = ["vector"]

    def analyze_impl_one(func, *args, **kwargs):
        desc = f"{len(args[0])} x {args[0][0].shape}, k_neighbors = {args[-2].k_neighbors}"
        cvtda.logging.logger().print(f"Started {func.__name__} with {desc}")
        # with cvtda.logging.DevNullLogger():
        #     func(*args, **kwargs)
        func(*args, **kwargs)
        cvtda.logging.logger().print(f"Finished {func.__name__} with {desc}")

    def make_analyze_impl_one_call_params(class_name: typing.Optional[int], analyzer: str, param: Params):
        result = []
        match analyzer:
            case "vector":
                result.append(analyze_vector)
                result.append(hidden_states)
            case "vectorizer":
                result.append(analyze_vectorizer)
                result.append(features)
            case "cubical_landscape":
                result.append(analyze_cubical)
                result.append(persistence_diagrams)
                result.append("landscape")
            case __:
                assert False, f"Unsupported analyzer: {analyzer}"
        subdumper = dumper.make_subdumper(analyzer)

        if class_name is not None:
            result[1] = [hs[class_labels == class_name] for hs in result[1]]
            subdumper = subdumper.make_subdumper(f"class_{int(class_name)}")

        return *result, param, subdumper

    joblib.Parallel(n_jobs=1)(
        joblib.delayed(analyze_impl_one)(*make_analyze_impl_one_call_params(class_name, analyzer, param))
        for class_name in [None, *torch.unique(class_labels or torch.tensor([]))]
        for analyzer in analyzers
        for param in params
    )


def analyze(
    hidden_states: typing.List[torch.Tensor],
    params: typing.Union[Params, typing.List[Params]],
    dumper: zigzag.utils.UniversalDumper,
    class_labels: typing.Optional[torch.Tensor] = None,
    verbosity: Verbosity = Verbosity.LOGS,
):
    params: typing.List[Params] = params if type(params) is list else [params]
    for param in params:
        param.num_layers = len(hidden_states)
    match verbosity:
        case Verbosity.ONLY_PROGRESSBAR:
            for param in cvtda.logging.logger().pbar(params):
                with cvtda.logging.DevNullLogger():
                    analyze_impl(hidden_states, [param], dumper, class_labels)
        case __:
            analyze_impl(hidden_states, params, dumper, class_labels)
