import typing

import torch
import pandas
import joblib
import cvtda.logging
import torch.utils.data

import zigzag.nn
import zigzag.utils
import zigzag.topology

from .utils import Params
from .utils import Verbosity
from .analysis import analyze_knn_graphs


def compute_knn_graph(
    layer_num: int,
    hidden_state: torch.Tensor,
    k_neighbors: typing.List[int],
    dumper: zigzag.utils.UniversalDumper,
    class_labels: typing.Optional[torch.Tensor] = None,
):
    cvtda.logging.logger().print(f"Processing hidden states of shape {hidden_state.shape}")
    if len(hidden_state.shape) == 4:
        features = dumper.execute(
            zigzag.topology.make_features_layer,
            f"features/{layer_num}",
            hidden_state,
            layer_num,
            dump_name=f"{dumper.directory_}/features",
        )
        # persistence_diagrams = dumper.execute(
        #     zigzag.topology.make_cubical, f"persistence_diagrams/{layer_num}", hidden_state
        # )

    class_names = [None]
    if class_labels is not None:
        class_names.extend(list(sorted(torch.unique(class_labels))))

    for k in k_neighbors:
        for class_name in class_names:
            name_suffix = f"{k}_neighbors/knn_graphs/{layer_num}"
            if class_name is not None:
                name_suffix = f"class_{class_name}/{name_suffix}"

            def get(data):
                if class_name is not None:
                    return data[class_labels == class_name]
                return data

            if len(hidden_state.shape) == 4:
                dumper.execute(zigzag.topology.make_knn_graph_vector, f"vectorizer/{name_suffix}", get(features), k)
                # dumper.execute(
                #     zigzag.topology.make_knn_graph_pds,
                #     f"cubical_landscape/{name_suffix}",
                #     get(persistence_diagrams),
                #     k,
                #     "landscape",
                # )
            else:
                dumper.execute(zigzag.topology.make_knn_graph_vector, f"vector/{name_suffix}", get(hidden_state), k)


def compute_knn_graphs(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    k_neighbors: typing.List[int],
    dumper: zigzag.utils.UniversalDumper,
    class_labels: typing.Optional[torch.Tensor] = None,
):
    if dumper.has_dump("num_layers"):
        return dumper.get_dump("num_layers")["num_layers"].iloc[0]
    for i, hidden_state in enumerate(zigzag.nn.yield_hidden_states(model, dataset)):
        compute_knn_graph(i, hidden_state, k_neighbors, dumper, class_labels)
    dumper.save_dump(pandas.DataFrame([{"num_layers": i + 1}]), "num_layers")
    return i + 1


def analyze_bulk(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    params: typing.Union[Params, typing.List[Params]],
    dumper: zigzag.utils.UniversalDumper,
    class_labels: typing.Optional[torch.Tensor] = None,
):
    params: typing.List[Params] = params if type(params) is list else [params]
    num_layers = compute_knn_graphs(model, dataset, [param.k_neighbors for param in params], dumper, class_labels)
    for param in params:
        param.num_layers = num_layers

    class_names = [None]
    if class_labels is not None:
        class_names.extend(list(sorted(torch.unique(class_labels))))

    def analyze_one(analyzer: str, class_name: typing.Optional[int], param: Params):
        if class_name is not None:
            subdumper = dumper.make_subdumper(f"{analyzer}/class_{class_name}/{param.k_neighbors}_neighbors")
        else:
            subdumper = dumper.make_subdumper(f"{analyzer}/{param.k_neighbors}_neighbors")

        if not subdumper.has_dump("knn_graphs"):
            cvtda.logging.logger().print(f"No knn graphs at {subdumper.directory_}, skipping")
            return

        cvtda.logging.logger().print(f"Started {subdumper.directory_}")
        # with cvtda.logging.DevNullLogger():
        #     analyze_knn_graphs(subdumper.get_dump("knn_graphs"), param, subdumper)
        analyze_knn_graphs(subdumper.get_dump("knn_graphs"), param, subdumper)
        cvtda.logging.logger().print(f"Finished {subdumper.directory_}")

    joblib.Parallel(n_jobs=1)(
        joblib.delayed(analyze_one)(analyzer, class_name, param)
        for class_name in class_names
        for analyzer in ["vector", "vectorizer", "cubical_landscape"]
        for param in params
    )
