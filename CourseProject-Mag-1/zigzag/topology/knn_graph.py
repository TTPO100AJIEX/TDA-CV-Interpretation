import typing

import numpy
import torch
import joblib
import scipy.sparse
import cvtda.dumping
import cvtda.logging
import gtda.homology
import gtda.diagrams
import cvtda.topology
import sklearn.neighbors


def make_knn_graph_vector(
    hidden_state: typing.Union[numpy.ndarray, torch.Tensor], k_neighbors: int
) -> scipy.sparse.csr_matrix:
    if hidden_state is torch.Tensor:
        hidden_state = hidden_state.numpy()
    hidden_state = hidden_state.reshape(len(hidden_state), -1)
    return sklearn.neighbors.kneighbors_graph(hidden_state, n_neighbors=k_neighbors, n_jobs=-1)


def make_knn_graphs_vector(
    hidden_states: typing.List[typing.Union[numpy.ndarray, torch.Tensor]], k_neighbors: int
) -> typing.List[scipy.sparse.csr_matrix]:
    return [
        make_knn_graph_vector(hs, k_neighbors) for hs in cvtda.logging.logger().pbar(hidden_states, desc="KNN graphs")
    ]


def make_knn_graph_pds(
    diagrams: numpy.ndarray, k_neighbors: int, metric: str, metric_params: typing.Optional[dict] = None
) -> scipy.sparse.csr_matrix:
    return sklearn.neighbors.kneighbors_graph(
        gtda.diagrams.PairwiseDistance(metric=metric, n_jobs=-1).fit_transform(diagrams),
        n_neighbors=k_neighbors,
        metric="precomputed",
        metric_params=metric_params,
        n_jobs=-1,
    )


def make_knn_graphs_pds(
    persistence: typing.List[numpy.ndarray], k_neighbors: int, metric: str, metric_params: typing.Optional[dict] = None
) -> typing.List[scipy.sparse.csr_matrix]:
    return [
        make_knn_graph_pds(diagrams, k_neighbors, metric, metric_params)
        for diagrams in cvtda.logging.logger().pbar(persistence, desc="KNN graphs")
    ]


def make_cubical(hidden_state: torch.Tensor) -> numpy.ndarray:
    hidden_state = numpy.linalg.norm(hidden_state.numpy(), axis=1)
    return gtda.homology.CubicalPersistence(n_jobs=-1).fit_transform(hidden_state)


def make_cubical_persistence(hidden_states: typing.List[torch.Tensor]) -> typing.List[numpy.ndarray]:
    return [make_cubical(hs) for hs in cvtda.logging.logger().pbar(hidden_states, desc="Cubical persistence")]


def make_features_layer(
    representations: torch.Tensor, layer_idx: int, dump_name: typing.Optional[str] = None
) -> numpy.ndarray:
    def vectorize_channel(representations: torch.Tensor, layer_idx: int, channel_idx: int):
        extractor = cvtda.topology.FeatureExtractor(
            n_jobs=1,
            settings=cvtda.topology.FeatureExtractor.Settings(
                greyscale=cvtda.topology.GreyscaleExtractor.PRESETS.reduced,
                inverted=cvtda.topology.GreyscaleExtractor.PRESETS.reduced,
                filtrations=cvtda.topology.FiltrationsExtractor.Settings(binarizer_thresholds=[]),
                point_clouds=cvtda.topology.PointCloudsExtractor.Settings(enabled=False),
                geometry=cvtda.topology.GeometryExtractor.Settings(
                    gray=cvtda.topology.GrayGeometryExtractor.Settings(
                        daisy=False, sift=False, orb=False, basic=False, curvature=False
                    )
                ),
            ),
        )
        mini, maxi = representations.min(), representations.max()
        if maxi == mini:
            return None
        with cvtda.logging.DevNullLogger():
            return extractor.fit_transform(
                ((representations - mini) / (maxi - mini)).numpy(),
                dump_name=cvtda.dumping.dump_name_concat(dump_name, f"layer_{layer_idx}/channel_{channel_idx}"),
            )

    if representations.shape[2] < 16:
        # Last several layers where locations do not really matter
        return representations.flatten(start_dim=1).numpy()
    res = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(vectorize_channel)(representations[:, channel_idx, :, :], layer_idx, channel_idx)
        for channel_idx in range(representations.shape[1])
    )
    return numpy.hstack([item for item in res if item is not None])


def make_features(
    hidden_states: typing.List[torch.Tensor], dump_name: typing.Optional[str] = None
) -> typing.List[numpy.ndarray]:
    return [
        make_features_layer(hidden_states[i], i, dump_name)
        for i in cvtda.logging.logger().pbar(list(range(len(hidden_states))), desc="Features")
    ]
