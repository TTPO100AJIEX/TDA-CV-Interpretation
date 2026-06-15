import numpy
from sklearn.neighbors import NearestNeighbors
from numpy.lib.stride_tricks import sliding_window_view

from .basis import PATCH_CONTRAST_LAPLACIAN
from .basis import MUMFORD_BASIS


def build_patch_cloud(
    images: numpy.ndarray,
    contrast_pct: float = 0.20,
    max_points: int = 50000,
    seed: int = 42
) -> numpy.ndarray:
    windows = sliding_window_view(images, (3, 3), axis=(1, 2)).reshape(-1, 9)

    log_p = numpy.log(windows + 1e-3)
    centered = log_p - log_p.mean(axis=1, keepdims=True)
    d_n_sq = numpy.einsum("ni,ij,nj->n", centered, PATCH_CONTRAST_LAPLACIAN, centered)
    d_n = numpy.sqrt(numpy.clip(d_n_sq, 0, None))

    threshold = numpy.quantile(d_n, 1 - contrast_pct)
    mask = d_n >= threshold

    hc_centered, hc_norms = centered[mask], d_n[mask]
    cloud = (hc_centered / hc_norms[:, None]) @ PATCH_CONTRAST_LAPLACIAN @ MUMFORD_BASIS

    if len(cloud) > max_points:
        numpy.random.seed(seed)
        idx = numpy.random.choice(len(cloud), max_points, replace=False)
        cloud = cloud[idx]
    return cloud


def density_filter(X: numpy.ndarray, k: int, top_p: float, n_jobs: int = -1) -> numpy.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(X)
    rho_k = nn.kneighbors(X)[0][:, -1]
    return X[rho_k <= numpy.quantile(rho_k, top_p)]
