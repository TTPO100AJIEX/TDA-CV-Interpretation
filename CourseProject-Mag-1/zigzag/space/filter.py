import numpy
import joblib
import scipy.spatial
from sklearn.neighbors import NearestNeighbors


def density_filter(X: numpy.ndarray, k: int, top_p: float, n_jobs: int = -1) -> numpy.ndarray:
    tree = scipy.spatial.KDTree(X)
    rho_k = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(lambda item: tree.query(item, k=[k + 1])[0][0])(item) for item in X
    )
    return X[rho_k <= numpy.quantile(rho_k, top_p)]


def denoise(X: numpy.ndarray, k: int = 20, iterations: int = 2, n_jobs: int = -1) -> numpy.ndarray:
    for _ in range(iterations):
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs, algorithm="kd_tree").fit(X)
        _, idx = nn.kneighbors(X)
        X = X[idx].mean(axis=1)
    return X
