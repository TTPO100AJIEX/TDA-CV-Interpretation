"""Klein-bottle image features.

Each high-contrast 3x3 patch of an image is assigned to its nearest Klein-bottle
codeword (its (alpha, beta) coordinates). The image-level descriptor is the
D-norm-weighted histogram of codeword indices on a spatial pyramid, normalised
per spatial cell and then globally. The output is a HOG-like descriptor with an
extra "profile" axis (linear edge <-> quadratic ridge) on top of orientation.
"""

from __future__ import annotations

import numpy
import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .basis import PATCH_CONTRAST_LAPLACIAN
from .basis import MUMFORD_BASIS
from .codebook import build_codebook

EPS = 1e-3


class FeatureExtractor:
    def __init__(
        self,
        n_alpha: int = 16,
        n_beta: int = 8,
        n_spatial: int = 4,
        contrast_pct: float = 0.20,
        batch_size: int = 200,
    ) -> None:
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_spatial = n_spatial
        self.contrast_pct = contrast_pct
        self.batch_size = batch_size

        self.codebook, self.alphas, self.betas = build_codebook(n_alpha, n_beta)
        self.n_codes = len(self.codebook)
        self.feature_dim = n_spatial * n_spatial * self.n_codes

        self._nn = NearestNeighbors(n_neighbors=1).fit(self.codebook)
        self.threshold_ = None

    def _patches_dnorm(self, images: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        B, H, W = images.shape
        Hp, Wp = H - 2, W - 2
        wins = sliding_window_view(images, (3, 3), axis=(1, 2)).reshape(B, Hp * Wp, 9)
        log_w = numpy.log(wins + EPS)
        centered = log_w - log_w.mean(axis=2, keepdims=True)
        d_n_sq = numpy.einsum("bki,ij,bkj->bk", centered, PATCH_CONTRAST_LAPLACIAN, centered)
        d_n = numpy.sqrt(numpy.clip(d_n_sq, 0, None))
        return centered, d_n

    def fit(self, images: numpy.ndarray, calibration_size: int = 5000) -> "FeatureExtractor":
        n = min(calibration_size, len(images))
        rng = numpy.random.RandomState(42)
        sub = rng.choice(len(images), n, replace=False)

        d_norm_chunks = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = images[sub[start:end]].astype(numpy.float64)
            _, d_n = self._patches_dnorm(batch)
            d_norm_chunks.append(d_n.flatten())
        d_norms = numpy.concatenate(d_norm_chunks)
        self.threshold_ = float(numpy.quantile(d_norms, 1 - self.contrast_pct))
        return self

    def transform(self, images: numpy.ndarray, progress: bool = True) -> numpy.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call fit() before transform().")

        n = len(images)
        H = images.shape[1]
        Hp = Wp = H - 2

        row_grid, col_grid = numpy.divmod(numpy.arange(Hp * Wp), Wp)
        cell_idx_flat = (row_grid * self.n_spatial // Hp) * self.n_spatial + (col_grid * self.n_spatial // Wp)

        features = numpy.zeros((n, self.feature_dim), dtype=numpy.float32)
        iterator = range(0, n, self.batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator, desc="Klein features")

        for start in iterator:
            end = min(start + self.batch_size, n)
            batch = images[start:end].astype(numpy.float64)
            centered, d_n = self._patches_dnorm(batch)
            mask = d_n >= self.threshold_

            for b in range(end - start):
                m = mask[b]
                if not m.any():
                    continue
                normed = centered[b][m] / d_n[b][m, None]
                coords = normed @ PATCH_CONTRAST_LAPLACIAN @ MUMFORD_BASIS
                _, codes = self._nn.kneighbors(coords)
                codes = codes.flatten()
                weights = d_n[b][m].astype(numpy.float32)
                bins = cell_idx_flat[m] * self.n_codes + codes
                features[start + b] = numpy.bincount(bins, weights=weights, minlength=self.feature_dim)

        f = features.reshape(-1, self.n_spatial * self.n_spatial, self.n_codes)
        f = f / (numpy.linalg.norm(f, axis=2, keepdims=True) + 1e-8)
        return normalize(f.reshape(-1, self.feature_dim), norm="l2").astype(numpy.float32)

    def fit_transform(self, images: numpy.ndarray, calibration_size: int = 5000) -> numpy.ndarray:
        return self.fit(images, calibration_size=calibration_size).transform(images)
