"""Klein-bottle codebook on S^7.

Carlsson et al. (2008, sect. 6) showed that the densest 2-manifold inside the
space of high-contrast 3x3 patches is topologically a Klein bottle, parametrised
by the map
    g(alpha, beta) = cos(beta) * (x cos(alpha) + y sin(alpha))^2
                   + sin(beta) * (x cos(alpha) + y sin(alpha))
followed by evaluation on the 3x3 grid H = {-1, 0, 1}^2, mean subtraction, and
D-normalisation. We sample a regular (alpha, beta) grid on the torus S^1 x S^1
and push each sample through this map to obtain a codebook of canonical local
appearances on S^7.
"""

from __future__ import annotations

import numpy

from .basis import PATCH_CONTRAST_LAPLACIAN
from .basis import MUMFORD_BASIS


def klein_polynomial(alpha: float, beta: float) -> numpy.ndarray:
    xs = numpy.array([-1.0, 0.0, 1.0])
    Xg, Yg = numpy.meshgrid(xs, xs, indexing="ij")
    u = Xg * numpy.cos(alpha) + Yg * numpy.sin(alpha)
    return (numpy.cos(beta) * u**2 + numpy.sin(beta) * u).flatten()


def build_codebook(n_alpha: int = 16, n_beta: int = 8) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    alphas_grid = numpy.linspace(0, 2 * numpy.pi, n_alpha, endpoint=False)
    betas_grid = numpy.linspace(0, 2 * numpy.pi, n_beta, endpoint=False)
    alpha = numpy.repeat(alphas_grid, n_beta)
    beta = numpy.tile(betas_grid, n_alpha)

    raw = numpy.array([klein_polynomial(a, b) for a, b in zip(alpha, beta)])
    centered = raw - raw.mean(axis=1, keepdims=True)
    d_sq = numpy.einsum("ni,ij,nj->n", centered, PATCH_CONTRAST_LAPLACIAN, centered)

    ok = d_sq > 1e-9
    centered = centered[ok]
    d_norm = numpy.sqrt(d_sq[ok])
    codebook = (centered / d_norm[:, None]) @ PATCH_CONTRAST_LAPLACIAN @ MUMFORD_BASIS
    return codebook, alpha[ok], beta[ok]
