"""Persistence-homology experiments on the patch cloud.

Replicates the protocol of Carlsson, Ishkhanov, de Silva, Zomorodian (2008),
"On the Local Behavior of Spaces of Natural Images" (IJCV 76:1-12):

  - Density filter ρ_k → X(k, p) (§2.3)
  - Random subsample |S| = 10000 from X(k, p) (§6, §9)
  - Two iterations of nearest-neighbour denoising, k=20 (§2.4, §9)
  - Weak witness complex (de Silva 2003) with l landmarks (§2.2)
  - Repeat across multiple seeds for stability (§6: "We repeat this many times
    ... to make sure the results are stable")
"""

from __future__ import annotations

import typing

import gudhi
import numpy
import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from .cloud import density_filter

REFERENCE_SIZE = 50000

# Carlsson et al. (2008): expected Betti signatures over Z/2 coefficients
# at each density-filter scale on natural-image patches.
CARLSSON_SIGNATURES = {
    (300, 0.30): ("primary circle", (1, 1, 0)),
    (15, 0.30): ("three-circle", (1, 5, 0)),
    (100, 0.10): ("Klein bottle", (1, 2, 1)),
}


def denoise(X: numpy.ndarray, k: int = 20, iterations: int = 2) -> numpy.ndarray:
    """Replace each point by the mean of its k nearest neighbours (§2.4).

    Carlsson uses 2 iterations with k=20 (§9). Reduces sampling noise so the
    witness complex has fewer spurious simplices near the diagonal.
    """
    for _ in range(iterations):
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
        _, idx = nn.kneighbors(X)
        X = X[idx].mean(axis=1)
    return X


def compute_persistence_witness(
    points: numpy.ndarray,
    n_landmarks: int = 500,
    maxdim: int = 2,
    max_alpha_square: float = 2.0,
    seed: int = 0,
) -> list[numpy.ndarray]:
    """Weak witness complex persistent homology (de Silva 2003).

    Landmarks are a random subset of `points` of size `n_landmarks`; every point
    serves as a witness. Filtration values are returned as α (not α²) so the
    barcode scale matches Carlsson's figures.
    """
    rng = numpy.random.RandomState(seed)
    n = len(points)
    n_landmarks = min(n_landmarks, n)
    landmark_idx = rng.choice(n, n_landmarks, replace=False)
    landmarks = points[landmark_idx].tolist()
    witnesses = points.tolist()

    wc = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
    st = wc.create_simplex_tree(max_alpha_square=max_alpha_square, limit_dimension=maxdim)
    diag = st.persistence(homology_coeff_field=2, min_persistence=0.0)

    dgms: list[list[list[float]]] = [[] for _ in range(maxdim + 1)]
    for dim, (b, d) in diag:
        if dim > maxdim:
            continue
        b = numpy.sqrt(max(b, 0.0))
        d = numpy.inf if numpy.isinf(d) else numpy.sqrt(max(d, 0.0))
        dgms[dim].append([b, d])

    return [numpy.array(d) if d else numpy.empty((0, 2)) for d in dgms]


def scale_k(k_paper: int, current_size: int, reference_size: int = REFERENCE_SIZE) -> int:
    """Translate the paper's k (calibrated to 5e4 reference points) to current size."""
    return max(1, int(round(k_paper * current_size / reference_size)))


def run_topology_experiments(
    X: numpy.ndarray,
    configs: typing.List[typing.Tuple[int, float]],
    maxdim: int = 2,
    n_landmarks: int = 200,
    subsample_size: int = 10000,
    n_seeds: int = 3,
    denoise_k: int = 20,
    denoise_iterations: int = 2,
    max_alpha_square: float = 2.0,
) -> dict[tuple[int, float], dict]:
    """Carlsson's protocol per config: density-filter → repeat over seeds of
    (subsample |S|=10000 → denoise → witness complex).
    """
    results = {}
    for k_paper, p in tqdm.tqdm(configs, desc="topology experiments"):
        k = scale_k(k_paper, len(X))
        print('#', len(X))
        subset = density_filter(X, k, p)
        print('UU')

        dgms_per_seed = []
        for seed in range(n_seeds):
            rng = numpy.random.RandomState(seed)
            print('^')
            if len(subset) > subsample_size:
                idx = rng.choice(len(subset), subsample_size, replace=False)
                S = subset[idx]
            else:
                S = subset
            print('!')
            S_denoised = denoise(S, k=denoise_k, iterations=denoise_iterations)
            print('@')
            dgms = compute_persistence_witness(
                S_denoised, n_landmarks=n_landmarks, maxdim=maxdim,
                max_alpha_square=max_alpha_square, seed=seed,
            )
            print('$')
            dgms_per_seed.append(dgms)
        print('&')
        results[(k_paper, p)] = {
            "dgms_per_seed": dgms_per_seed,
            "n_points": len(subset),
            "subsample_size": min(subsample_size, len(subset)),
        }
    return results


def _finite_persistences(dgm: numpy.ndarray) -> numpy.ndarray:
    finite = dgm[numpy.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return numpy.array([])
    return finite[:, 1] - finite[:, 0]


def _signal_count(persistences: numpy.ndarray, ratio: float = 0.4) -> int:
    if len(persistences) == 0:
        return 0
    max_p = float(persistences.max())
    med = float(numpy.median(persistences))
    if max_p < 2.0 * med:
        return 0
    return int((persistences > ratio * max_p).sum())


def _measured_betti(dgms: list[numpy.ndarray], ratio: float = 0.4) -> tuple[int, ...]:
    betti = [1]  # H_0 always has one infinite bar
    for k in range(1, len(dgms)):
        betti.append(_signal_count(_finite_persistences(dgms[k]), ratio))
    return tuple(betti)


def _consensus_betti(dgms_per_seed: list[list[numpy.ndarray]], ratio: float = 0.4) -> tuple[int, ...]:
    """Median per-dim Betti across seeds — Carlsson's stability check."""
    betti_per_seed = numpy.array([_measured_betti(d, ratio) for d in dgms_per_seed])
    return tuple(int(numpy.median(betti_per_seed[:, k])) for k in range(betti_per_seed.shape[1]))


def plot_topology_results(
    results: dict[tuple[int, float], dict],
    signatures: dict | None = None,
    top_k: int = 5,
    ratio: float = 0.4,
) -> plt.Figure:
    """Persistence diagrams per config with all seeds overlaid.

    Tight clusters of points across seeds = stable topological feature.
    Scattered points across seeds = sampling noise.
    """
    if signatures is None:
        signatures = CARLSSON_SIGNATURES

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.8), squeeze=False)
    axes = axes[0]

    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "s", "^"]

    for ax, ((k_paper, p), entry) in zip(axes, results.items()):
        dgms_per_seed = entry["dgms_per_seed"]
        n_seeds = len(dgms_per_seed)

        scales = []
        for dgms in dgms_per_seed:
            for dgm in dgms:
                finite = dgm[numpy.isfinite(dgm[:, 1])]
                if len(finite):
                    scales.append(finite[:, 1].max())
        scale = max(scales, default=1.0)
        pad = 0.05 * scale

        ax.plot([0, scale + pad], [0, scale + pad], color="black", linestyle="--", linewidth=0.8, alpha=0.6)

        info_lines = []
        ndims = len(dgms_per_seed[0])
        for dim in range(ndims):
            all_finite = []
            for dgms in dgms_per_seed:
                f = dgms[dim][numpy.isfinite(dgms[dim][:, 1])]
                if len(f):
                    all_finite.append(f)
            if not all_finite:
                info_lines.append(fr"$H_{dim}$: —")
                continue
            stacked = numpy.vstack(all_finite)
            ax.scatter(
                stacked[:, 0], stacked[:, 1],
                c=colors[dim], marker=markers[dim], s=28, alpha=0.35,
                edgecolors="none", label=fr"$H_{dim}$",
            )
            for dgms in dgms_per_seed:
                f = dgms[dim][numpy.isfinite(dgms[dim][:, 1])]
                if not len(f):
                    continue
                persistences = f[:, 1] - f[:, 0]
                order = numpy.argsort(-persistences)[:top_k]
                ax.scatter(
                    f[order, 0], f[order, 1],
                    c=colors[dim], marker=markers[dim], s=140,
                    edgecolors="black", linewidths=1.0, zorder=5, alpha=0.7,
                )
            top_per_seed = []
            for dgms in dgms_per_seed:
                p_seed = _finite_persistences(dgms[dim])
                top3 = numpy.sort(p_seed)[::-1][:3] if len(p_seed) else numpy.zeros(3)
                top_per_seed.append(numpy.pad(top3, (0, 3 - len(top3))))
            medians = numpy.median(numpy.stack(top_per_seed), axis=0)
            info_lines.append(fr"$H_{dim}$ med: " + ", ".join(f"{v:.2f}" for v in medians))

        consensus = _consensus_betti(dgms_per_seed, ratio=ratio)
        name, expected = signatures.get((k_paper, p), (None, None))
        if expected is not None:
            match = "✓" if consensus == tuple(expected) else "✗"
            verdict = fr"consensus $\beta = {consensus}$ vs {name} {expected} {match}"
        else:
            verdict = fr"consensus $\beta = {consensus}$"

        ax.text(
            0.02, 0.98, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85),
        )

        ax.set_xlim(-pad, scale + pad)
        ax.set_ylim(-pad, scale + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        title = (
            f"X({k_paper}, {int(p * 100)}%) — {entry['n_points']} pts, "
            f"|S|={entry['subsample_size']}, {n_seeds} seeds\n{verdict}"
        )
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig
