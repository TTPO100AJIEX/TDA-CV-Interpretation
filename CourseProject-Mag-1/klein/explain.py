"""Per-image interpretation of predictions from a linear classifier
trained on Klein-bottle features.

For a linear model the class score is s_c(x) = w_c . x + b_c, so each feature j
contributes w_{c,j} * x_j to the predicted class. Every feature index decomposes
into (spatial_cell, codeword): the top contributors therefore decode into a
human-readable "lots of [codeword pattern] in [spatial region] supported class c".
"""

from __future__ import annotations

import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.lib.stride_tricks import sliding_window_view

from .basis import PATCH_CONTRAST_LAPLACIAN
from .basis import MUMFORD_BASIS
from .codebook import klein_polynomial
from .features import EPS
from .features import FeatureExtractor


_ROW_WORDS_4 = ["top", "upper", "lower", "bottom"]
_COL_WORDS_4 = ["left", "center-left", "center-right", "right"]


def _spatial_name(cell_idx: int, n_spatial: int) -> str:
    r = cell_idx // n_spatial
    c = cell_idx % n_spatial
    if n_spatial == 4:
        return f"{_ROW_WORDS_4[r]} {_COL_WORDS_4[c]}"
    return f"cell ({r}, {c})"


def _describe_codeword(alpha: float, beta: float) -> str:
    deg = numpy.rad2deg(alpha) % 180
    kind = "edge" if abs(numpy.sin(beta)) > 0.7 else "ridge"
    return f"{kind} @ {deg:.0f}°"


def _assign_patches(image: numpy.ndarray, fe: FeatureExtractor) -> dict | None:
    """For a single image, return per-patch (cell_idx, codeword, row, col, d_norm)."""
    H = image.shape[0]
    Hp = Wp = H - 2

    windows = sliding_window_view(image.astype(numpy.float64), (3, 3)).reshape(-1, 9)
    log_w = numpy.log(windows + EPS)
    centered = log_w - log_w.mean(axis=1, keepdims=True)
    d_n_sq = numpy.einsum("ni,ij,nj->n", centered, PATCH_CONTRAST_LAPLACIAN, centered)
    d_n = numpy.sqrt(numpy.clip(d_n_sq, 0, None))

    mask = d_n >= fe.threshold_
    if not mask.any():
        return None

    normed = centered[mask] / d_n[mask, None]
    coords = normed @ PATCH_CONTRAST_LAPLACIAN @ MUMFORD_BASIS
    _, codes = fe._nn.kneighbors(coords)
    codes = codes.flatten()

    flat_idx = numpy.flatnonzero(mask)
    rows = flat_idx // Wp
    cols = flat_idx % Wp
    cell_idx = (rows * fe.n_spatial // Hp) * fe.n_spatial + (cols * fe.n_spatial // Wp)

    return dict(rows=rows, cols=cols, cell_idx=cell_idx, codes=codes, weights=d_n[mask])


def explain_prediction(
    clf,
    fe: FeatureExtractor,
    image: numpy.ndarray,
    true_label: int | None = None,
    top_k: int = 4,
) -> dict:
    """Return a structured explanation of clf's prediction on a single image.

    Keys:
        predicted_class, true_label, decision (per-class scores),
        top_features (list of dicts: cell, codeword, alpha, beta, contribution, description),
        assignments (per-patch codeword assignments for plotting).
    """
    feat = fe.transform(image[None], progress=False)[0]
    decision = clf.decision_function(feat[None])[0]
    pred = int(decision.argmax())

    w = clf.coef_[pred]
    contrib = w * feat
    top = numpy.argsort(-contrib)[:top_k]

    features = []
    for idx in top:
        idx = int(idx)
        cell = idx // fe.n_codes
        code = idx % fe.n_codes
        alpha = float(fe.alphas[code])
        beta = float(fe.betas[code])
        features.append(
            dict(
                cell=cell,
                codeword=code,
                alpha=alpha,
                beta=beta,
                contribution=float(contrib[idx]),
                description=f"{_describe_codeword(alpha, beta)} in {_spatial_name(cell, fe.n_spatial)}",
            )
        )

    return dict(
        predicted_class=pred,
        true_label=true_label,
        decision=decision,
        top_features=features,
        assignments=_assign_patches(image, fe),
    )


def plot_explanation(image: numpy.ndarray, explanation: dict, fe: FeatureExtractor) -> plt.Figure:
    """Render the explanation: original image + one panel per top contributing feature."""
    top = explanation["top_features"]
    pred = explanation["predicted_class"]
    true = explanation["true_label"]
    assignments = explanation["assignments"]

    Hp = image.shape[0] - 2
    cell_size = Hp / fe.n_spatial

    n_cols = 1 + len(top)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 2.9))

    ax0 = axes[0]
    ax0.imshow(image, cmap="gray")
    if true is not None:
        match = "✓" if pred == true else "✗"
        ax0.set_title(f"true: {true}, pred: {pred} {match}")
    else:
        ax0.set_title(f"pred: {pred}")
    ax0.set_xticks([])
    ax0.set_yticks([])

    for i, f in enumerate(top):
        ax = axes[1 + i]
        ax.imshow(image, cmap="gray", alpha=0.55)

        cell_r = f["cell"] // fe.n_spatial
        cell_c = f["cell"] % fe.n_spatial
        rect = mpatches.Rectangle(
            (cell_c * cell_size - 0.5, cell_r * cell_size - 0.5),
            cell_size + 2,
            cell_size + 2,
            linewidth=1.5,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect)

        if assignments is not None:
            sel = (assignments["cell_idx"] == f["cell"]) & (assignments["codes"] == f["codeword"])
            ax.scatter(assignments["cols"][sel] + 1, assignments["rows"][sel] + 1, s=18, c="red", marker="s", alpha=0.8)

        inset = ax.inset_axes([0.66, 0.66, 0.32, 0.32])
        cw = klein_polynomial(f["alpha"], f["beta"]).reshape(3, 3)
        vmax = numpy.abs(cw).max()
        inset.imshow(cw, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        inset.set_xticks([])
        inset.set_yticks([])
        for sp in inset.spines.values():
            sp.set_edgecolor("yellow")

        ax.set_title(f"{f['description']}\n$w\\cdot x = {f['contribution']:+.3f}$", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig
