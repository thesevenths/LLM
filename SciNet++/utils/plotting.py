"""
utils.plotting
==============
SciNet++ (2026) -- figure generation for every stage.

All functions write PNGs into a configurable ``output_dir`` (no more hardcoded
"outputs/") and use the non-interactive Agg backend so the pipeline can run
headless on servers. Provided plots:

* ``scatter_latent``    -- one latent dimension vs a ground-truth concept.
* ``pca_plot``          -- 2D PCA of the latent space (concept geometry).
* ``prediction_plot``   -- raw-signal ground truth vs model prediction.
* ``horizon_metric_plot`` -- V-JEPA latent error / cosine vs rollout horizon.
* ``probe_parity_plot`` -- physics-probe parity (predicted vs true concept).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def _save(fig, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def scatter_latent(latent, target, name, output_dir="outputs"):
    """Scatter one latent dimension against a ground-truth concept."""
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(latent, target, s=8, alpha=0.6)
    plt.xlabel("Latent dimension")
    plt.ylabel(name)
    return _save(fig, output_dir, f"latent_vs_{name}")


def pca_plot(z, output_dir="outputs", name="pca"):
    """2D PCA projection of the latent space."""
    y = PCA(n_components=2).fit_transform(np.asarray(z))
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(y[:, 0], y[:, 1], s=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    return _save(fig, output_dir, name)


def prediction_plot(gt, pred, output_dir="outputs", name="prediction"):
    """Overlay a ground-truth signal and the model's prediction."""
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.asarray(gt).ravel(), label="Ground Truth")
    plt.plot(np.asarray(pred).ravel(), label="Prediction")
    plt.legend()
    return _save(fig, output_dir, name)


def horizon_metric_plot(horizons, values, ylabel, output_dir="outputs", name="horizon"):
    """Line plot of a per-horizon metric (V-JEPA multi-step evaluation)."""
    fig = plt.figure(figsize=(6, 4))
    plt.plot(list(horizons), list(values), marker="o")
    plt.xlabel("Rollout horizon (blocks)")
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.xticks(list(horizons), [str(h) for h in horizons])
    plt.grid(True, which="both", alpha=0.3)
    return _save(fig, output_dir, name)


def probe_parity_plot(true, pred, names, output_dir="outputs", name="probe_parity"):
    """Parity plots (predicted vs true) for each physical concept."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    if true.ndim == 1:
        true = true[:, None]
        pred = pred[:, None]
    n_cols = true.shape[1]

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.2), squeeze=False)
    for j in range(n_cols):
        ax = axes[0][j]
        ax.scatter(true[:, j], pred[:, j], s=8, alpha=0.6)
        lo = float(min(true[:, j].min(), pred[:, j].min()))
        hi = float(max(true[:, j].max(), pred[:, j].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xlabel(f"true {names[j]}")
        ax.set_ylabel(f"predicted {names[j]}")
        ax.legend(fontsize=8)
    return _save(fig, output_dir, name)


def attribution_heatmap(
    matrix,
    label_names,
    latent_dim,
    r2_scores=None,
    pr_scores=None,
    output_dir="outputs",
    name="attribution_matrix",
):
    """Latent Attribution Matrix heatmap: concepts (rows) x latent dims (cols).

    Each cell [j, k] shows the effective weight from latent dim z_k to concept j,
    extracted from the trained probe's composed linear map W2 @ W1.

    Uses pcolormesh instead of imshow to avoid the FixedLocator mismatch bug:
    imshow treats the matrix as a pixel image and locks axis ticks to pixel
    positions (e.g. 2 rows * DPI scaling = 16 ticks), which conflicts with
    any attempt to set custom tick labels. pcolormesh works in data coordinates
    so tick counts always match the matrix dimensions.

    Annotations:
      - Cell values show the raw weight (signed contribution).
      - Row headers include R^2 (if provided) so you can see recovery quality.
      - A sidebar or title notes the participation ratio (disentanglement score).

    Colour scale is symmetric around zero (diverging cmap) so positive and
    negative contributions are equally visible.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n_concepts, n_latent = matrix.shape

    # Build y-axis labels with optional R^2 and PR annotations
    ylabels = []
    for j, cname in enumerate(label_names):
        parts = [cname]
        if r2_scores is not None:
            parts.append(f"R²={r2_scores[j]:.3f}")
        if pr_scores is not None:
            parts.append(f"PR={pr_scores[j]:.2f}")
        ylabels.append("  ".join(parts))

    xlabels = [f"z{k}" for k in range(n_latent)]

    # Symmetric colour limits for diverging colormap
    vmax = float(np.abs(matrix).max()) + 1e-8

    fig, ax = plt.subplots(figsize=(max(6, n_latent * 1.4), max(3, n_concepts * 1.2)))

    # pcolormesh uses data coordinates: x edges = [0, 1, ..., n_latent],
    # y edges = [0, 1, ..., n_concepts]. Tick centres land at 0.5, 1.5, ...
    mesh = ax.pcolormesh(
        np.arange(n_latent + 1),          # x edges
        np.arange(n_concepts + 1),         # y edges
        matrix,                            # data (n_concepts x n_latent)
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    # Annotate each cell with its value (centre of each cell = i+0.5, j+0.5)
    for i in range(n_concepts):
        for j in range(n_latent):
            val = matrix[i, j]
            text_color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(j + 0.5, i + 0.5, f"{val:+.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    # Set ticks at cell centres; pcolormesh guarantees these match the data dims
    ax.set_xticks([k + 0.5 for k in range(n_latent)])
    ax.set_xticklabels(xlabels)
    ax.set_yticks([k + 0.5 for k in range(n_concepts)])
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Latent dimension")
    ax.set_title("Latent Attribution Matrix\n(concept → latent dependency)")
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="Effective weight")
    return _save(fig, output_dir, name)
