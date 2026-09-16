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
