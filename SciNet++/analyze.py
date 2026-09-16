"""
analyze.py
==========
SciNet++ (2026) -- Stage 2: physics probe (SciNet concept discovery).

Freezes the trained encoder, extracts the latent representation of each test
trajectory, and trains a small read-out probe to regress the ground-truth
physical concepts (gamma/omega for the pendulum, energy for the double
pendulum). A high probe R^2 means the unsupervised V-JEPA latent space really
did *discover* the underlying physical concept -- the central claim of SciNet.

This stage also writes the artefacts consumed by the AI-Feynman stage
(symbolic.py):
    <output_dir>/latent.npy        (N, latent_dim) frozen test latents
    <output_dir>/labels.npy        (N, label_dim)  ground-truth concepts
    <output_dir>/<concept>.npy     one file per concept (e.g. gamma.npy)
    <output_dir>/label_names.json  ordered concept names

Usage:
    python analyze.py --config configs/pendulum.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import build_dataset, state_dim, validate_config
from models.world_model import WorldModel
from models.probe import Probe
from utils.config import ensure_output_dir, load_config, resolve_device, make_run_id
from utils.metrics import r2, rmse
from utils.plotting import pca_plot, probe_parity_plot, scatter_latent
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physics probe for SciNet++")
    parser.add_argument("--config", default="configs/pendulum.yaml")
    parser.add_argument(
        "--run-id",
        default=None,
        help="reuse a specific run sub-directory (timestamp); auto-generated if omitted",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="force a fresh timestamped run sub-directory even if one exists",
    )
    return parser.parse_args()


@torch.no_grad()
def extract_latents(model, dataset, device, batch_size=256):
    """Encode the context block (block 0) of every trajectory with the frozen encoder.

    IMPORTANT: only block 0 is used. This means the physical concepts must be
    identifiable from a single encoder window -- if the block time-span is too
    short, concept discovery will fail regardless of encoder architecture.
    See the pitfall memory for details.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    zs, ys = [], []
    for batch in loader:
        blocks = batch["blocks"].to(device)
        z = model.encode(blocks[:, 0, :])   # encode ONLY block 0 per trajectory
        zs.append(z.cpu().numpy())
        ys.append(batch["labels"].numpy())   # ground-truth concepts (gamma, omega) or (energy,)
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def best_latent_dim(Z, y):
    """Index of the latent dimension most correlated (|pearson|) with concept y.

    Used to identify which latent dim 'discovered' a given physical concept.
    Vectorised: computes correlation of all dims at once via normalised dot product.

    Example output: 'most-correlated latent dim=z1' means z1 has the highest
    absolute Pearson correlation with this concept -- it's the dim that encodes it.
    """
    Zc = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)  # standardise each latent dim
    yc = (y - y.mean()) / (y.std() + 1e-8)                # standardise concept values
    corr = (Zc.T @ yc) / len(y)                            # pearson correlation per dim
    return int(np.argmax(np.abs(corr)))                    # dim with strongest |correlation|


def train_probe(probe, Z, Y, cfg, device):
    """Fit the probe on standardized targets with mini-batch Adam.

    The probe is a small MLP: latent_dim -> hidden -> num_concepts.
    It learns to map frozen V-JEPA latents to ground-truth physics concepts.
    A high R^2 after training means the latent space genuinely encodes the concept.

    Targets are standardised (zero mean, unit std) so the probe doesn't need to
    learn the scale; predictions are de-standardised before computing R^2/RMSE.
    """
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(Z, dtype=torch.float32),
            torch.as_tensor(Y, dtype=torch.float32),
        ),
        batch_size=256,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(probe.parameters(), lr=cfg["probe"]["lr"])
    for _ in range(cfg["probe"]["epochs"]):
        probe.train()
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            loss = F.mse_loss(probe(zb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return probe


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.new:
        cfg["run_id"] = make_run_id()
    elif args.run_id:
        cfg["run_id"] = args.run_id
    validate_config(cfg)

    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    out_dir = ensure_output_dir(cfg)

    # ---- rebuild model and load the trained weights -----------------------
    # Reconstruct architecture from config, then load trained weights.
    # Freeze ALL parameters: we only extract latents, never modify the encoder.
    model = WorldModel.from_config(cfg, state_dim(cfg)).to(device)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    for p in model.parameters():
        p.requires_grad_(False)   # freeze encoder + predictor + target

    train_set = build_dataset(cfg, split="train")
    test_set = build_dataset(cfg, split="test")
    label_dim = test_set.label_dim
    label_names = list(test_set.label_names)

    # ---- extract frozen latents -------------------------------------------
    # Encode every trajectory's context block (block 0) into a latent vector.
    # These latents are the "discovered concepts" that the probe will test.
    Ztr, Ytr = extract_latents(model, train_set, device)   # train split latents + labels
    Zte, Yte = extract_latents(model, test_set, device)    # test split latents + labels

    # Standardise concept labels using TRAIN statistics (avoid data leakage)
    mu, sd = Ytr.mean(axis=0), Ytr.std(axis=0) + 1e-8
    Ytr_n, Yte_n = (Ytr - mu) / sd, (Yte - mu) / sd       # normalised targets for probe training

    # ---- train + evaluate the physics probe -------------------------------
    # Train a small MLP probe: z -> concept. High R^2 = concept was discovered.
    probe = Probe.from_config(cfg, cfg["model"]["latent_dim"], label_dim).to(device)
    train_probe(probe, Ztr, Ytr_n, cfg, device)

    # Evaluate on TEST latents (never seen by probe during training)
    probe.eval()
    with torch.no_grad():
        pred_n = (
            probe(torch.as_tensor(Zte, dtype=torch.float32, device=device))
            .cpu()
            .numpy()
        )
    pred_raw = pred_n * sd + mu   # de-standardise predictions back to original units

    # Compute R^2 per concept: how much variance does the probe explain?
    r2_scores = np.atleast_1d(r2(Yte, pred_raw))
    print("=== Physics probe (concept discovery) ===")
    for j, name in enumerate(label_names):
        best_dim = best_latent_dim(Zte, Yte[:, j])  # which z-dim encodes this concept?
        print(
            f"  {name:8s}: R2={r2_scores[j]:+.3f}  "
            f"RMSE={rmse(Yte[:, j], pred_raw[:, j]):.4f}  "
            f"most-correlated latent dim=z{best_dim}"
        )

    # ---- persist artefacts for symbolic.py (fixes the broken handoff) -----
    # Save latents + labels so symbolic.py can run PySR: latent -> formula
    np.save(os.path.join(out_dir, "latent.npy"), Zte)     # (N, latent_dim) frozen test latents
    np.save(os.path.join(out_dir, "labels.npy"), Yte)     # (N, num_concepts) ground truth
    for j, name in enumerate(label_names):
        np.save(os.path.join(out_dir, f"{name}.npy"), Yte[:, j])  # per-concept file
    with open(os.path.join(out_dir, "label_names.json"), "w", encoding="utf-8") as fh:
        json.dump(label_names, fh)   # ordered concept names for symbolic.py
    print(f"Saved latents + concept labels to {out_dir}/")

    # ---- figures ----------------------------------------------------------
    pca_plot(Zte, output_dir=out_dir, name="pca")
    probe_parity_plot(Yte, pred_raw, label_names, output_dir=out_dir)
    for j, name in enumerate(label_names):
        best_dim = best_latent_dim(Zte, Yte[:, j])
        scatter_latent(Zte[:, best_dim], Yte[:, j], name, output_dir=out_dir)
    print(f"Wrote plots to {out_dir}/")


if __name__ == "__main__":
    main()
