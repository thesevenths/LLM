"""
tta.py
======
SciNet++ (2026) -- Stage 5: AdaJEPA-style online test-time adaptation.

The previous version of this file was a non-runnable fragment (it referenced
``wm``, ``past``, ``future``, ``torch`` and ``F`` that were never imported or
defined). This is a complete, self-contained script that realises the AdaJEPA
idea: at deployment time, keep adapting to the incoming (unlabelled) data by
minimising the *self-supervised* JEPA objective -- predict the EMA target
encoder's future latents -- with no ground-truth labels at all.

What it does:
    1. Loads the trained world model.
    2. Draws a small unlabelled test batch (optionally corrupted with Gaussian
       observation noise via --noise-std to simulate a deployment shift).
    3. Measures the multi-step latent prediction error BEFORE adaptation.
    4. Runs ``cfg['tta']['steps']`` gradient steps adapting the predictor (and,
       if ``cfg['tta']['adapt_encoder']`` is true, the online encoder + EMA
       target), regularised by VICReg to avoid collapse while adapting.
    5. Reports the per-horizon error AFTER adaptation and the improvement.

Usage:
    python tta.py --config configs/pendulum.yaml
    python tta.py --config configs/pendulum.yaml --noise-std 0.05
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import build_dataset, state_dim, validate_config
from models.world_model import WorldModel
from utils.config import ensure_output_dir, load_config, resolve_device, make_run_id, setup_logging
from utils.losses import prediction_loss, vicreg_loss
from utils.metrics import cosine_similarity
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaJEPA test-time adaptation")
    parser.add_argument("--config", default="configs/pendulum.yaml")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Gaussian observation noise on the test batch (simulates a shift).",
    )
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


def build_tta_batch(cfg: dict, device, noise_std: float) -> torch.Tensor:
    """Draw a small unlabelled batch of block sequences for adaptation.

    TTA works WITHOUT labels -- it uses the self-supervised JEPA objective
    (predict future latents from current latent). The optional Gaussian noise
    simulates a deployment distribution shift: the model was trained on clean
    data but now receives noisy observations, and must adapt using only the
    self-supervised signal.
    """
    tta_cfg = copy.deepcopy(cfg)
    tta_cfg["data"]["test_samples"] = cfg["tta"]["n_samples"]  # override sample count
    dataset = build_dataset(tta_cfg, split="test")
    loader = DataLoader(dataset, batch_size=cfg["tta"]["n_samples"], shuffle=False)
    blocks = next(iter(loader))["blocks"].to(device)
    if noise_std > 0:
        # Add Gaussian observation noise to simulate deployment shift
        blocks = blocks + noise_std * torch.randn_like(blocks)
    return blocks


@torch.no_grad()
def measure(model, blocks, horizons) -> dict:
    """Label-free per-horizon latent MSE + cosine against the EMA target.

    This is the key insight of AdaJEPA: we can MEASURE prediction quality
    without any ground-truth labels, because the EMA target encoder provides
    a self-supervised reference. This allows before/after comparison of TTA.
    """
    model.eval()
    preds, targets, _ = model(blocks, horizons, context_block=0)
    out = {}
    for k in horizons:
        p, t = preds[k], targets[k]
        out[k] = (
            float(torch.mean((p - t) ** 2)),                          # MSE
            cosine_similarity(p.cpu().numpy(), t.cpu().numpy()),      # cosine
        )
    return out


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
    setup_logging(out_dir, name="tta")
    horizons = sorted(cfg["jepa"]["rollout_steps"])

    model = WorldModel.from_config(cfg, state_dim(cfg)).to(device)
    ckpt = torch.load(os.path.join(out_dir, "checkpoint.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    blocks = build_tta_batch(cfg, device, args.noise_std)

    # ---- freeze everything, then unfreeze what TTA adapts -----------------
    # By default, only the predictor is adapted (encoder stays frozen).
    # If adapt_encoder=True, the online encoder also adapts (and EMA target updates).
    for p in model.parameters():
        p.requires_grad_(False)
    adapt = list(model.predictor.parameters())       # always adapt predictor
    if cfg["tta"]["adapt_encoder"]:
        adapt += list(model.encoder.parameters())    # optionally adapt encoder too
    for p in adapt:
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(adapt, lr=cfg["tta"]["lr"])
    var_w = cfg["jepa"]["vicreg_var_weight"]   # VICReg weights reused from training
    cov_w = cfg["jepa"]["vicreg_cov_weight"]

    # Measure performance BEFORE adaptation (baseline)
    before = measure(model, blocks, horizons)

    # ---- label-free self-supervised adaptation loop -----------------------
    # The loss is IDENTICAL to training: pred_loss + VICReg. No labels needed!
    # This is the AdaJEPA insight: the same self-supervised objective that
    # trained the model can continue adapting it at test time.
    model.train()
    for step in range(cfg["tta"]["steps"]):
        preds, targets, z_ctx = model(blocks, horizons, context_block=0)
        pred_l, _ = prediction_loss(preds, targets, horizons)  # predict future latents
        vic_l = vicreg_loss(z_ctx, var_w, cov_w)               # prevent collapse during adaptation
        loss = pred_l + vic_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # If adapting encoder, keep target encoder in sync via EMA
        if cfg["tta"]["adapt_encoder"]:
            model.update_target_encoder(cfg["jepa"]["ema_momentum"])

        if step % max(cfg["tta"]["steps"] // 5, 1) == 0:
            print(f"  tta step {step:3d}  loss={float(loss.detach()):.5f}")

    # Measure performance AFTER adaptation
    after = measure(model, blocks, horizons)

    # ---- report -----------------------------------------------------------
    # Compare before vs after: improvement comes entirely from self-supervised
    # adaptation with NO labels. This demonstrates the AdaJEPA principle.
    print("\n=== AdaJEPA test-time adaptation ===")
    if args.noise_std > 0:
        print(f"(test batch corrupted with Gaussian noise std={args.noise_std})")
    print(f"{'horizon':>8} | {'MSE before':>11} | {'MSE after':>11} | "
          f"{'cos before':>10} | {'cos after':>10}")
    print("-" * 62)
    for k in horizons:
        mb, cb = before[k]
        ma, ca = after[k]
        print(f"{k:>8} | {mb:>11.5f} | {ma:>11.5f} | {cb:>10.4f} | {ca:>10.4f}")

    mean_before = float(np.mean([before[k][0] for k in horizons]))
    mean_after = float(np.mean([after[k][0] for k in horizons]))
    delta = 100.0 * (mean_before - mean_after) / (mean_before + 1e-12)
    print(f"\nMean latent MSE improved by {delta:+.2f}% "
          f"({mean_before:.5f} -> {mean_after:.5f}) with no labels used.")


if __name__ == "__main__":
    main()
