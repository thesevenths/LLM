"""
dynamics.normalization
======================

Train-only normalisation for raw-state dynamics experiments.

The normalisation is chosen so that the residual identity holds EXACTLY in the
normalised coordinate system:

    x_norm       = (x       - state_mean) / state_std
    u_norm       = (u       - cond_mean)  / cond_std
    delta_norm   = (x_next  - x)         / state_std      <-- NOTE: state_std
    =>  x_norm_next = x_norm + delta_norm

Using ``state_std`` (rather than a separate delta std) for the increment is what
makes ``x + delta`` and all baselines (including Persistence) consistent. All
statistics are computed from the TRAIN split only and persisted to disk; the
evaluation script reloads them and applies the identical transform to
validation / frozen_test. There is deliberately no ``fit`` method that accepts
a test split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
DEFAULT_EPS = 1e-8


@dataclass
class NormalizationStats:
    """Per-dimension train statistics (all plain CPU float32 tensors)."""

    state_mean: torch.Tensor          # [D]
    state_std: torch.Tensor           # [D], clamped to >= eps
    cond_mean: Optional[torch.Tensor]  # [U] or None
    cond_std: Optional[torch.Tensor]   # [U] or None
    eps: float = DEFAULT_EPS
    source_split: str = "train"
    n_valid_transitions: int = 0

    # -- transforms --------------------------------------------------------- #
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        m = self.state_mean.to(x.device, x.dtype)
        s = self.state_std.to(x.device, x.dtype)
        return (x - m) / s

    def denormalize_state(self, x_norm: torch.Tensor) -> torch.Tensor:
        m = self.state_mean.to(x_norm.device, x_norm.dtype)
        s = self.state_std.to(x_norm.device, x_norm.dtype)
        return x_norm * s + m

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Map a raw-space increment into the normalised residual space."""
        s = self.state_std.to(delta.device, delta.dtype)
        return delta / s

    def denormalize_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        s = self.state_std.to(delta_norm.device, delta_norm.dtype)
        return delta_norm * s

    def normalize_condition(self, u: torch.Tensor) -> torch.Tensor:
        if self.cond_mean is None or self.cond_std is None:
            raise ValueError(
                "this NormalizationStats has no condition statistics; the "
                "training split was unconditioned"
            )
        m = self.cond_mean.to(u.device, u.dtype)
        s = self.cond_std.to(u.device, u.dtype)
        return (u - m) / s

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "cond_mean": None if self.cond_mean is None else self.cond_mean.tolist(),
            "cond_std": None if self.cond_std is None else self.cond_std.tolist(),
            "eps": float(self.eps),
            "source_split": self.source_split,
            "n_valid_transitions": int(self.n_valid_transitions),
        }

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizationStats":
        return cls(
            state_mean=torch.tensor(d["state_mean"], dtype=torch.float32),
            state_std=torch.tensor(d["state_std"], dtype=torch.float32),
            cond_mean=(
                None if d.get("cond_mean") is None
                else torch.tensor(d["cond_mean"], dtype=torch.float32)
            ),
            cond_std=(
                None if d.get("cond_std") is None
                else torch.tensor(d["cond_std"], dtype=torch.float32)
            ),
            eps=float(d.get("eps", DEFAULT_EPS)),
            source_split=str(d.get("source_split", "train")),
            n_valid_transitions=int(d.get("n_valid_transitions", 0)),
        )

    @classmethod
    def load(cls, path) -> "NormalizationStats":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_normalization(
    train_batch,
    eps: float = DEFAULT_EPS,
) -> NormalizationStats:
    """Fit statistics on a TRAIN :class:`~dynamics.contracts.DynamicsBatch` only.

    Only ``valid_mask == True`` transitions contribute. The function signature
    accepts a single batch to make it impossible to accidentally mix splits:
    callers must pass the train split explicitly.
    """
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError(f"eps must be a positive number, got {eps!r}")
    x_t, u_t, _x_next, mask = train_batch.to_training_pairs()
    if x_t.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"observations must be float, got {x_t.dtype}")
    if not torch.isfinite(x_t).all():
        raise ValueError("train observations contain non-finite values (NaN/Inf)")
    if mask.dtype != torch.bool:
        raise TypeError("valid_mask must be bool")
    if int(mask.sum().item()) < 1:
        raise ValueError("cannot fit normalisation: no valid train transitions")

    xv = x_t[mask].detach().cpu().to(torch.float64)
    state_mean = xv.mean(dim=0).to(torch.float32)
    state_std = xv.std(dim=0, unbiased=False).to(torch.float32).clamp_min(float(eps))

    if u_t is not None:
        if u_t.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"conditions must be float, got {u_t.dtype}")
        if not torch.isfinite(u_t).all():
            raise ValueError("train conditions contain non-finite values (NaN/Inf)")
        uv = u_t[mask].detach().cpu().to(torch.float64)
        cond_mean = uv.mean(dim=0).to(torch.float32)
        cond_std = uv.std(dim=0, unbiased=False).to(torch.float32).clamp_min(float(eps))
    else:
        cond_mean = None
        cond_std = None

    return NormalizationStats(
        state_mean=state_mean,
        state_std=state_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
        eps=float(eps),
        source_split="train",
        n_valid_transitions=int(mask.sum().item()),
    )


__all__ = ["NormalizationStats", "fit_normalization", "DEFAULT_EPS"]
