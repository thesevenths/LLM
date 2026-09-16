"""
data.dataset
============
SciNet++ (2026) -- block-sequence dataset and generator factory.

Modern JEPA models predict *latent representations of future context* rather
than reconstructing raw inputs. To make that concrete for 1D/short time series,
each trajectory of length ``L`` is split into ``num_blocks = L // block_size``
consecutive, non-overlapping blocks. A block is the atomic unit the encoder maps
to a latent vector, and the predictor rolls forward block-by-block:

    z_b --f--> z_{b+1} --f--> ... --f--> z_{b+k}     (k = rollout horizon)

This is what makes the *same* predictor usable for multi-step training (train.py)
and multi-step evaluation (evaluate.py), fixing the old train/eval mismatch.

The factory :func:`build_dataset` selects the generator from the config so the
identical pipeline runs on both the 1D pendulum and the chaotic double pendulum.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .pendulum import PendulumGenerator
from .double_pendulum import DoublePendulumGenerator
from .newton import NewtonGenerator

_GENERATORS = {
    "pendulum": PendulumGenerator,
    "double_pendulum": DoublePendulumGenerator,
    "newton": NewtonGenerator,
}


def build_generator(cfg: dict):
    """Instantiate the generator named by ``cfg['data']['generator']``.

    ``cfg['data']['t_max']`` (optional) sets the physical time-span of each
    trajectory. A longer span makes slow effects -- e.g. the damping rate gamma
    of the pendulum -- identifiable within a single encoder block, which is the
    Direction-2 lever for better concept discovery.
    """
    name = cfg["data"]["generator"]
    if name not in _GENERATORS:
        raise ValueError(
            f"Unknown generator '{name}'. Expected one of {list(_GENERATORS)}."
        )
    length = cfg["data"]["sequence_length"]
    t_max = cfg["data"].get("t_max", None)
    if name == "pendulum":
        return PendulumGenerator(length=length, t_max=t_max if t_max else 10.0)
    if name == "newton":
        # allow per-experiment tuning of force/mass ranges from the config
        kwargs = dict(cfg["data"].get("newton", {}))
        return NewtonGenerator(length=length, t_max=t_max if t_max else 10.0, **kwargs)
    # double pendulum: the analogous knob is the integration duration
    return DoublePendulumGenerator(length=length, duration=t_max if t_max else 10.0)


def encoder_input_dim(cfg: dict) -> int:
    """Dimensionality of a single encoded block = block_size * state_dim."""
    state_dim = _GENERATORS[cfg["data"]["generator"]].state_dim
    return int(cfg["data"]["block_size"]) * int(state_dim)


def state_dim(cfg: dict) -> int:
    """Per-timestep feature count of the selected generator (1 or 4)."""
    return int(_GENERATORS[cfg["data"]["generator"]].state_dim)


def num_blocks(cfg: dict) -> int:
    """Number of blocks per trajectory."""
    return int(cfg["data"]["sequence_length"]) // int(cfg["data"]["block_size"])


def validate_config(cfg: dict) -> None:
    """Fail fast on inconsistent configs (blocks vs. rollout horizons)."""
    nb = num_blocks(cfg)
    max_horizon = max(cfg["jepa"]["rollout_steps"])
    if nb - 1 < max_horizon:
        raise ValueError(
            f"sequence_length/block_size yields {nb} blocks, but rollout_steps "
            f"needs up to {max_horizon} (require num_blocks-1 >= max horizon). "
            f"Increase data.sequence_length or shrink jepa.rollout_steps."
        )
    if cfg["data"]["sequence_length"] % cfg["data"]["block_size"] != 0:
        raise ValueError("data.sequence_length must be divisible by data.block_size.")


class BlockSequenceDataset(Dataset):
    """Wrap raw trajectories into per-block tensors plus concept labels."""

    def __init__(self, X: np.ndarray, labels: np.ndarray, block_size: int) -> None:
        n, length, state_dim = X.shape
        nb = length // block_size
        trimmed = X[:, : nb * block_size, :]
        # (n, num_blocks, block_size * state_dim)
        blocks = trimmed.reshape(n, nb, block_size * state_dim)

        self.blocks = np.ascontiguousarray(blocks, dtype=np.float32)
        self.labels = np.ascontiguousarray(labels, dtype=np.float32)
        self.num_blocks = nb
        self.block_dim = block_size * state_dim
        self.label_dim = self.labels.shape[1]

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, idx: int):
        return {
            "blocks": torch.from_numpy(self.blocks[idx]),   # (num_blocks, block_dim)
            "labels": torch.from_numpy(self.labels[idx]),   # (label_dim,)
        }


def build_dataset(cfg: dict, split: str = "train") -> BlockSequenceDataset:
    """Generate a dataset for ``split`` in {'train', 'test'} from the config."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    generator = build_generator(cfg)
    n = cfg["data"]["train_samples"] if split == "train" else cfg["data"]["test_samples"]

    X, labels = generator.sample(n)
    dataset = BlockSequenceDataset(X, labels, cfg["data"]["block_size"])
    # expose concept names for downstream stages (probe / symbolic regression)
    dataset.label_names = generator.label_names
    return dataset
