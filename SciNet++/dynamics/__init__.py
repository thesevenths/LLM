"""
dynamics
=======

Generic, domain-independent dynamics data contract for SciNet++.

This package introduces the *conditioned-dynamics* capability on top of the
existing unconditional Scientific-Modeling core. It is **additive only**:

- It does NOT import anything from the ``robotics`` package (the robotics
  adapter remains a domain adapter, never a dependency of the generic core).
- It does NOT modify ``models/predictor.py``, ``models/world_model.py``,
  ``data/dataset.py``, or the existing JEPA training pipeline.

The single most important object is :class:`DynamicsBatch`, which encodes the
transition-level contract

    (x_t, u_t) -> x_{t+1}

for arbitrary scientific domains (Pendulum, Double-Pendulum, Newton, Pusher-v5,
...). ``conditions = None`` means *unconditioned* (no external control
variable), which is semantically distinct from a zero-valued condition.
"""

from __future__ import annotations

from .contracts import (
    DynamicsBatch,
    dynamics_batch_from_episodes,
    dynamics_batch_from_windows,
    slice_rows,
    training_pairs,
)
from .validation import summarize_batch, validate_dynamics_batch

__all__ = [
    "DynamicsBatch",
    "dynamics_batch_from_episodes",
    "dynamics_batch_from_windows",
    "training_pairs",
    "slice_rows",
    "validate_dynamics_batch",
    "summarize_batch",
]
