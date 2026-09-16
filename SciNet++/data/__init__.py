"""
data
====
SciNet++ (2026) data package.

Provides the physical trajectory generators (1D damped pendulum and chaotic
double pendulum) and the block-sequence dataset + config-driven factory used by
every stage of the pipeline.
"""

from .pendulum import PendulumGenerator
from .double_pendulum import DoublePendulumGenerator
from .dataset import (
    BlockSequenceDataset,
    build_dataset,
    build_generator,
    encoder_input_dim,
    state_dim,
    num_blocks,
    validate_config,
)

__all__ = [
    "PendulumGenerator",
    "DoublePendulumGenerator",
    "BlockSequenceDataset",
    "build_dataset",
    "build_generator",
    "encoder_input_dim",
    "state_dim",
    "num_blocks",
    "validate_config",
]
