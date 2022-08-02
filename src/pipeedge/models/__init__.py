"""Models module."""
from typing import Tuple, Type, Union
from torch import nn, Tensor

ModuleShardData: Type = Union[Tensor, Tuple[Tensor, ...]]
"""A module shard input/output type."""

class ModuleShard(nn.Module):
    """Abstract parent class for module shards."""
    # pylint: disable=abstract-method

    def __init__(self, stage: int, start_layer: int, end_layer: int):
        super().__init__()
        self.stage = stage
        self.start_layer = start_layer
        self.end_layer = end_layer
