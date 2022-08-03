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

def get_microbatch_size(shard_data: ModuleShardData, verify: bool=False):
    """Get the microbatch size from shard data."""
    if isinstance(shard_data, Tensor):
        shard_data = (shard_data,)
    ubatch_size = 0 if len(shard_data) == 0 else len(shard_data[0])
    if verify:
        # Sanity check that tensors are the same length
        for tensor in shard_data:
            assert isinstance(tensor, Tensor)
            assert len(tensor) == ubatch_size
    return ubatch_size
