"""Transformers module."""
from typing import Tuple, Type, Union
from torch import nn, Tensor
from transformers import PretrainedConfig
from .. import ModuleShard, ModuleShardConfig

TransformerShardData: Type = Union[Tensor, Tuple[Tensor, Tensor]]
"""A transformer shard input/output type."""

class TransformerShard(ModuleShard):
    """Abstract parent class for transformer shards."""
    # pylint: disable=abstract-method

    def __init__(self, config: PretrainedConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.model_layers = nn.ModuleList()
