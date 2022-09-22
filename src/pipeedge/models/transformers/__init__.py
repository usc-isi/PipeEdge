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

    @staticmethod
    def parse_forward_data(data: TransformerShardData) -> Tuple[Tensor, Tensor]:
        """Get the `layer` and `skip` tensors from inter-layer data."""
        if isinstance(data, tuple):
            assert len(data) == 2
            t_layer, t_skip = data[0], data[1]
        else:
            t_layer, t_skip = data, data
        assert isinstance(t_layer, Tensor)
        assert isinstance(t_skip, Tensor)
        return t_layer, t_skip
