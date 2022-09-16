"""Transformers module."""
from collections.abc import Mapping
import os
from typing import Tuple, Type, Union
import psutil
from torch import nn, Tensor
from transformers import AutoConfig
from .. import ModuleShard, ModuleShardConfig

TransformerShardData: Type = Union[Tensor, Tuple[Tensor, Tensor]]
"""A transformer shard input/output type."""

class TransformerShard(ModuleShard):
    """Abstract parent class for transformer shards."""
    # pylint: disable=abstract-method

    def __init__(self, shard_config: ModuleShardConfig, model_name: str,
                 model_weights: Union[str, Mapping], load_weight: bool):
        super().__init__(shard_config)
        self.model_name = model_name
        self.model_weights = model_weights
        self.load_weight = load_weight

        self.process = psutil.Process(os.getpid())
        self.config = AutoConfig.from_pretrained(model_name)

        ## operations/transformer layers set
        self.first_ops = nn.ModuleList()
        self.vit_layers = nn.ModuleList()
        self.last_ops = nn.ModuleList()

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
