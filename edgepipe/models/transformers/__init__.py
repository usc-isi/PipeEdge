"""Transformers module."""
import os
from typing import Tuple, Type, Union
import psutil
from torch import nn, Tensor
from transformers import AutoConfig
from .. import ModuleShard

TransformerShardData: Type = Union[Tensor, Tuple[Tensor, Tensor]]
"""A transformer shard input/output type."""

class TransformerShard(ModuleShard):
    """Abstract parent class for transformer shards."""
    # pylint: disable=abstract-method

    def __init__(self, stage, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight):
        super().__init__(stage, start_layer, end_layer)
        self.model_name = model_name
        self.weights_file_name = model_file
        self.is_first = is_first
        self.is_last = is_last
        self.load_weight = load_weight

        self.operators_list = [ "LayerNorm + Attention",
                                "Attention Output + residuel Connection",
                                "LayerNorm + MLP-1",
                                "MLP-2 + residuel Connection" ]
        self.process = psutil.Process(os.getpid())
        self.config = AutoConfig.from_pretrained(model_name)

        ## operations/transformer layers set
        self.first_ops = nn.ModuleList()
        self.vit_layers = nn.ModuleList()
        self.last_ops = nn.ModuleList()
