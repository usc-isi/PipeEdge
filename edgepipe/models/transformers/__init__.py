"""Transformers module."""
import threading
import os
import psutil
from torch import nn
from transformers import AutoConfig

class TransformerShard(nn.Module):
    """Parent class for transformer shards."""

    def __init__(self, stage, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight=True):
        super().__init__()
        self.stage = stage
        self.model_name = model_name
        self.weights_file_name = model_file
        self.is_first = is_first
        self.is_last = is_last
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.load_weight = load_weight

        self.operators_list = [ "LayerNorm + Attention",
                                "Attention Output + residuel Connection",
                                "LayerNorm + MLP-1",
                                "MLP-2 + residuel Connection" ]
        self.process = psutil.Process(os.getpid())
        self.config = AutoConfig.from_pretrained(model_name)
        self._lock = threading.Lock()

        ## operations/transformer layers set
        self.first_ops = nn.ModuleList()
        self.vit_layers = nn.ModuleList()
        self.last_ops = nn.ModuleList()

        # profiling
        self.total_time = 0
        self.total_batch = 0
        self.total_data = 0
        self.batch_0_finish = 0

    def forward(self, x):
        """Still-abstract forward function."""
        raise NotImplementedError
