"""BERT transformers."""
from collections.abc import Mapping
import logging
import math
from typing import Union
import numpy as np
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertModel
from transformers.models.bert.modeling_bert import (
    BertEmbeddings, BertIntermediate, BertLayer, BertOutput, BertPooler, BertSelfAttention,
    BertSelfOutput
)
from .. import ModuleShardConfig
from . import TransformerShard, TransformerShardData


logger = logging.getLogger(__name__)


def _forward_kernel(layer, x, skip, kernel_id):
    if kernel_id == 1:
        x = layer[0](x)[0]
    elif kernel_id == 2:
        x = layer[0](x, skip)
        skip = x
    elif kernel_id == 3:
        x = layer[0](x)
    else:
        x = layer[0](x, skip)
        skip = x
    return x, skip


class BertTransformerShard(TransformerShard):
    """BERT transformer shard based on `BertModel`."""

    def __init__(self, shard_config: ModuleShardConfig, model_name: str,
                 model_weights: Union[str, Mapping]):
        super().__init__(shard_config, model_name, model_weights)
        self.embeddings = None

        logger.debug(">>>> Model name: %s", model_name)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", self.model_weights)
            with np.load(self.model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        ## first Shard
        if self.shard_config.is_first:
            logger.debug(">>>> Load embeddings layer for the first shard")
            self.embeddings = BertEmbeddings(self.config)
            self.embeddings.eval()
            self._load_weights_first(weights)

        current_layer_idx = self.shard_config.layer_start

        ## partial model layer
        if self.shard_config.layer_start %4 != 1 or (self.shard_config.layer_start+3 > self.shard_config.layer_end):
            for i in range(self.shard_config.layer_start, min(self.shard_config.layer_end, math.ceil(self.shard_config.layer_start/4)*4)+1):
                logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
                layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1)
                layer.eval()
                self.first_ops.append(layer)
            current_layer_idx = min(self.shard_config.layer_end+1, math.ceil(self.shard_config.layer_start/4)*4+1)

        ## whole model layers
        while current_layer_idx + 3 <= self.shard_config.layer_end:
            logger.debug(">>>> Load the %d-th layer", math.ceil(current_layer_idx/4)-1)
            with torch.no_grad():
                layer = BertLayer(self.config)
            self._load_weights_layer(weights, math.ceil(current_layer_idx/4)-1, layer)
            layer.eval()
            self.model_layers.append(layer)
            current_layer_idx += 4

        ## partial model layer after whole model layers
        for i in range(current_layer_idx, self.shard_config.layer_end+1):
            logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
            layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1)
            layer.eval()
            self.last_ops.append(layer)

        ## last Shard
        if self.shard_config.is_last:
            logger.debug(">>>> Load pooler for the last shard")
            self.bertpooler = BertPooler(self.config)
            self.bertpooler.eval()
            self._load_weights_last(weights)

    def _build_kernel(self, weights, kernel_id, model_layer_id):
        layers = nn.ModuleList()
        if kernel_id == 1:
            layers.append(BertSelfAttention(self.config))
        elif kernel_id == 2:
            layers.append(BertSelfOutput(self.config))
        elif kernel_id == 3:
            layers.append(BertIntermediate(self.config))
        else:
            layers.append(BertOutput(self.config))
        self._load_weights_layer(weights, model_layer_id, layers, kernel_id=kernel_id)
        return layers

    @torch.no_grad()
    def _load_weights_first(self, weights):
        self.embeddings.position_ids.copy_(torch.from_numpy((weights["embeddings.position_ids"])))
        self.embeddings.word_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.word_embeddings.weight']))
        self.embeddings.position_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.position_embeddings.weight']))
        self.embeddings.token_type_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.token_type_embeddings.weight']))
        self.embeddings.LayerNorm.weight.copy_(torch.from_numpy(weights['embeddings.LayerNorm.weight']))
        self.embeddings.LayerNorm.bias.copy_(torch.from_numpy(weights['embeddings.LayerNorm.bias']))

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.bertpooler.dense.weight.copy_(torch.from_numpy(weights["pooler.dense.weight"]))
        self.bertpooler.dense.bias.copy_(torch.from_numpy(weights['pooler.dense.bias']))

    @torch.no_grad()
    def _load_weights_layer(self, weights, model_layer_id, model_layer, kernel_id=None):
        root = f"encoder.layer.{model_layer_id}."

        if kernel_id in (None, 1):
            lref = model_layer.attention.self if kernel_id is None else model_layer[0]
            lref.query.weight.copy_(torch.from_numpy(weights[root + "attention.self.query.weight"]))
            lref.key.weight.copy_(torch.from_numpy(weights[root + "attention.self.key.weight"]))
            lref.value.weight.copy_(torch.from_numpy(weights[root + "attention.self.value.weight"]))
            lref.query.bias.copy_(torch.from_numpy(weights[root + "attention.self.query.bias"]))
            lref.key.bias.copy_(torch.from_numpy(weights[root + "attention.self.key.bias"]))
            lref.value.bias.copy_(torch.from_numpy(weights[root + "attention.self.value.bias"]))

        if kernel_id in (None, 2):
            lref = model_layer.attention.output if kernel_id is None else model_layer[0]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "attention.output.dense.weight"]))
            lref.LayerNorm.weight.copy_(torch.from_numpy(weights[root + "attention.output.LayerNorm.weight"]))
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "attention.output.dense.bias"]))
            lref.LayerNorm.bias.copy_(torch.from_numpy(weights[root + "attention.output.LayerNorm.bias"]))

        if kernel_id in (None, 3):
            lref = model_layer.intermediate if kernel_id is None else model_layer[0]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "intermediate.dense.weight"]))
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "intermediate.dense.bias"]))

        if kernel_id in (None, 0):
            lref = model_layer.output if kernel_id is None else model_layer[0]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "output.dense.weight"]))
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "output.dense.bias"]))
            lref.LayerNorm.weight.copy_(torch.from_numpy(weights[root + "output.LayerNorm.weight"]))
            lref.LayerNorm.bias.copy_(torch.from_numpy(weights[root + "output.LayerNorm.bias"]))

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        x, skip = TransformerShard.parse_forward_data(data)

        if self.shard_config.is_first:
            x = self.embeddings(x)
            skip = x

        for i, op in enumerate(self.first_ops):
            x, skip = _forward_kernel(op, x, skip, (self.shard_config.layer_start+i)%4)

        for layer in self.model_layers:
            with torch.no_grad():
                x = layer(x)[0]
                skip = x

        for i, op in enumerate(self.last_ops):
            # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with _load_weights_layer()
            x, skip = _forward_kernel(op, x, skip, (i+1)%4)

        if self.shard_config.is_last:
            x = self.bertpooler(x)

        if self.shard_config.layer_end % 2 == 0:
            return x
        return x, skip

    @staticmethod
    def save_weights(model_name: str, model_file: str) -> None:
        """Save the model weights file."""
        model = BertModel.from_pretrained(model_name)
        state_dict = model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        np.savez(model_file, **weights)


class BertTransformerShardForSequenceClassification(TransformerShard):
    """BERT transformer shard based on `BertForSequenceClassification`."""

    def __init__(self, shard_config: ModuleShardConfig, model_name: str,
                 model_weights: Union[str, Mapping]):
        super().__init__(shard_config, model_name, model_weights)
        self.bert = None
        self.classifier = None

        logger.debug(">>>> Model name: %s", model_name)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", self.model_weights)
            with np.load(self.model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        # All stages do something with self.bert
        bert_weights = {}
        if weights is not None:
            # Extract weights for inner BERT model
            for key, val in weights.items():
                if key.startswith('bert.'):
                    bert_weights[key[len('bert.'):]] = val
        self.bert = BertTransformerShard(self.shard_config, self.model_name, bert_weights)

        if self.shard_config.is_last:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
            logger.debug(">>>> Load classifier for the last shard")
            if weights is not None:
                with torch.no_grad():
                    self.classifier.weight.copy_(torch.from_numpy(weights['classifier.weight']))
                    self.classifier.bias.copy_(torch.from_numpy(weights['classifier.bias']))

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        data = self.bert(data)
        if self.shard_config.is_last:
            data = self.classifier(data)
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str) -> None:
        """Save the model weights file."""
        model = BertForSequenceClassification.from_pretrained(model_name)
        state_dict = model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        np.savez(model_file, **weights)
