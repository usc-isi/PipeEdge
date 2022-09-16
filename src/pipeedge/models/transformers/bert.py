"""BERT transformers."""
from collections.abc import Mapping
import logging
import math
import time
from typing import Union
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput, BertLayer
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
    """BERT transformer shard."""

    def __init__(self, shard_config: ModuleShardConfig, model_name: str,
                 model_weights: Union[str, Mapping], load_weight: bool=True):
        super().__init__(shard_config, model_name, model_weights, load_weight)
        self.embeddings = None

        logger.debug(">>>> Model name: %s", model_name)
        if self.load_weight:
            if isinstance(model_weights, str):
                logger.debug(">>>> Load weight file: %s", self.model_weights)
                with np.load(self.model_weights) as weights:
                    self._make_layer(weights)
            else:
                self._make_layer(model_weights)
        else:
            self._make_layer(None)

        logger.info("======= Finish Build BertTransformerShard%d ==========", self.shard_config.stage)

    def _make_layer(self, weights):
        ## first Shard
        if self.shard_config.is_first:
            self.embeddings = BertEmbeddings(self.config)
            self.embeddings.eval()
            logger.debug(">>>> Load embeddings layer for the first shard")
            if self.load_weight:
                self._load_layer_weights(weights, 0, None, load_first = True, load_last=False, load_kernel = False, kernel_id=None)
                logger.debug(">>>> Load weights for embeddings layer")

        current_layer_idx = self.shard_config.layer_start

        ## first ununit part
        if self.shard_config.layer_start %4 != 1 or (self.shard_config.layer_start+3 > self.shard_config.layer_end):
            logger.debug(">>>> For the first model part, load weight is %s:", self.load_weight)
            for i in range(self.shard_config.layer_start, min(self.shard_config.layer_end, math.ceil(self.shard_config.layer_start/4)*4)+1):
                logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
                layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1, self.load_weight)
                layer.eval()
                self.first_ops.append(layer)
            current_layer_idx = min(self.shard_config.layer_end+1, math.ceil(self.shard_config.layer_start/4)*4+1)

        ## mid unit part, the whole vit_layer
        while current_layer_idx + 3 <= self.shard_config.layer_end:
            with torch.no_grad():
                layer = BertLayer(self.config)
            if self.load_weight:
                layer = self._load_layer_weights(weights, math.ceil(current_layer_idx/4)-1, layer)
            layer.eval()
            self.vit_layers.append(layer)
            logger.debug(">>>> Load the %d-th %s Layer, load weight is %s",
                          math.ceil(current_layer_idx/4)-1, self.model_name, self.load_weight)
            current_layer_idx += 4

        ## last unit part
        if self.shard_config.layer_end >= current_layer_idx:
            logger.debug(">>>> For the last model part, load weight is %s:", self.load_weight)
        for i in range(current_layer_idx, self.shard_config.layer_end+1):
            logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
            layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1, self.load_weight)
            if self.load_weight:
                layer = self._load_layer_weights(weights, math.ceil(i/4)-1, layer, False, False, True, i%4)
            layer.eval()
            self.last_ops.append(layer)

        ## last Shard
        if self.shard_config.is_last:
            self.bertpooler = BertPooler(self.config)
            self.bertpooler.eval()
            if self.load_weight:
                self._load_layer_weights(weights, 0, None, load_first = False, load_last=True, load_kernel = False, kernel_id=None)
                logger.debug(">>>> Load weights for layernorm and last shard")


        if self.load_weight:
            logger.debug(">>>> Finish load weights")
        else:
            logger.debug(">>>> Do NOT load weights")

    def _build_kernel(self, weights, kernel_id, vit_layer_id, load_weight=True):
        layers = nn.ModuleList()
        if kernel_id == 1:
            layers.append(BertSelfAttention(self.config))
        elif kernel_id == 2:
            layers.append(BertSelfOutput(self.config))
        elif kernel_id == 3:
            layers.append(BertIntermediate(self.config))
        else:
            layers.append(BertOutput(self.config))
        if load_weight:
            self._load_layer_weights(weights, vit_layer_id, layers, False, False, load_weight, kernel_id)
        return layers

    def _load_layer_weights(self, weights, id, transformer_layer, load_first = False, load_last=False, load_kernel = False, kernel_id=None):
        ROOT = f"encoder.layer.{id}."
        ATTENTION_Q = "attention.self.query."
        ATTENTION_K = "attention.self.key."
        ATTENTION_V = "attention.self.value."
        ATTENTION_OUT_DENSE = "attention.output.dense."
        ATTENTION_OUT_LAYERNORM = "attention.output.LayerNorm."
        INTERMEDIATE = "intermediate.dense."
        OUTPUT_DENSE = "output.dense."
        OUTPUT_LAYER = "output.LayerNorm."
        WEIGHT = "weight"
        BIAS = "bias"
        if load_first:
            with torch.no_grad():
                self.embeddings.position_ids.copy_(torch.from_numpy((weights["embeddings.position_ids"])))
                self.embeddings.word_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.word_embeddings.weight']))
                self.embeddings.position_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.position_embeddings.weight']))
                self.embeddings.token_type_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.token_type_embeddings.weight']))
                self.embeddings.LayerNorm.weight.copy_(torch.from_numpy(weights['embeddings.LayerNorm.weight']))
                self.embeddings.LayerNorm.bias.copy_(torch.from_numpy(weights['embeddings.LayerNorm.bias']))

        if load_last:
            with torch.no_grad():
                self.bertpooler.dense.weight.copy_(torch.from_numpy(weights["pooler.dense.weight"]))
                self.bertpooler.dense.bias.copy_(torch.from_numpy(weights['pooler.dense.bias']))


        if not load_first and not load_last:
            with torch.no_grad():
                if not load_kernel:
                    query_weight = torch.from_numpy(weights[ROOT + ATTENTION_Q + WEIGHT])
                    key_weight = torch.from_numpy(weights[ROOT + ATTENTION_K + WEIGHT])
                    value_weight = torch.from_numpy(weights[ROOT + ATTENTION_V + WEIGHT])
                    out_dense_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT_DENSE + WEIGHT])
                    output_layernorm_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT_LAYERNORM + WEIGHT])
                    intermediate_dense_weight = torch.from_numpy(weights[ROOT+INTERMEDIATE+WEIGHT])
                    dense_weight = torch.from_numpy(weights[ROOT + OUTPUT_DENSE + WEIGHT])
                    layernorm_weight = torch.from_numpy(weights[ROOT + OUTPUT_LAYER + WEIGHT])

                    query_bias = torch.from_numpy(weights[ROOT + ATTENTION_Q + BIAS])
                    key_bias = torch.from_numpy(weights[ROOT + ATTENTION_K + BIAS])
                    value_bias = torch.from_numpy(weights[ROOT + ATTENTION_V + BIAS])
                    out_dense_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT_DENSE + BIAS])
                    output_layernorm_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT_LAYERNORM + BIAS])
                    intermediate_dense_bias = torch.from_numpy(weights[ROOT+INTERMEDIATE+BIAS])
                    dense_bias = torch.from_numpy(weights[ROOT + OUTPUT_DENSE + BIAS])
                    layernorm_bias = torch.from_numpy(weights[ROOT + OUTPUT_LAYER + BIAS])

                    transformer_layer.attention.self.query.weight.copy_(query_weight)
                    transformer_layer.attention.self.key.weight.copy_(key_weight)
                    transformer_layer.attention.self.value.weight.copy_(value_weight)
                    transformer_layer.attention.output.dense.weight.copy_(out_dense_weight)
                    transformer_layer.attention.output.LayerNorm.weight.copy_(output_layernorm_weight)

                    transformer_layer.attention.self.query.bias.copy_(query_bias)
                    transformer_layer.attention.self.key.bias.copy_(key_bias)
                    transformer_layer.attention.self.value.bias.copy_(value_bias)
                    transformer_layer.attention.output.dense.bias.copy_(out_dense_bias)
                    transformer_layer.attention.output.LayerNorm.bias.copy_(output_layernorm_bias)

                    transformer_layer.intermediate.dense.weight.copy_(intermediate_dense_weight)
                    transformer_layer.intermediate.dense.bias.copy_(intermediate_dense_bias )

                    transformer_layer.output.dense.weight.copy_(dense_weight)
                    transformer_layer.output.dense.bias.copy_(dense_bias)
                    transformer_layer.output.LayerNorm.weight.copy_(layernorm_weight)
                    transformer_layer.output.LayerNorm.bias.copy_(layernorm_bias)
                    logger.debug("memory %d MB", self.process.memory_info().rss // 1000000)

                elif kernel_id == 1:

                    query_weight = torch.from_numpy(weights[ROOT + ATTENTION_Q + WEIGHT])
                    key_weight = torch.from_numpy(weights[ROOT + ATTENTION_K + WEIGHT])
                    value_weight = torch.from_numpy(weights[ROOT + ATTENTION_V + WEIGHT])
                    query_bias = torch.from_numpy(weights[ROOT + ATTENTION_Q + BIAS])
                    key_bias = torch.from_numpy(weights[ROOT + ATTENTION_K + BIAS])
                    value_bias = torch.from_numpy(weights[ROOT + ATTENTION_V + BIAS])
                    transformer_layer[0].query.weight.copy_(query_weight)
                    transformer_layer[0].key.weight.copy_(key_weight)
                    transformer_layer[0].value.weight.copy_(value_weight)
                    transformer_layer[0].query.bias.copy_(query_bias)
                    transformer_layer[0].key.bias.copy_(key_bias)
                    transformer_layer[0].value.bias.copy_(value_bias)

                elif kernel_id == 2:
                    out_dense_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT_DENSE + WEIGHT])
                    output_layernorm_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT_LAYERNORM + WEIGHT])
                    out_dense_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT_DENSE + BIAS])
                    output_layernorm_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT_LAYERNORM + BIAS])
                    transformer_layer[0].dense.weight.copy_(out_dense_weight)
                    transformer_layer[0].LayerNorm.weight.copy_(output_layernorm_weight)
                    transformer_layer[0].dense.bias.copy_(out_dense_bias)
                    transformer_layer[0].LayerNorm.bias.copy_(output_layernorm_bias)
                elif kernel_id == 3:
                    intermediate_dense_weight = torch.from_numpy(weights[ROOT+INTERMEDIATE+WEIGHT])
                    intermediate_dense_bias = torch.from_numpy(weights[ROOT+INTERMEDIATE+BIAS])
                    transformer_layer[0].dense.weight.copy_(intermediate_dense_weight)
                    transformer_layer[0].dense.bias.copy_(intermediate_dense_bias )
                elif kernel_id == 0:
                    dense_weight = torch.from_numpy(weights[ROOT + OUTPUT_DENSE + WEIGHT])
                    layernorm_weight = torch.from_numpy(weights[ROOT + OUTPUT_LAYER + WEIGHT])
                    dense_bias = torch.from_numpy(weights[ROOT + OUTPUT_DENSE + BIAS])
                    layernorm_bias = torch.from_numpy(weights[ROOT + OUTPUT_LAYER + BIAS])

                    transformer_layer[0].dense.weight.copy_(dense_weight)
                    transformer_layer[0].dense.bias.copy_(dense_bias)
                    transformer_layer[0].LayerNorm.weight.copy_(layernorm_weight)
                    transformer_layer[0].LayerNorm.bias.copy_(layernorm_bias)

        return transformer_layer

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        start = time.time()
        x, skip = TransformerShard.parse_forward_data(data)

        if self.shard_config.is_first:
            x = self.embeddings(x)
            skip = x

        for i, op in enumerate(self.first_ops):
            x, skip = _forward_kernel(op, x, skip, (self.shard_config.layer_start+i)%4)

        for layer in self.vit_layers:
            with torch.no_grad():
                x = layer(x)[0]
                skip = x

        for i, op in enumerate(self.last_ops):
            # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with _load_layer_weights()
            x, skip = _forward_kernel(op, x, skip, (i+1)%4)

        if self.shard_config.is_last:
            x = self.bertpooler(x)
        end = time.time()

        logger.info("Shard%d: computed microbatch in: %f sec", self.shard_config.stage, end - start)
        logger.info("Shard%d: memory: %d MB", self.shard_config.stage, self.process.memory_info().rss / 1000000)

        if self.shard_config.layer_end % 2 == 0:
            return x
        return x, skip

    @staticmethod
    def save_weights(model_name: str, model_file: str) -> None:
        """Save the model weights file."""
        model = AutoModel.from_pretrained(model_name)
        state_dict = model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        np.savez(model_file, **weights)


class BertTransformerShardForSequenceClassification(TransformerShard):
    """BERT transformer shard for sequence classification."""

    def __init__(self, shard_config: ModuleShardConfig, model_name: str,
                 model_weights: Union[str, Mapping], load_weight: bool=True):
        super().__init__(shard_config, model_name, model_weights, load_weight)
        self.bert = None
        self.classifier = None

        logger.debug(">>>> Model name: %s", model_name)
        if self.load_weight:
            if isinstance(model_weights, str):
                logger.debug(">>>> Load weight file: %s", self.model_weights)
                with np.load(self.model_weights) as weights:
                    self._make_layer(weights)
            else:
                self._make_layer(model_weights)
        else:
            self._make_layer(None)
        logger.info(
            "======= Finish Build BertTransformerShardForSequenceClassification%d ==========",
            self.shard_config.stage)

    def _make_layer(self, weights):
        # All stages do something with self.bert
        bert_weights = {}
        if self.load_weight:
            # Extract weights for inner BERT model
            for key, val in weights.items():
                if key.startswith('bert.'):
                    bert_weights[key[len('bert.'):]] = val
        self.bert = BertTransformerShard(self.shard_config, self.model_name, bert_weights,
                                         self.load_weight)

        if self.shard_config.is_last:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
            if self.load_weight:
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
