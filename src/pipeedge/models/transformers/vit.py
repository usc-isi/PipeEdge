"""ViT Transformers."""
from collections.abc import Mapping
import logging
import math
import os
import time
from typing import Optional, Union
import numpy as np
import requests
import torch
from torch import nn
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTSelfAttention, ViTSelfOutput, ViTIntermediate, ViTOutput
from .. import ModuleShardConfig
from . import TransformerShard, TransformerShardData


logger = logging.getLogger(__name__)

_WEIGHTS_URLS = {
    'google/vit-base-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz',
    'google/vit-large-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz',
    'google/vit-huge-patch14-224-in21k': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
}


def _forward_kernel(layer, x, skip, kernel_id):
    if kernel_id == 1:
        x = layer[0](x)
        x = layer[1](x)[0]
    elif kernel_id == 2:
        x = layer[0](x, skip)
        x += skip
        skip = x
    elif kernel_id == 3:
        x = layer[0](x)
        x = layer[1](x)
    else:
        x = layer[0](x, skip)
        skip = x
    return x, skip


class ViTTransformerShard(TransformerShard):
    """ViT transformer shard."""

    def __init__(self, shard_config: ModuleShardConfig, model_name: str,
                 model_weights: Union[str, Mapping], load_weight: bool=True):
        super().__init__(shard_config, model_name, model_weights, load_weight)
        if self.model_name == 'google/vit-huge-patch14-224-in21k':
            # This ViT-Huge model doesn't include classification, so we have to set this ourselves
            # NOTE: not setting 'id2label' or 'label2id'
            self.config.num_labels = 21843
        self.embeddings = None
        self.layernorm = None
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

        logger.info("======= Finish Build ViTTransformerShard%d ==========", self.shard_config.stage)

    def _make_layer(self, weights):
        ## first Shard
        if self.shard_config.is_first:
            self.embeddings = ViTEmbeddings(self.config)
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
                self.first_ops.append(layer)
            current_layer_idx = min(self.shard_config.layer_end+1, math.ceil(self.shard_config.layer_start/4)*4+1)

        ## mid unit part, the whole vit_layer
        while current_layer_idx + 3 <= self.shard_config.layer_end:
            layer = ViTLayer(self.config)
            if self.load_weight:
                layer = self._load_layer_weights(weights, math.ceil(current_layer_idx/4)-1, layer)
            self.vit_layers.append(layer)
            logger.debug(">>>> Load the %d-th ViT Layer, load weight is %s",
                          math.ceil(current_layer_idx/4)-1, self.load_weight)
            current_layer_idx += 4

        ## last unit part
        if self.shard_config.layer_end >= current_layer_idx:
            logger.debug(">>>> For the last model part, load weight is %s:", self.load_weight)
        for i in range(current_layer_idx, self.shard_config.layer_end+1):
            logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
            layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1, self.load_weight)
            if self.load_weight:
                layer = self._load_layer_weights(weights, math.ceil(i/4)-1, layer, False, False, True, i%4)
            self.last_ops.append(layer)

        ## last Shard
        if self.shard_config.is_last:
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            logger.debug(">>>> Load layernorm for the last shard")
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()
            logger.debug(">>>> Load classifier for the last shard")
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
            layers.append(nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps))
            layers.append(ViTSelfAttention(self.config))
        elif kernel_id == 2:
            layers.append(ViTSelfOutput(self.config))
        elif kernel_id == 3:
            layers.append(nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps))
            layers.append( ViTIntermediate(self.config))
        else:
            layers.append(ViTOutput(self.config))
        if load_weight:
            self._load_layer_weights(weights, vit_layer_id, layers, False, False, load_weight, kernel_id)
        return layers

    def _load_layer_weights(self, weights, id, transformer_layer, load_first = False, load_last=False, load_kernel = False, kernel_id=None):
        ROOT = f"Transformer/encoderblock_{id}"
        ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
        ATTENTION_K = "MultiHeadDotProductAttention_1/key"
        ATTENTION_V = "MultiHeadDotProductAttention_1/value"
        ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
        FC_0 = "MlpBlock_3/Dense_0"
        FC_1 = "MlpBlock_3/Dense_1"
        ATTENTION_NORM = "LayerNorm_0"
        MLP_NORM = "LayerNorm_2"
        hidden_size = self.config.hidden_size
        if load_first:
            with torch.no_grad():
                self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
                conv_weight = weights["embedding/kernel"]
                # O, I, J, K = conv_weight.shape
                # logger.debug(f"conv_shape is {O, I, J, K}, pe weight shape is {self.embeddings.patch_embeddings.projection.weight.shape}")
                # conv_weight = conv_weight.reshape(K,J,O,I)
                conv_weight = conv_weight.transpose([3, 2, 0, 1])
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))
                # logger.debug(">>>> Load embedding for the first shard")

        if load_last:
            with torch.no_grad():
                self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
                self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))
                # head_kernel = np.transpose(weights["head/kernel"])
                # logger.debug(f"classifier weight is {self.classifier.weight.shape}, head kernel weight shape is {head_kernel.shape}")
                self.classifier.weight.copy_(torch.from_numpy(np.transpose(weights["head/kernel"])))
                self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))
                # logger.debug(">>>> Load Layernorm, classifier for the last shard")


        if not load_first and not load_last:
            with torch.no_grad():
                if not load_kernel:

                    query_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")]).view(hidden_size, hidden_size).t()
                    logger.debug("query weight shape is %s", query_weight.shape)
                    key_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_K, "kernel")]).view(hidden_size, hidden_size).t()
                    value_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_V, "kernel")]).view(hidden_size, hidden_size).t()
                    out_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")]).view(hidden_size, hidden_size).t()

                    query_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "bias")]).view(-1)
                    key_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_K, "bias")]).view(-1)
                    value_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_V, "bias")]).view(-1)
                    out_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "bias")]).view(-1)

                    transformer_layer.attention.attention.query.weight.copy_(query_weight)
                    transformer_layer.attention.attention.key.weight.copy_(key_weight)
                    transformer_layer.attention.attention.value.weight.copy_(value_weight)
                    transformer_layer.attention.output.dense.weight.copy_(out_weight)

                    transformer_layer.attention.attention.query.bias.copy_(query_bias)
                    transformer_layer.attention.attention.key.bias.copy_(key_bias)
                    transformer_layer.attention.attention.value.bias.copy_(value_bias)
                    transformer_layer.attention.output.dense.bias.copy_(out_bias)

                    mlp_weight_0 = torch.from_numpy(weights[os.path.join(ROOT, FC_0, "kernel")]).t()
                    mlp_weight_1 = torch.from_numpy(weights[os.path.join(ROOT, FC_1, "kernel")]).t()
                    mlp_bias_0 = torch.from_numpy(weights[os.path.join(ROOT, FC_0, "bias")]).t()
                    mlp_bias_1 = torch.from_numpy(weights[os.path.join(ROOT, FC_1, "bias")]).t()

                    transformer_layer.intermediate.dense.weight.copy_(mlp_weight_0)
                    transformer_layer.intermediate.dense.bias.copy_(mlp_bias_0)
                    transformer_layer.output.dense.weight.copy_(mlp_weight_1)
                    transformer_layer.output.dense.bias.copy_(mlp_bias_1)

                    transformer_layer.layernorm_before.weight.copy_(torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_NORM, "scale")]))
                    transformer_layer.layernorm_before.bias.copy_(torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_NORM, "bias")]))
                    transformer_layer.layernorm_after.weight.copy_(torch.from_numpy(weights[os.path.join(ROOT, MLP_NORM, "scale")]))
                    transformer_layer.layernorm_after.bias.copy_(torch.from_numpy(weights[os.path.join(ROOT, MLP_NORM, "bias")]))
                    logger.debug("memory %d MB", self.process.memory_info().rss // 1000000)


                elif kernel_id == 1:

                    query_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")]).view(hidden_size, hidden_size).t()
                    key_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_K, "kernel")]).view(hidden_size, hidden_size).t()
                    value_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_V, "kernel")]).view(hidden_size, hidden_size).t()

                    query_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "bias")]).view(-1)
                    key_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_K, "bias")]).view(-1)
                    value_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_V, "bias")]).view(-1)

                    transformer_layer[0].weight.copy_(torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_NORM, "scale")]))
                    transformer_layer[0].bias.copy_(torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_NORM, "bias")]))
                    transformer_layer[1].query.weight.copy_(query_weight)
                    transformer_layer[1].key.weight.copy_(key_weight)
                    transformer_layer[1].value.weight.copy_(value_weight)

                    transformer_layer[1].query.bias.copy_(query_bias)
                    transformer_layer[1].key.bias.copy_(key_bias)
                    transformer_layer[1].value.bias.copy_(value_bias)

                elif kernel_id == 2:
                    out_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")]).view(hidden_size, hidden_size).t()
                    out_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "bias")]).view(-1)
                    transformer_layer[0].dense.weight.copy_(out_weight)
                    transformer_layer[0].dense.bias.copy_(out_bias)
                elif kernel_id == 3:
                    transformer_layer[0].weight.copy_(torch.from_numpy(weights[os.path.join(ROOT, MLP_NORM, "scale")]))
                    transformer_layer[0].bias.copy_(torch.from_numpy(weights[os.path.join(ROOT, MLP_NORM, "bias")]))
                    mlp_weight_0 = torch.from_numpy(weights[os.path.join(ROOT, FC_0, "kernel")]).t()
                    mlp_bias_0 = torch.from_numpy(weights[os.path.join(ROOT, FC_0, "bias")]).t()
                    transformer_layer[1].dense.weight.copy_(mlp_weight_0)
                    transformer_layer[1].dense.bias.copy_(mlp_bias_0)
                elif kernel_id == 0:
                    mlp_weight_1 = torch.from_numpy(weights[os.path.join(ROOT, FC_1, "kernel")]).t()
                    mlp_bias_1 = torch.from_numpy(weights[os.path.join(ROOT, FC_1, "bias")]).t()
                    transformer_layer[0].dense.weight.copy_(mlp_weight_1)
                    transformer_layer[0].dense.bias.copy_(mlp_bias_1)

        return transformer_layer

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        logger.debug("Start memory %d MB", self.process.memory_info().rss / 1000000)
        start = time.time()
        x, skip = TransformerShard.parse_forward_data(data)

        if self.shard_config.is_first:
            x = self.embeddings(x)
            skip = x

        for i, op in enumerate(self.first_ops):
            x, skip = _forward_kernel(op, x, skip, (self.shard_config.layer_start+i)%4)

        for i, layer in enumerate(self.vit_layers):
            logger.debug("Before %d: %d MB", i, self.process.memory_info().rss / 1000000)
            x = layer(x)[0]
            logger.debug("After %d: %d MB", i, self.process.memory_info().rss / 1000000)
            skip = x
        logger.debug("vit-layer memory %d MB", self.process.memory_info().rss / 1000000)

        for i, op in enumerate(self.last_ops):
            # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with _load_layer_weights()
            x, skip = _forward_kernel(op, x, skip, (i+1)%4)

        if self.shard_config.is_last:
            x = self.layernorm(x)
            x = self.classifier(x[:, 0, :])
        end = time.time()

        logger.info("Shard%d: computed microbatch in: %f sec", self.shard_config.stage, end - start)
        logger.info("Shard%d: memory: %d MB", self.shard_config.stage, self.process.memory_info().rss / 1000000)

        if self.shard_config.layer_end % 2 == 0:
            return x
        return x, skip

    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None) -> None:
        """Save the model weights file."""
        if url is None:
            url = _WEIGHTS_URLS[model_name]
        logger.info('Downloading model: %s: %s', model_name, url)
        req = requests.get(url, stream=True)
        req.raise_for_status()
        with open(model_file, 'wb') as file:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())
