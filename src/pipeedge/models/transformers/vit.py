"""ViT Transformers."""
from collections.abc import Mapping
import logging
import math
import os
from typing import Optional, Union
import numpy as np
import requests
import torch
from torch import nn
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings, ViTIntermediate, ViTLayer, ViTOutput, ViTSelfAttention, ViTSelfOutput
)
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
    """ViT transformer shard based on `ViTModel`."""

    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config, model_weights)
        if self.config.name_or_path == 'google/vit-huge-patch14-224-in21k':
            # This ViT-Huge model doesn't include classification, so we have to set this ourselves
            # NOTE: not setting 'id2label' or 'label2id'
            self.config.num_labels = 21843
        self.embeddings = None
        self.layernorm = None
        self.classifier = None

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
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
            self.embeddings = ViTEmbeddings(self.config)
            self._load_weights_first(weights)

        current_layer_idx = self.shard_config.layer_start

        ## partial model layer
        if self.shard_config.layer_start %4 != 1 or (self.shard_config.layer_start+3 > self.shard_config.layer_end):
            for i in range(self.shard_config.layer_start, min(self.shard_config.layer_end, math.ceil(self.shard_config.layer_start/4)*4)+1):
                logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
                layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1)
                self.first_ops.append(layer)
            current_layer_idx = min(self.shard_config.layer_end+1, math.ceil(self.shard_config.layer_start/4)*4+1)

        ## whole model layers
        while current_layer_idx + 3 <= self.shard_config.layer_end:
            logger.debug(">>>> Load the %d-th layer", math.ceil(current_layer_idx/4)-1)
            layer = ViTLayer(self.config)
            self._load_weights_layer(weights, math.ceil(current_layer_idx/4)-1, layer)
            self.model_layers.append(layer)
            current_layer_idx += 4

        ## partial model layer after whole model layers
        for i in range(current_layer_idx, self.shard_config.layer_end+1):
            logger.debug("    Load the %d-th operation for %d-th layer", i%4, math.ceil(i/4)-1)
            layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1)
            self.last_ops.append(layer)

        ## last Shard
        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm and classifier for the last shard")
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()
            self._load_weights_last(weights)

    def _build_kernel(self, weights, kernel_id, model_layer_id):
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
        self._load_weights_layer(weights, model_layer_id, layers, kernel_id=kernel_id)
        return layers

    @torch.no_grad()
    def _load_weights_first(self, weights):
        self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
        conv_weight = weights["embedding/kernel"]
        # O, I, J, K = conv_weight.shape
        # conv_weight = conv_weight.reshape(K,J,O,I)
        conv_weight = conv_weight.transpose([3, 2, 0, 1])
        self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
        self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
        self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))
        self.classifier.weight.copy_(torch.from_numpy(np.transpose(weights["head/kernel"])))
        self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))

    @torch.no_grad()
    def _load_weights_layer(self, weights, model_layer_id, model_layer, kernel_id=None):
        root = f"Transformer/encoderblock_{model_layer_id}/"
        hidden_size = self.config.hidden_size

        if kernel_id in (None, 1):
            lref = model_layer.layernorm_before if kernel_id is None else model_layer[0]
            lref.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_0/scale"]))
            lref.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_0/bias"]))
            lref = model_layer.attention.attention if kernel_id is None else model_layer[1]
            lref.query.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/kernel"]).view(hidden_size, hidden_size).t())
            lref.key.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/kernel"]).view(hidden_size, hidden_size).t())
            lref.value.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/kernel"]).view(hidden_size, hidden_size).t())
            lref.query.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/bias"]).view(-1))
            lref.key.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/bias"]).view(-1))
            lref.value.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/bias"]).view(-1))

        if kernel_id in (None, 2):
            lref = model_layer.attention.output if kernel_id is None else model_layer[0]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/kernel"]).view(hidden_size, hidden_size).t())
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/bias"]).view(-1))

        if kernel_id in (None, 3):
            lref = model_layer.layernorm_after if kernel_id is None else model_layer[0]
            lref.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_2/scale"]))
            lref.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_2/bias"]))
            lref = model_layer.intermediate if kernel_id is None else model_layer[1]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/kernel"]).t())
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/bias"]).t())

        if kernel_id in (None, 0):
            lref = model_layer.output if kernel_id is None else model_layer[0]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/kernel"]).t())
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/bias"]).t())

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        x, skip = TransformerShard.parse_forward_data(data)

        if self.shard_config.is_first:
            x = self.embeddings(x)
            skip = x

        for i, op in enumerate(self.first_ops):
            x, skip = _forward_kernel(op, x, skip, (self.shard_config.layer_start+i)%4)

        for i, layer in enumerate(self.model_layers):
            x = layer(x)[0]
            skip = x

        for i, op in enumerate(self.last_ops):
            # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with _load_weights_layer()
            x, skip = _forward_kernel(op, x, skip, (i+1)%4)

        if self.shard_config.is_last:
            x = self.layernorm(x)
            x = self.classifier(x[:, 0, :])

        if self.shard_config.layer_end % 2 == 0:
            return x
        return x, skip

    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        """Save the model weights file."""
        if url is None:
            url = _WEIGHTS_URLS[model_name]
        logger.info('Downloading model: %s: %s', model_name, url)
        req = requests.get(url, stream=True, timeout=timeout_sec)
        req.raise_for_status()
        with open(model_file, 'wb') as file:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())
