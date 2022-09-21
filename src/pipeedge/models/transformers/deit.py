"""DeiT Transformers."""
from collections.abc import Mapping
import logging
import math
from typing import Optional, Union
import numpy as np
import torch
from torch import nn
from transformers import DeiTConfig
from transformers.models.deit.modeling_deit import DeiTEmbeddings
from transformers.models.vit.modeling_vit import (
    ViTIntermediate, ViTLayer, ViTOutput, ViTSelfAttention, ViTSelfOutput
)
from .. import ModuleShardConfig
from . import TransformerShard, TransformerShardData


logger = logging.getLogger(__name__)

_HUB_MODEL_NAMES = {
    'facebook/deit-base-distilled-patch16-224': 'deit_base_distilled_patch16_224',
    'facebook/deit-small-distilled-patch16-224': 'deit_small_distilled_patch16_224',
    'facebook/deit-tiny-distilled-patch16-224': 'deit_tiny_distilled_patch16_224',
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


class DeiTModelShard(TransformerShard):
    """Module shard based on `DeiTModel`."""

    def __init__(self, config: DeiTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config, model_weights)
        self.embeddings = None
        self.layernorm = None

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
            self.embeddings = DeiTEmbeddings(self.config)
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
            logger.debug(">>>> Load layernorm for the last shard")
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
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
        self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["pos_embed"])))
        self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(weights["patch_embed.proj.weight"]))
        self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["patch_embed.proj.bias"]))

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.layernorm.weight.copy_(torch.from_numpy(weights["norm.weight"]))
        self.layernorm.bias.copy_(torch.from_numpy(weights["norm.bias"]))

    @torch.no_grad()
    def _load_weights_layer(self, weights, model_layer_id, model_layer, kernel_id=None):
        root = f"blocks.{model_layer_id}."
        embed_dim = self.config.hidden_size

        if kernel_id in (None, 1):
            lref = model_layer.layernorm_before if kernel_id is None else model_layer[0]
            lref.weight.copy_(torch.from_numpy(weights[root + "norm1.weight"]))
            lref.bias.copy_(torch.from_numpy(weights[root + "norm1.bias"]))
            lref = model_layer.attention.attention if kernel_id is None else model_layer[1]
            qkv_weight = weights[root + "attn.qkv.weight"]
            lref.query.weight.copy_(torch.from_numpy(qkv_weight[0:embed_dim,:]))
            lref.key.weight.copy_(torch.from_numpy(qkv_weight[embed_dim:embed_dim*2,:]))
            lref.value.weight.copy_(torch.from_numpy(qkv_weight[embed_dim*2:embed_dim*3,:]))
            qkv_bias = weights[root + "attn.qkv.bias"]
            lref.query.bias.copy_(torch.from_numpy(qkv_bias[0:embed_dim,]))
            lref.key.bias.copy_(torch.from_numpy(qkv_bias[embed_dim:embed_dim*2]))
            lref.value.bias.copy_(torch.from_numpy(qkv_bias[embed_dim*2:embed_dim*3]))

        if kernel_id in (None, 2):
            lref = model_layer.attention.output if kernel_id is None else model_layer[0]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "attn.proj.weight"]))
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "attn.proj.bias"]))

        if kernel_id in (None, 3):
            lref = model_layer.layernorm_after if kernel_id is None else model_layer[0]
            lref.weight.copy_(torch.from_numpy(weights[root + "norm2.weight"]))
            lref.bias.copy_(torch.from_numpy(weights[root + "norm2.bias"]))
            lref = model_layer.intermediate if kernel_id is None else model_layer[1]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "mlp.fc1.weight"]))
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "mlp.fc1.bias"]))

        if kernel_id in (None, 0):
            lref = model_layer.output if kernel_id is None else model_layer[0]
            lref.dense.weight.copy_(torch.from_numpy(weights[root + "mlp.fc2.weight"]))
            lref.dense.bias.copy_(torch.from_numpy(weights[root + "mlp.fc2.bias"]))

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

        if self.shard_config.layer_end % 2 == 0:
            return x
        return x, skip

    # NOTE: repo has a dependency on the timm package, which isn't an automatic torch dependency
    @staticmethod
    def save_weights(model_name: str, model_file: str, hub_repo: str='facebookresearch/deit:main',
                     hub_model_name: Optional[str]=None) -> None:
        """Save the model weights file."""
        if hub_model_name is None:
            if model_name in _HUB_MODEL_NAMES:
                hub_model_name = _HUB_MODEL_NAMES[model_name]
                logger.debug("Mapping model name to torch hub equivalent: %s: %s", model_name,
                             hub_model_name)
            else:
                hub_model_name = model_name
        model = torch.hub.load(hub_repo, hub_model_name, pretrained=True)
        state_dict = model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        np.savez(model_file, **weights)


class DeiTShardForImageClassification(TransformerShard):
    """Module shard based on `DeiTForImageClassification`."""

    def __init__(self, config: DeiTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config, model_weights)
        self.deit = None
        self.classifier = None

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", self.model_weights)
            with np.load(self.model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        ## all shards use the inner DeiT model
        self.deit = DeiTModelShard(self.config, self.shard_config, weights)

        ## last Shard
        if self.shard_config.is_last:
            logger.debug(">>>> Load classifier for the last shard")
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.classifier.weight.copy_(torch.from_numpy(weights["head.weight"]))
        self.classifier.bias.copy_(torch.from_numpy(weights["head.bias"]))

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        data = self.deit(data)
        if self.shard_config.is_last:
            data = self.classifier(data[:, 0, :])
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str, hub_repo: str='facebookresearch/deit:main',
                     hub_model_name: Optional[str]=None) -> None:
        """Save the model weights file."""
        DeiTModelShard.save_weights(model_name, model_file, hub_repo=hub_repo,
                                    hub_model_name=hub_model_name)
