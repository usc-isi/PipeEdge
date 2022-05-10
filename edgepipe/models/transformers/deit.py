"""DeiT Transformers."""
import logging
import math
import time
import numpy as np
import torch
from torch import nn
from transformers.models.deit.modeling_deit import DeiTEmbeddings
from transformers.models.vit.modeling_vit import ViTLayer, ViTSelfAttention, ViTSelfOutput, ViTIntermediate, ViTOutput
from . import TransformerShard


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


class DeiTTransformerShard(TransformerShard):
    """DeiT transformer shard."""

    def __init__(self, stage, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight=True):
        super().__init__(stage, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight)
        self.embeddings = None
        self.layernorm = None
        self.classifier = None

        logging.debug(">>>> Model name: %s", model_name)
        if self.load_weight:
            logging.debug(">>>> Load weight file: %s", self.weights_file_name)
            with np.load(self.weights_file_name) as weights:
                self._make_layer(weights)
        else:
            self._make_layer(None)

        logging.info("======= Finish Build DeiTTransformerShard%d ==========", self.stage)

    def _make_layer(self, weights):
        ## first Shard
        if self.is_first:
            self.embeddings = DeiTEmbeddings(self.config)
            logging.debug(">>>> Load embeddings layer for the first shard")
            if self.load_weight:
                self._load_layer_weights(weights, 0, None, load_first = True, load_last=False, load_kernel = False, kernel_id=None)
                logging.debug(">>>> Load weights for embeddings layer")

        current_layer_idx = self.start_layer

        ## first ununit part
        if self.start_layer %4 != 1 or (self.start_layer+3 > self.end_layer):
            logging.debug(">>>> For the first model part, load weight is %s:", self.load_weight)
            for i in range(self.start_layer, min(self.end_layer, math.ceil(self.start_layer/4)*4)+1):
                logging.debug("    Load the %d-th operation (%s) for %d-th vit layer",
                              i%4, self.operators_list[(i-1)%4], math.ceil(i/4)-1)
                layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1, self.load_weight)
                self.first_ops.append(layer)
            current_layer_idx = min(self.end_layer+1, math.ceil(self.start_layer/4)*4+1)

        ## mid unit part, the whole vit_layer
        while current_layer_idx + 3 <= self.end_layer:
            layer = ViTLayer(self.config)
            if self.load_weight:
                layer = self._load_layer_weights(weights, math.ceil(current_layer_idx/4)-1, layer)
            self.vit_layers.append(layer)
            logging.debug(">>>> Load the %d-th ViT Layer, load weight is %s",
                          math.ceil(current_layer_idx/4)-1, self.load_weight)
            current_layer_idx += 4

        ## last unit part
        if self.end_layer >= current_layer_idx:
            logging.debug(">>>> For the last model part, load weight is %s:", self.load_weight)
        for i in range(current_layer_idx, self.end_layer+1):
            logging.debug("    Load the %d-th operation (%s) for %d-th vit layer",
                          i%4, self.operators_list[(i-1)%4], math.ceil(i/4)-1)
            layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1, self.load_weight)
            if self.load_weight:
                layer = self._load_layer_weights(weights, math.ceil(i/4)-1, layer, False, False, True, i%4)
            self.last_ops.append(layer)

        ## last Shard
        if self.is_last:
            num_label = self.config.num_labels
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            logging.debug(">>>> Load layernorm for the last shard")
            if self.model_name == 'google/vit-huge-patch14-224-in21k':
                num_label = 21843
            self.classifier = nn.Linear(self.config.hidden_size, num_label) if self.config.num_labels > 0 else nn.Identity()
            logging.debug(">>>> Load classifier for the last shard")
            if self.load_weight:
                self._load_layer_weights(weights, 0, None, load_first = False, load_last=True, load_kernel = False, kernel_id=None)
                logging.debug(">>>> Load weights for layernorm and last shard")


        if self.load_weight:
            logging.debug(">>>> Finish load weights")
        else:
            logging.debug(">>>> Do NOT load weights")

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
        ROOT = f"blocks.{id}."
        ATTENTION_QKV = "attn.qkv."
        ATTENTION_OUT = "attn.proj."
        FC_0 = "mlp.fc1."
        FC_1 = "mlp.fc2."
        ATTENTION_NORM = "norm1."
        MLP_NORM = "norm2."
        embed_dim = self.config.hidden_size
        if load_first:
            with torch.no_grad():
                self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["pos_embed"])))
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(weights["patch_embed.proj.weight"]))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["patch_embed.proj.bias"]))
                # logging.debug(">>>> Load embedding for the first shard")

        if load_last:
            with torch.no_grad():
                self.layernorm.weight.copy_(torch.from_numpy(weights["norm.weight"]))
                self.layernorm.bias.copy_(torch.from_numpy(weights["norm.bias"]))
                self.classifier.weight.copy_(torch.from_numpy(weights["head.weight"]))
                self.classifier.bias.copy_(torch.from_numpy(weights["head.bias"]))
                # logging.debug(">>>> Load Layernorm, classifier for the last shard")

        if not load_first and not load_last:
            with torch.no_grad():
                if not load_kernel:
                    qkv_f = str(ROOT + ATTENTION_QKV + "weight")
                    qkv_weight = weights[qkv_f]
                    query_weight = torch.from_numpy(qkv_weight[0:embed_dim,:])
                    key_weight = torch.from_numpy(qkv_weight[embed_dim:embed_dim*2,:])
                    value_weight = torch.from_numpy(qkv_weight[embed_dim*2:embed_dim*3,:])
                    out_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT + "weight"])

                    qkv_b = str(ROOT + ATTENTION_QKV + "bias")
                    qkv_bias = weights[qkv_b]
                    query_bias = torch.from_numpy(qkv_bias[0:embed_dim])
                    key_bias = torch.from_numpy(qkv_bias[embed_dim:embed_dim*2])
                    value_bias= torch.from_numpy(qkv_bias[embed_dim*2:embed_dim*3])
                    out_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT + "bias"])

                    transformer_layer.attention.attention.query.weight.copy_(query_weight)
                    transformer_layer.attention.attention.key.weight.copy_(key_weight)
                    transformer_layer.attention.attention.value.weight.copy_(value_weight)
                    transformer_layer.attention.output.dense.weight.copy_(out_weight)

                    transformer_layer.attention.attention.query.bias.copy_(query_bias)
                    transformer_layer.attention.attention.key.bias.copy_(key_bias)
                    transformer_layer.attention.attention.value.bias.copy_(value_bias)
                    transformer_layer.attention.output.dense.bias.copy_(out_bias)

                    mlp_weight_0 = torch.from_numpy(weights[ROOT + FC_0 + "weight"])
                    mlp_weight_1 = torch.from_numpy(weights[ROOT + FC_1 + "weight"])
                    mlp_bias_0 = torch.from_numpy(weights[ROOT + FC_0 + "bias"])
                    mlp_bias_1 = torch.from_numpy(weights[ROOT + FC_1 + "bias"])

                    transformer_layer.intermediate.dense.weight.copy_(mlp_weight_0)
                    transformer_layer.intermediate.dense.bias.copy_(mlp_bias_0)
                    transformer_layer.output.dense.weight.copy_(mlp_weight_1)
                    transformer_layer.output.dense.bias.copy_(mlp_bias_1)

                    transformer_layer.layernorm_before.weight.copy_(torch.from_numpy(weights[ROOT +  ATTENTION_NORM + "weight"]))
                    transformer_layer.layernorm_before.bias.copy_(torch.from_numpy(weights[ROOT + ATTENTION_NORM + "bias"]))
                    transformer_layer.layernorm_after.weight.copy_(torch.from_numpy(weights[ROOT + MLP_NORM + "weight"]))
                    transformer_layer.layernorm_after.bias.copy_(torch.from_numpy(weights[ROOT + MLP_NORM + "bias"]))
                    logging.debug("memory %d MB", self.process.memory_info().rss // 1000000)


                elif kernel_id == 1:

                    qkv_f = str(ROOT + ATTENTION_QKV + "weight")
                    qkv_weight = weights[qkv_f]
                    query_weight = torch.from_numpy(qkv_weight[0:embed_dim,:])
                    key_weight = torch.from_numpy(qkv_weight[embed_dim:embed_dim*2,:])
                    value_weight = torch.from_numpy(qkv_weight[embed_dim*2:embed_dim*3,:])

                    qkv_b = str(ROOT + ATTENTION_QKV + "bias")
                    qkv_bias = weights[qkv_b]
                    query_bias = torch.from_numpy(qkv_bias[0:embed_dim,])
                    key_bias = torch.from_numpy(qkv_bias[embed_dim:embed_dim*2])
                    value_bias= torch.from_numpy(qkv_bias[embed_dim*2:embed_dim*3])

                    transformer_layer[0].weight.copy_(torch.from_numpy(weights[ROOT +  ATTENTION_NORM + "weight"]))
                    transformer_layer[0].bias.copy_(torch.from_numpy(weights[ROOT +  ATTENTION_NORM + "bias"]))
                    transformer_layer[1].query.weight.copy_(query_weight)
                    transformer_layer[1].key.weight.copy_(key_weight)
                    transformer_layer[1].value.weight.copy_(value_weight)

                    transformer_layer[1].query.bias.copy_(query_bias)
                    transformer_layer[1].key.bias.copy_(key_bias)
                    transformer_layer[1].value.bias.copy_(value_bias)

                elif kernel_id == 2:
                    out_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT + "weight"])
                    out_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT + "bias"])
                    transformer_layer[0].dense.weight.copy_(out_weight)
                    transformer_layer[0].dense.bias.copy_(out_bias)
                elif kernel_id == 3:
                    transformer_layer[0].weight.copy_(torch.from_numpy(weights[ROOT + MLP_NORM + "weight"]))
                    transformer_layer[0].bias.copy_(torch.from_numpy(weights[ROOT + MLP_NORM + "bias"]))
                    mlp_weight_0 = torch.from_numpy(weights[ROOT + FC_0 + "weight"])
                    mlp_bias_0 = torch.from_numpy(weights[ROOT + FC_0 + "bias"])
                    transformer_layer[1].dense.weight.copy_(mlp_weight_0)
                    transformer_layer[1].dense.bias.copy_(mlp_bias_0)
                elif kernel_id == 0:
                    mlp_weight_1 = torch.from_numpy(weights[ROOT + FC_1 + "weight"])
                    mlp_bias_1 = torch.from_numpy(weights[ROOT + FC_1 + "bias"])
                    transformer_layer[0].dense.weight.copy_(mlp_weight_1)
                    transformer_layer[0].dense.bias.copy_(mlp_bias_1)

        return transformer_layer

    @torch.no_grad()
    def forward(self, x):
        with self._lock:
            logging.debug("Start memory %d MB", self.process.memory_info().rss / 1000000)
            start = time.time()
            if self.is_first:
                x = self.embeddings(x)
                skip = x
            else:
                x, skip = x[0], x[1]

            for i, op in enumerate(self.first_ops):
                x, skip = _forward_kernel(op, x, skip, (self.start_layer+i)%4)

            for i, layer in enumerate(self.vit_layers):
                logging.debug("Before %d: %d MB", i, self.process.memory_info().rss / 1000000)
                x = layer(x)[0]
                logging.debug("After %d: %d MB", i, self.process.memory_info().rss / 1000000)
                skip = x
            logging.debug("vit-layer memory %d MB", self.process.memory_info().rss / 1000000)

            for i, op in enumerate(self.last_ops):
                # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with _load_layer_weights()
                x, skip = _forward_kernel(op, x, skip, (i+1)%4)

            if self.is_last:
                x = self.layernorm(x)
                x = self.classifier(x[:, 0, :])
            logging.debug("Last memory %d MB", self.process.memory_info().rss / 1000000)
            if self.total_batch == 0:
                self.batch_0_finish = time.time()
            else:
                finish_batch_time = time.time()
                self.total_data += x.shape[0]
                tmp_throughput = self.total_data/(finish_batch_time-self.batch_0_finish)
                logging.info("temporarily throughput is %f", tmp_throughput)

            end = time.time()
            self.total_time +=  (end - start)
            self.total_batch += 1

        logging.info("Round %d: memory %d MB",
                     self.total_batch, self.process.memory_info().rss / 1000000)
        logging.info("Shard%d finishes %d microbatch, time is %f, total time is %f",
                     self.stage, self.total_batch, end - start, self.total_time)
        if self.is_last:
            return x
        return x, skip
