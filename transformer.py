import gc
import logging
import math
import os
import threading
import time
import numpy as np
import psutil
import torch
from torch import nn
from torch.distributed import rpc
from torch.distributed.rpc import RRef
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput, BertLayer
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTSelfAttention, ViTSelfOutput, ViTIntermediate, ViTOutput

#########################################################
#           Define Model Parallel Transformer           #
#########################################################

class TransformerShard(nn.Module):
    """Parent class for transformer shards."""

    def __init__(self, rank, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight=True, is_rpc=False):
        super().__init__()
        self.rank = rank
        self.model_name = model_name
        self.weights_file_name = model_file
        self.is_first = is_first
        self.is_last = is_last
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.load_weight = load_weight
        self.is_rpc = is_rpc

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


class BertTransformerShard(TransformerShard):
    def __init__(self, rank, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight=True, is_rpc=False):
        super().__init__(rank, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight, is_rpc)
        self.embeddings = None

        print(f">>>> Model name {model_name}")
        if self.load_weight:
            print(f">>>> Load weight file {self.weights_file_name}")
        self._make_layer()
        print(f"======= Finish Build BertTransformerShard{self.rank} ==========")

    def _make_layer(self):
        ## first Shard
        if self.is_first:
            self.embeddings = BertEmbeddings(self.config)
            self.embeddings.eval()
            print(f">>>> Load embeddings layer for the first shard ")
            if self.load_weight:
                self.load_layer_weights(0, None, load_first = True, load_last=False, load_kernel = False, kernel_id=None)
                print(f">>>> Load weights for embeddings layer ")

        current_layer_idx = self.start_layer

        ## first ununit part
        if self.start_layer %4 != 1 or (self.start_layer+3 > self.end_layer):
            print(f">>>> For the first model part, load weight is {self.load_weight}:")
            for i in range(self.start_layer, min(self.end_layer, math.ceil(self.start_layer/4)*4)+1):
                print(f"    Load the {i%4}-th operation ({self.operators_list[(i-1)%4]}) for {math.ceil(i/4)-1}-th vit layer")
                layer = self._build_kernel(i%4, math.ceil(i/4)-1, self.load_weight)
                layer.eval()
                self.first_ops.append(layer)
            current_layer_idx = min(self.end_layer+1, math.ceil(self.start_layer/4)*4+1)

        ## mid unit part, the whole vit_layer
        while current_layer_idx + 3 <= self.end_layer:
            with torch.no_grad():
                layer = BertLayer(self.config)
            if self.load_weight:
                layer = self.load_layer_weights(math.ceil(current_layer_idx/4)-1, layer)
            layer.eval()
            self.vit_layers.append(layer)
            print(f">>>> Load the {math.ceil(current_layer_idx/4)-1}-th {self.model_name} Layer, load weight is {self.load_weight}")
            current_layer_idx += 4

        ## last unit part
        if self.end_layer >= current_layer_idx:
            print(f">>>> For the last model part, load weight is {self.load_weight}:")
        for i in range(current_layer_idx, self.end_layer+1):
            print(f"    Load the {i%4}-th operation ({self.operators_list[(i-1)%4]}) for {math.ceil(i/4)-1}-th vit layer")
            layer = self._build_kernel(i%4, math.ceil(i/4)-1, self.load_weight)
            if self.load_weight:
                layer = self.load_layer_weights(math.ceil(i/4)-1, layer, False, False, True, i%4)
            layer.eval()
            self.last_ops.append(layer)

        ## last Shard
        if self.is_last:
            self.bertpooler = BertPooler(self.config)
            self.bertpooler.eval()
            if self.load_weight:
                self.load_layer_weights(0, None, load_first = False, load_last=True, load_kernel = False, kernel_id=None)
                print(">>>> Load weights for layernorm and last shard")


        if self.load_weight:
            print(">>>> Finish load weights")
        else:
            print(">>>> Do NOT load weights")

    def _build_kernel(self, kernel_id, vit_layer_id, load_weight=True):
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
            self.load_layer_weights(vit_layer_id, layers, False, False, load_weight, kernel_id)
        return layers

    def load_layer_weights(self, id, transformer_layer, load_first = False, load_last=False, load_kernel = False, kernel_id=None):
        weights = np.load(self.weights_file_name)
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
                    print(f"memory {self.process.memory_info().rss // 1000000} MB")

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

    def forward_kernel(self, layer, x, skip, kernel_id):
        if kernel_id == 1:
            x = layer[0](x)
        elif kernel_id == 2:
            x = x[0]
            x = layer[0](x, skip)
            skip = x
        elif kernel_id == 3:
            x = layer[0](x)
        else:
            x = layer[0](x, skip)
            skip = x
        return x, skip


    @torch.no_grad()
    def forward(self, x):
        if self.is_rpc:
            x = x.to_here()
        with self._lock:
            start = time.time()
            if self.is_first:
                x = self.embeddings(x)
                skip = x
            else:
                x, skip = x[0], x[1]

            for i, op in enumerate(self.first_ops):
                x, skip = self.forward_kernel(op, x, skip, (self.start_layer+i)%4)

            for layer in self.vit_layers:
                with torch.no_grad():
                    x = layer(x)[0]
                    skip = x

            for i, op in enumerate(self.last_ops):
                # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with load_layer_weights()
                x, skip = self.forward_kernel(op, x, skip, (i+1)%4)

            if self.is_last:
                x = self.bertpooler(x)
            end = time.time()
            if self.total_batch == 0:
                self.batch_0_finish = time.time()
            else:
                finish_batch_time = time.time()
                self.total_data += x.shape[0] ##14 #split_size
                tmp_throughput = self.total_data/(finish_batch_time-self.batch_0_finish)
                print(f"total data is {self.total_data}, time is {finish_batch_time-self.batch_0_finish}, temporarily throughput is  {tmp_throughput} ")

        self.total_time +=  (end - start)
        self.total_batch += 1
        print(f"Round {self.total_batch}: memory {self.process.memory_info().rss // 1000000} MB")
        print(f"Shard{self.rank} finishes {self.total_batch} microbatch, time is {end -start}, total time is {self.total_time}")
        if self.is_last:
            return x
        return x, skip

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class ViTTransformerShard(TransformerShard):
    def __init__(self, rank, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight=True, is_rpc=False):
        super().__init__(rank, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight, is_rpc)
        self.embeddings = None
        self.layernorm = None
        self.classifier = None

        print(f">>>> Model name {model_name}")
        if self.load_weight:
            print(f">>>> Load weight file {self.weights_file_name}")
            logging.info(f">>>> Load weight file f{self.weights_file_name}")
        self._make_layer()
        print(f"======= Finish Build ViTTransformerShard{self.rank} ==========")
        logging.info(f"======= Finish Build ViTTransformerShard{self.rank} ==========")

    def _make_layer(self):
        ## first Shard
        if self.is_first:
            self.embeddings = ViTEmbeddings(self.config)
            print(f">>>> Load embeddings layer for the first shard ")
            if self.load_weight:
                self.load_layer_weights(0, None, load_first = True, load_last=False, load_kernel = False, kernel_id=None)
                print(f">>>> Load weights for embeddings layer ")

        current_layer_idx = self.start_layer

        ## first ununit part
        if self.start_layer %4 != 1 or (self.start_layer+3 > self.end_layer):
            print(f">>>> For the first model part, load weight is {self.load_weight}:")
            for i in range(self.start_layer, min(self.end_layer, math.ceil(self.start_layer/4)*4)+1):
                print(f"    Load the {i%4}-th operation ({self.operators_list[(i-1)%4]}) for {math.ceil(i/4)-1}-th vit layer")
                layer = self._build_kernel(i%4, math.ceil(i/4)-1, self.load_weight)
                self.first_ops.append(layer)
            current_layer_idx = min(self.end_layer+1, math.ceil(self.start_layer/4)*4+1)

        ## mid unit part, the whole vit_layer
        while current_layer_idx + 3 <= self.end_layer:
            layer = ViTLayer(self.config)
            if self.load_weight:
                layer = self.load_layer_weights(math.ceil(current_layer_idx/4)-1, layer)
            self.vit_layers.append(layer)
            print(f">>>> Load the {math.ceil(current_layer_idx/4)-1}-th ViT Layer, load weight is {self.load_weight}")
            current_layer_idx += 4

        ## last unit part
        if self.end_layer >= current_layer_idx:
            print(f">>>> For the last model part, load weight is {self.load_weight}:")
        for i in range(current_layer_idx, self.end_layer+1):
            print(f"    Load the {i%4}-th operation ({self.operators_list[(i-1)%4]}) for {math.ceil(i/4)-1}-th vit layer")
            layer = self._build_kernel(i%4, math.ceil(i/4)-1, self.load_weight)
            if self.load_weight:
                layer = self.load_layer_weights(math.ceil(i/4)-1, layer, False, False, True, i%4)
            self.last_ops.append(layer)

        ## last Shard
        if self.is_last:
            num_label = self.config.num_labels
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            print(f">>>> Load layernorm for the last shard")
            if self.model_name == 'google/vit-huge-patch14-224-in21k':
                num_label = 21843
            self.classifier = nn.Linear(self.config.hidden_size, num_label) if self.config.num_labels > 0 else nn.Identity()
            print(f">>>> Load classifier for the last shard")
            if self.load_weight:
                self.load_layer_weights(0, None, load_first = False, load_last=True, load_kernel = False, kernel_id=None)
                print(">>>> Load weights for layernorm and last shard")


        if self.load_weight:
            print(">>>> Finish load weights")
        else:
            print(">>>> Do NOT load weights")

    def _build_kernel(self, kernel_id, vit_layer_id, load_weight=True):
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
            self.load_layer_weights(vit_layer_id, layers, False, False, load_weight, kernel_id)
        return layers

    def load_layer_weights(self, id, transformer_layer, load_first = False, load_last=False, load_kernel = False, kernel_id=None):
        weights = np.load(self.weights_file_name)
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
                # print(f"conv_shape is {O, I, J, K}, pe weight shape is {self.embeddings.patch_embeddings.projection.weight.shape}")
                # conv_weight = conv_weight.reshape(K,J,O,I)
                conv_weight = conv_weight.transpose([3, 2, 0, 1])
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))
                # print(f">>>> Load embedding for the first shard")

        if load_last:
            with torch.no_grad():
                self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
                self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))
                # head_kernel = np.transpose(weights["head/kernel"])
                # print(f"classifier weight is {self.classifier.weight.shape}, head kernel weight shape is {head_kernel.shape}")
                self.classifier.weight.copy_(torch.from_numpy(np.transpose(weights["head/kernel"])))
                self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))
                # print(f">>>> Load Layernorm, classifier for the last shard")


        if not load_first and not load_last:
            with torch.no_grad():
                if not load_kernel:

                    query_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")]).view(hidden_size, hidden_size).t()
                    print("query weight shape is ", query_weight.shape)
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
                    print(f"memory {self.process.memory_info().rss // 1000000} MB")


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

    def forward_kernel(self, layer, x, skip, kernel_id):
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


    @torch.no_grad()
    def forward(self, x):
        if self.is_rpc:
            x = x.to_here()
        x, skip = x[0], x[1]
        with self._lock:
            logging.info(f"Start memory {self.process.memory_info().rss / 1000000} MB")
            start = time.time()
            if self.is_first:
                x = self.embeddings(x)
                skip = x

            for i, op in enumerate(self.first_ops):
                x, skip = self.forward_kernel(op, x, skip, (self.start_layer+i)%4)

            for i, layer in enumerate(self.vit_layers):
                logging.info(f"Before {i}: {self.process.memory_info().rss / 1000000} MB")
                x = layer(x)[0]
                logging.info(f"After {i}: {self.process.memory_info().rss / 1000000} MB")
                gc.collect()
                skip = x
            logging.info(f"vit-layer memory {self.process.memory_info().rss / 1000000} MB")

            for i, op in enumerate(self.last_ops):
                # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with load_layer_weights()
                x, skip = self.forward_kernel(op, x, skip, (i+1)%4)

            if self.is_last:
                x = self.layernorm(x)
                x = self.classifier(x[:, 0, :])
            logging.info(f"Last memory {self.process.memory_info().rss / 1000000} MB")
            if self.total_batch == 0:
                self.batch_0_finish = time.time()
            else:
                finish_batch_time = time.time()
                self.total_data += x.shape[0]
                tmp_throughput = self.total_data/(finish_batch_time-self.batch_0_finish)
                print(f"temporarily throughput is  {tmp_throughput} ")
                logging.info(f"temporarily throughput is {tmp_throughput}")

            end = time.time()
            self.total_time +=  (end - start)
            self.total_batch += 1

        print(f"Round {self.total_batch}: memory {self.process.memory_info().rss / 1000000} MB")
        print(f"Shard{self.rank} finishes {self.total_batch} microbatch, time is {end -start}, total time is {self.total_time}")
        logging.info(f"Round {self.total_batch}: memory {self.process.memory_info().rss / 1000000} MB")
        logging.info(f"Shard{self.rank} finishes {self.total_batch} microbatch, time is {end -start}, total time is {self.total_time}")
        if self.is_last:
            return x
        return x, skip


#########################################################
#             Stitch Shards into one Module             #
#########################################################

class DistTransformer(nn.Module):
    """Parent class for distributed transformers."""
    def __init__(self, world_size, num_split):
        super().__init__()
        self.world_size = world_size
        self.num_split = num_split
        self.rref_list = []

    def forward(self, xs):
        """Still-abstract forward function."""
        raise NotImplementedError


class BertDistTransformer(DistTransformer):
    def __init__(self, model_name, model_file, world_size, partition, num_split):
        super().__init__(world_size, num_split)
        for i in range(world_size):
            # Build Transformer Shard
            is_first = i == 0
            is_last = i == world_size-1
            rref = rpc.remote(f"worker{i}", BertTransformerShard,
                              args=(i, model_name, model_file, is_first, is_last, partition[2*i],
                                    partition[2*i+1], True, True))
            self.rref_list.append(rref)

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            x_rref = RRef(x)
            for i in range(self.world_size-1):
                x_rref = self.rref_list[i].remote().__call__(x_rref)
            y_rref = self.rref_list[self.world_size-1].rpc_async().__call__(x_rref)
            out_futures.append(y_rref)
        # res = torch.cat(torch.futures.wait_all(out_futures))
        # res = x_rref.to_here()
        # del out_futures
        # gc.collect()
        return torch.cat(torch.futures.wait_all(out_futures))


class ViTDistTransformer(DistTransformer):
    def __init__(self, model_name, model_file, world_size, partition, num_split):
        super().__init__(world_size, num_split)
        for i in range(world_size):
            # Build Transformer Shard
            is_first = i == 0
            is_last = i == world_size-1
            rref = rpc.remote(f"worker{i}", ViTTransformerShard,
                              args=(i, model_name, model_file, is_first, is_last, partition[2*i],
                                    partition[2*i+1], True, True))
            self.rref_list.append(rref)

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            skip = torch.zeros(x.size())
            x = torch.stack((x, skip), 0)
            x_rref = RRef(x)
            for i in range(self.world_size-1):
                x_rref = self.rref_list[i].remote().__call__(x_rref)
            y_rref = self.rref_list[self.world_size-1].rpc_async().__call__(x_rref)
            out_futures.append(y_rref)
        # res = torch.cat(torch.futures.wait_all(out_futures))
        # res = x_rref.to_here()
        # del out_futures
        # gc.collect()
        return torch.cat(torch.futures.wait_all(out_futures))
