import os
import sys
import gc
import math
import threading
import psutil
import requests
import time
import torch
from math import floor
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from transformers import AutoConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings,  ViTSelfAttention, ViTSelfOutput,ViTIntermediate, ViTOutput
process = psutil.Process(os.getpid())
operators_list = ["LayerNorm + Attention", "Attention Output + residuel Connection", "LayerNorm + MLP-1", "MLP-2 + residuel Connection"]
class ProfileTransformer(nn.Module):
    def __init__(self, model_name):
        super(ProfileTransformer, self).__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        print(f">>>> Profile Model {model_name}")      
        self.embeddings = None
        self.layernorm = None
        self.classifier = None
        self._lock = threading.Lock()
        self.total_time = 0
        self.total_batch = 0
        self.start_layer=1
        self.profile_time = []
        self.transfer_data_shape = []
        self.record_kernel_3 = True

        ## operations/transformer layers set
        self.ops = nn.ModuleList()
        ## weight file anme
        if self.model_name == 'google/vit-base-patch16-224':
            self.weights_file_name = 'ViT-B_16-224.npz'
            self.end_layer = 48
        elif self.model_name == 'google/vit-large-patch16-224':
            self.weights_file_name = 'ViT-L_16-224.npz'
            self.end_layer = 96
        elif self.model_name == 'google/vit-huge-patch14-224-in21k':
            self.weights_file_name = 'ViT-H_14.npz'
            self.end_layer = 128
        self._make_layer()

    
    def _make_layer(self):
        ## first Shard
        self.embeddings = ViTEmbeddings(self.config)
        self.load_layer_weights(0, None, load_first = True, load_last=False, load_kernel = False, kernel_id=None)
        for i in range(self.start_layer, self.end_layer+1):
            layer = self._build_kernel(i%4, math.ceil(i/4)-1, True)
            # layer = self.load_layer_weights(math.ceil(i/4)-1, layer, False, False, True, i%4)
            self.ops.append(layer)
    
        ## last Shard
        num_label = self.config.num_labels
        self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        if self.model_name == 'google/vit-huge-patch14-224-in21k':
            num_label = 21843
        self.classifier = nn.Linear(self.config.hidden_size, num_label) if self.config.num_labels > 0 else nn.Identity()
        self.load_layer_weights(0, None, load_first = False, load_last=True, load_kernel = False, kernel_id=None)



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
        self.hidden_size = self.config.hidden_size
        if load_first:
            with torch.no_grad():
                self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
                conv_weight = weights["embedding/kernel"]
                O, I, J, K = conv_weight.shape
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
                head_kernel = np.transpose(weights["head/kernel"])  
                # print(f"classifier weight is {self.classifier.weight.shape}, head kernel weight shape is {head_kernel.shape}")
                self.classifier.weight.copy_(torch.from_numpy(head_kernel))
                self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))  
                # print(f">>>> Load Layernorm, classifier for the last shard")  


        with torch.no_grad():
            if kernel_id == 1:
                query_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                key_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                value_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()

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
                del query_weight, key_weight, value_weight, query_bias, key_bias, value_bias
            elif kernel_id == 2:
                out_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                out_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "bias")]).view(-1)
                transformer_layer[0].dense.weight.copy_(out_weight)
                transformer_layer[0].dense.bias.copy_(out_bias)
                del out_weight, out_bias
            elif kernel_id == 3:
                transformer_layer[0].weight.copy_(torch.from_numpy(weights[os.path.join(ROOT, MLP_NORM, "scale")]))
                transformer_layer[0].bias.copy_(torch.from_numpy(weights[os.path.join(ROOT, MLP_NORM, "bias")]))
                mlp_weight_0 = torch.from_numpy(weights[os.path.join(ROOT, FC_0, "kernel")]).t()
                mlp_bias_0 = torch.from_numpy(weights[os.path.join(ROOT, FC_0, "bias")]).t()
                transformer_layer[1].dense.weight.copy_(mlp_weight_0)
                transformer_layer[1].dense.bias.copy_(mlp_bias_0)
                del mlp_weight_0, mlp_bias_0
            elif kernel_id == 0:
                mlp_weight_1 = torch.from_numpy(weights[os.path.join(ROOT, FC_1, "kernel")]).t()
                mlp_bias_1 = torch.from_numpy(weights[os.path.join(ROOT, FC_1, "bias")]).t()
                transformer_layer[0].dense.weight.copy_(mlp_weight_1)
                transformer_layer[0].dense.bias.copy_(mlp_bias_1)
                del mlp_weight_1, mlp_bias_1

        del weights
        gc.collect()
        return transformer_layer
    
    def forward_kernel(self, layer, x, skip, i):
        start_kernel = time.time()
        kernel_id = (self.start_layer+i)%4
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
            if self.record_kernel_3:
                a,b,c=x.shape
                self.transfer_data_shape.append(a*b*c)
                self.record_kernel_3 = False
        else:
            x = layer[0](x, skip)
            skip = x
        end_kernel = time.time()
        self.profile_time.append(end_kernel-start_kernel)
        return x, skip


    @torch.no_grad()
    def forward(self, x):
        
        with self._lock:
            start = time.time()
            x = self.embeddings(x)
            skip = x
            a,b,c = x.shape
            self.transfer_data_shape.append(a*b*c)
            for i, op in enumerate(self.ops):
                x, skip = self.forward_kernel(op, x, skip, i)
            x = self.layernorm(x)
            x = self.classifier(x[:, 0, :])
            end = time.time()

        self.total_time +=  (end - start)
        return self.profile_time, end-start,self.transfer_data_shape

if __name__=="__main__":
    batch_size = 8     
    model_name = "google/vit-base-patch16-224" 
    model = ProfileTransformer(model_name)
    inputs = torch.randn(batch_size,3,224,224)
    time_p, total_time,data_shape = model(inputs)
    print(f"time is {time_p}, total_time is {total_time}, data shape is {data_shape}")
