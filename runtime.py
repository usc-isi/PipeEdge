import os
import sys
import gc
import threading
import requests
import time
import torch
from functools import wraps
from typing import Optional
import numpy as np
from PIL import Image
from torch import Tensor
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.nn import functional as F
from transformers import AutoConfig, ViTFeatureExtractor, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer

#########################################################
#                 Check Enviroment Settings             #
#########################################################

device = torch.device('cpu')
parallel_threads = 2
torch.set_num_threads(parallel_threads)
torch.set_num_interop_threads(parallel_threads)
torch.set_grad_enabled(False)
print(f"Use device: {device},  # parallel intra nodes threads: {torch.get_num_threads()}, # parallel inter nodes threads: {torch.get_num_interop_threads()}")


#########################################################
#           Define Model Parallel Transformer           #
#########################################################
class TransformerBase(nn.Module):
    def __init__(self, model_name, is_first, is_last, start_layer, end_layer, load_weight=True):
        super(TransformerBase, self).__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        # print(f"Model Config: {self.config}")
        print(f">>>> Model name {model_name}")
        self.is_first = is_first
        self.is_last = is_last
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.load_weight = load_weight
        self.model = self._make_layer()
    
    def _make_layer(self):
        build_layers = []
        ## first Shard
        if self.is_first:
            self.embeddings = ViTEmbeddings(self.config)

        ## [start_layer, end_layer)
        self.layers = nn.ModuleList([ViTLayer(self.config) for i in range(start_layer, end_layer)])

        ## last Shard
        if self.is_last:
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()

        if self.load_weight == True:
            self.load_weights()
            print(f">>>> Finish loading weights")
        else:
            print(f">>>> Do NOT load weights")

        ## build the model shard
        if self.is_first:
            build_layers.append(self.embeddings)
        for i, layer in enumerate(self.layers):
            build_layers.append(layer)
        if self.is_last:
            build_layers.append(self.layernorm)
            build_layers.append(self.classifier)
        return nn.Sequential(*build_layers)

    
    def load_weights(self):
        if self.model_name == 'google/vit-base-patch16-224':
            weights_file_name = 'imagenet21k+imagenet2012_ViT-B_16-224.npz'
        elif self.model_name == 'google/vit-large-patch16-224':
            weights_file_name = 'imagenet21k+imagenet2012_ViT-L_16-224.npz'
        weights = np.load(weights_file_name)
        print(f">>>> Load weight file f{weights_file_name}")

        if self.is_first:
            with torch.no_grad():
                self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
                conv_weight = weights["embedding/kernel"]
                O, I, J, K = conv_weight.shape
                print(f"conv_shape is {O, I, J, K}, pe weight shape is {self.embeddings.patch_embeddings.projection.weight.shape}")
                # conv_weight = conv_weight.reshape(K,J,O,I)
                conv_weight = conv_weight.transpose([3, 2, 0, 1])
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))
                print(f">>>> Load embedding for the first shard")

        for i, layer in enumerate(self.layers):
            layer = self.load_layer_weights(weights, i, layer)
            print(f">>>> Load {i}-th transformer layer")
            
        if self.is_last:
            with torch.no_grad():
                self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
                self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))
                head_kernel = np.transpose(weights["head/kernel"])  
                # print(f"classifier weight is {self.classifier.weight.shape}, head kernel weight shape is {head_kernel.shape}")
                self.classifier.weight.copy_(torch.from_numpy(head_kernel))
                self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))  
                print(f">>>> Load Layernorm, classifier for the last shard")  

    def load_layer_weights(self, weights, id, transformer_layer):
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
        with torch.no_grad():
            query_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

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
        return transformer_layer

    def forward(self, x):
        x = self.embeddings(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)[0]
        x = self.layernorm(x)
        x = self.classifier(x[:, 0, :])
        return x

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]



if __name__=="__main__":
    model_name = "google/vit-base-patch16-224"
    is_first = True
    is_last = True
    start_layer = 0
    end_layer = 12
    load_weight = True
    model = TransformerBase(model_name, is_first, is_last, start_layer, end_layer, load_weight)
    print(model)
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(inputs['pixel_values'])
    # logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = outputs .argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
