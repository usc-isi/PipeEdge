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
from runtime import TransformerShard
from transformers.models.vit.modeling_vit import ViTEmbeddings,  ViTSelfAttention, ViTSelfOutput,ViTIntermediate, ViTOutput
process = psutil.Process(os.getpid())
operators_list = ["LayerNorm + Attention", "Attention Output + residuel Connection", "LayerNorm + MLP-1", "MLP-2 + residuel Connection"]
class ProfileTransformer(TransformerShard):
    def __init__(self, model_name, repeat_time=10):
        super(ProfileTransformer, self).__init__(0, model_name, True, True, 1, 0, load_weight=True)
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
        self.transfer_data_shape = []
        self.record_kernel_3 = True
        self.repeat_time = repeat_time

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
        self.profile_time = [0]*self.end_layer
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

    def forward_kernel(self, layer, x, skip, i):
        start_kernel = time.time()
        kernel_id = (self.start_layer+i)%4
        if kernel_id == 1:
            x = layer[0](x)
            x = layer[1](x)[0]
            if self.record_kernel_3:
                a,b,c=x.shape
                self.transfer_data_shape.append(a*b*c)
            
        elif kernel_id == 2:
            x = layer[0](x, skip)
            x += skip
            skip = x
            if self.record_kernel_3:
                a,b,c=x.shape
                self.transfer_data_shape.append(a*b*c)
        elif kernel_id == 3:
            x = layer[0](x)
            x = layer[1](x)
            if self.record_kernel_3:
                a,b,c=x.shape
                self.transfer_data_shape.append(a*b*c)        
        else:
            x = layer[0](x, skip)
            skip = x
            if self.record_kernel_3:
                a,b,c=x.shape
                self.transfer_data_shape.append(a*b*c)
                self.record_kernel_3 = False
        end_kernel = time.time()
        self.profile_time[i] += (end_kernel-start_kernel)
        return x, skip


    @torch.no_grad()
    def forward(self, data):
        for iter in range(self.repeat_time):
            start = time.time()
            x = self.embeddings(data)
            emb_time = time.time() - start
            self.profile_time[0] += emb_time
            skip = x
            for i, op in enumerate(self.ops):
                x, skip = self.forward_kernel(op, x, skip, i)
            tmp_1 = time.time()
            x = self.layernorm(x)
            x = self.classifier(x[:, 0, :])
            self.profile_time[self.end_layer-1] += time.time()-tmp_1
            end = time.time()
            self.total_time +=  (end - start)
            print(f">>>> Finish profile {iter+1}/{self.repeat_time}, time is {end - start}")
        self.profile_time = [i/self.repeat_time for i in self.profile_time]
        print(f">>>> Sum kernel time is {sum(self.profile_time)}")
        self.total_time /= self.repeat_time
        return self.profile_time, self.total_time,self.transfer_data_shape

if __name__=="__main__":
    batch_size = 8     
    model_name = "google/vit-base-patch16-224" 
    model = ProfileTransformer(model_name)
    inputs = torch.randn(batch_size,3,224,224)
    time_p, total_time,data_shape = model(inputs)
    print(f"time is {time_p}, total_time is {total_time}, data shape is {data_shape}")
