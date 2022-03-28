import os
import sys
import gc
import math
import threading
import argparse
import time
import torch
import psutil
from math import floor
import numpy as np
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from transformers import AutoConfig
sys.path.append(os.path.join(sys.path[0], '..'))
from edgepipe.models.transformers.vit import ViTTransformerShard
from transformers.models.vit.modeling_vit import ViTEmbeddings,  ViTSelfAttention, ViTSelfOutput,ViTIntermediate, ViTOutput
process = psutil.Process(os.getpid())
operators_list = ["LayerNorm + Attention", "Attention Output + residuel Connection", "LayerNorm + MLP-1", "MLP-2 + residuel Connection"]
class ProfileTransformer(ViTTransformerShard):
    def __init__(self, model_name, model_file, start_layer=1, end_layer=-1, repeat_time=10):
        super().__init__(0, model_name, model_file, True, True, 1, 0, load_weight=True)
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        print(f">>>> Profile Model {model_name}")      
        self.embeddings = None
        self.layernorm = None
        self.classifier = None
        self._lock = threading.Lock()
        self.total_time = 0
        self.total_batch = 0
        self.start_layer=start_layer
        self.transfer_data_shape = []
        self.record_kernel_3 = True
        self.repeat_time = repeat_time

        ## operations/transformer layers set
        self.ops = nn.ModuleList()
        ## weight file anme
        if end_layer != -1:
            self.end_layer = end_layer
        elif self.model_name == 'google/vit-base-patch16-224':
            self.weights_file_name = 'ViT-B_16-224.npz'
            self.end_layer = 48
        elif self.model_name == 'google/vit-large-patch16-224':
            self.weights_file_name = 'ViT-L_16-224.npz'
            self.end_layer = 96
        elif self.model_name == 'google/vit-huge-patch14-224-in21k':
            self.weights_file_name = 'ViT-H_14.npz'
            self.end_layer = 128
        self.profile_time = [0]*(self.end_layer-self.start_layer+1)
        self.kernel_memory = [0]*(self.end_layer-self.start_layer+1)
        self._make_layer()

    
    def _make_layer(self):
        ## first Shard
        self.embeddings = ViTEmbeddings(self.config)
        self._load_layer_weights(0, None, load_first = True, load_last=False, load_kernel = False, kernel_id=None)
        for i in range(self.start_layer, self.end_layer+1):
            layer = self._build_kernel(i%4, math.ceil(i/4)-1, True)
            # layer = self.load_layer_weights(math.ceil(i/4)-1, layer, False, False, True, i%4)
            self.ops.append(layer)
            process = psutil.Process(os.getpid())
            print(f"loading layer {i}, memory {process.memory_info().rss / 1000000} MB")
            self.kernel_memory[i-self.start_layer] = process.memory_info().rss / 1000000
            print(self.kernel_memory)
    
        ## last Shard
        num_label = self.config.num_labels
        self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        if self.model_name == 'google/vit-huge-patch14-224-in21k':
            num_label = 21843
        self.classifier = nn.Linear(self.config.hidden_size, num_label) if self.config.num_labels > 0 else nn.Identity()
        self._load_layer_weights(0, None, load_first = False, load_last=True, load_kernel = False, kernel_id=None)

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
            # self.profile_time[0] += emb_time
            skip = x
            for i, op in enumerate(self.ops):
                x, skip = self.forward_kernel(op, x, skip, i)
            tmp_1 = time.time()
            x = self.layernorm(x)
            x = self.classifier(x[:, 0, :])
            # self.profile_time[self.end_layer-1] += time.time()-tmp_1
            end = time.time()
            self.total_time +=  (end - start)
            print(f">>>> Finish profile {iter+1}/{self.repeat_time}, time is {end - start}")
            # process = psutil.Process(os.getpid())
            # print(f"memory {process.memory_info().rss // 1000000} MB")
        self.profile_time = [i/self.repeat_time for i in self.profile_time]
        print(f">>>> Sum kernel time is {sum(self.profile_time)}")
        self.total_time /= self.repeat_time
        return self.profile_time, self.total_time,self.transfer_data_shape, self.kernel_memory

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Proile Model")

    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224", choices=["google/vit-base-patch16-224", 
    "google/vit-large-patch16-224", "google/vit-huge-patch14-224-in21k"], help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str, help="the model file, if not in working directory")
    parser.add_argument("-st", "--start-layer", default=1, type=int, help="start layer")
    parser.add_argument("-ed", "--end-layer", default=-1, type=int, help="end layer")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("-r", "--repeat-time", default=10, type=int, help="repeat time for profiling")
    parser.add_argument("-s", "--save-result",  action="store_true", help="save the results")
    parser.add_argument("-a", "--append-result",  action="store_true", help="append the results")
    parser.add_argument("-c", "--cpu-name", default="unknown_cpu", help="cpu name for saving")
    

    args = parser.parse_args()
    batch_size = args.batch_size     
    model_name = args.model_name
    repeat_time  = args.repeat_time
    start_layer = args.start_layer
    end_layer = args.end_layer
    model_files_default = {
        'google/vit-base-patch16-224': 'ViT-B_16-224.npz',
        'google/vit-large-patch16-224':'ViT-L_16-224.npz',
        'google/vit-huge-patch14-224-in21k': 'ViT-H_14.npz',
        'bert-base-uncased': 'BERT-B.npz',
        'bert-large-uncased': 'BERT-L.npz',
    }
    model_file = args.model_file
    if model_file is None:
        model_file = model_files_default[model_name]
    print(f">>>> Start profiling: model name {model_name} \n>>>> repeat time {repeat_time} \n>>>> batch size {batch_size} \
    \n>>>> save file {args.save_result} \n>>>> append result is {args.append_result}")
    model = ProfileTransformer(model_name, model_file, start_layer, end_layer, repeat_time)
    inputs = torch.randn(batch_size,3,224,224)
    process = psutil.Process(os.getpid())
    print(f"memory {process.memory_info().rss // 1000000} MB")
    time_p, total_time,data_shape, kernel_memory = model(inputs)
    print(f"kernel memory is {kernel_memory}")
    print(f"time is {time_p}, total_time is {total_time}, \ndata shape is {data_shape}")
    print("============== Finish Profiling Model ===============")
    if args.save_result:
        file_name = model_name.split('/')[-1] + "_"+args.cpu_name + "_" + str(batch_size)
        if args.append_result:
            time_pre = np.load(file_name+".npz")['time']
            shape_pre = np.load(file_name+".npz")['shape']
            memory_pre = np.load(file_name+".npz")['memory']
            time_p = np.append(time_pre, time_p)
            data_shape = np.append(shape_pre, data_shape)
            kernel_memory = np.append(kernel_memory, memory_pre )
            print(f"Append the data to previous result, current time is {time_p.shape}:{time_p}")
            print(f"Data shape is {data_shape.shape} : {data_shape}")
        np.savez(file_name, time=time_p, shape=data_shape, memory=kernel_memory)
        print(f"Save the profiling result, the file name is {file_name}")
