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

## Force pytorch use CPU
device = torch.device('cpu')
# parallel_threads = 2
# torch.set_num_threads(parallel_threads)
# torch.set_num_interop_threads(parallel_threads)
torch.set_grad_enabled(False)
print(f"Use device: {device},  # parallel intra nodes threads: {torch.get_num_threads()}, # parallel inter nodes threads: {torch.get_num_interop_threads()}")


#########################################################
#           Define Model Parallel Transformer           #
#########################################################
class TransformerBase(nn.Module):
    def __init__(self, rank, model_name, is_first, is_last, start_layer, end_layer, load_weight=True):
        super(TransformerBase, self).__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        # print(f"Model Config: {self.config}")
        print(f">>>> Model name {model_name}")
        self.rank = rank
        self.is_first = is_first
        self.is_last = is_last
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.load_weight = load_weight
        self._lock = threading.Lock()
        self.total_time = 0
        self.total_batch = 0
        self.model = self._make_layer()
        print(f"======= Build TransformerShard{self.rank} ==========")
    
    def _make_layer(self):
        build_layers = []
        ## first Shard
        if self.is_first:
            self.embeddings = ViTEmbeddings(self.config)

        ## [start_layer, end_layer)
        self.layers = nn.ModuleList([ViTLayer(self.config) for i in range(self.start_layer, self.end_layer)])

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
                # print(f"conv_shape is {O, I, J, K}, pe weight shape is {self.embeddings.patch_embeddings.projection.weight.shape}")
                # conv_weight = conv_weight.reshape(K,J,O,I)
                conv_weight = conv_weight.transpose([3, 2, 0, 1])
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))
                print(f">>>> Load embedding for the first shard")

        for i, layer in enumerate(self.layers):
            layer = self.load_layer_weights(weights, self.start_layer+i, layer)
            print(f">>>> Load {self.start_layer+i}-th transformer layer")
            
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

    @torch.no_grad()
    def forward(self, x):
        x = x.to_here()
        with self._lock:
            start = time.time()
            if self.is_first:
                x = self.embeddings(x)
            for i, layer in enumerate(self.layers):
                x = layer(x)[0]
            if self.is_last:
                x = self.layernorm(x)
                x = self.classifier(x[:, 0, :])
            end = time.time()
        self.total_time +=  (end - start)
        self.total_batch += 1
        print(f"Shard{self.rank} finishes {self.total_batch} microbatch, time is {end -start}, total time is {self.total_time}")
        return x

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


#########################################################
#                Build Transformer Shard                #
#########################################################

## Class factory
def _create_transformershard(class_name, rank, model_name, is_first, is_last, start_layer, end_layer, load_weight=True):
    class TransformerShardCls(TransformerBase):
        def __init__(self):
            super(TransformerShardCls, self).__init__(rank, model_name, is_first, is_last, start_layer, end_layer, load_weight=True)
    TransformerShardCls.__qualname__ = class_name
    return TransformerShardCls

# *****  Define the World Size and partition Method ******#
total_rank = 2 
partition = [0, 6, 6, 12]   
# ***********************  End  **************************#

_shard_class = [f'TransformerShard{i+1}' for i in range(total_rank)]

rank = 0
shard_class_list = []
for _name in _shard_class:
    if rank == 0:
        globals()[_name] = _create_transformershard(_name, rank, 'google/vit-base-patch16-224', True, False, partition[2*rank], partition[2*rank+1], True )
    elif rank == total_rank-1:
        globals()[_name] = _create_transformershard(_name, rank, 'google/vit-base-patch16-224', False, True, partition[2*rank], partition[2*rank+1], True )
    else:
        globals()[_name] = _create_transformershard(_name, rank, 'google/vit-base-patch16-224', False, False, partition[2*rank], partition[2*rank+1], True )
    shard_class_list.append(eval(_name))
    rank += 1


#########################################################
#             Stitch Shards into one Module             #
#########################################################
class DistTransformer(nn.Module):
    def __init__(self, model_name, num_split, workers, *args, **kwargs):
        super(DistTransformer, self).__init__()
        self.num_split = num_split
        self.rref_list = []
        for i in range(total_rank):
            exec(f"self.p{i+1}_rref= rpc.remote(workers[{i}],shard_class_list[{i}])")
            self.rref_list.append(eval(f"self.p{i+1}_rref"))

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            x_rref = RRef(x)
            for i in range(total_rank):
                if i == 0:
                    x_rref = self.rref_list[i].remote().forward(x_rref)
                else:
                    x_rref = self.rref_list[i].rpc_async().forward(x_rref)
            out_futures.append(x_rref)

        return torch.cat(torch.futures.wait_all(out_futures))



#########################################################
#                   Run RPC Processes                   #
#########################################################

# 'google/vit-base-patch16-224'
# 'google/vit-large-patch16-224'
# 'google/vit-huge-patch14-224-in21k'
model_name= 'google/vit-base-patch16-224'
num_batches = 1
batch_size = 16

## random data
# img = torch.randn(3, 384, 384)

## ground truth: Egyptian cat
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
imgs = [image for i in range(batch_size)]



latencies = []
throughputs = []

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

def run_master(split_size):
    # put the two model parts on worker1 and worker2 respectively
    print("Run mastering \n")
    for si in range(len(split_size)):
        # print(f"Start Calcluate split size {split_size[si]}")
        model =  DistTransformer(model_name, split_size[si], ["worker0", "worker1"])
        ## for verification 
        origin_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        inputs = feature_extractor(images=imgs, return_tensors="pt")
        tik = time.time()
        for i in range(num_batches):
            # generate random inputs and labels       
            outputs = model(inputs['pixel_values'])
            print(outputs.shape)
            predicted_class_idx = outputs[0].argmax(-1).item()
            print("Predicted class:", origin_model.config.id2label[predicted_class_idx])
        ## Calculate time
        tok = time.time()
        latency = tok-tik
        throughput = num_batches*batch_size / latency
        # print(f"Split size is {split_size[si]}, Total program execution time = {tok - tik}")
        latencies.append(latency)
        throughputs.append(throughput)
         
    best_choice = -1
    best_throughput  = -1
    for i in range(len(split_size)):
        print(f"Split size {split_size[i]}, latency is {latencies[i]}, throughput is {throughputs[i]}")
        if throughputs[i] > best_throughput:
            best_throughput = throughputs[i]
            best_choice = i 
    print("\n---------------- Split output line ----------------")
    print(f"\nBest split size is {split_size[best_choice]}, latency is {latencies[best_choice]}, throughput is {throughputs[best_choice]}\n")
    
   


def run_worker(rank, world_size, num_split):

    os.environ['MASTER_ADDR'] = '127.0.0.1' #'10.52.3.142'
    os.environ['MASTER_PORT'] = '29501'
    os.environ["TP_SOCKET_IFNAME"] = "lo0"
    os.environ["GLOO_SOCKET_IFNAME"] = 'lo0'

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128,rpc_timeout=3000)


    if rank == 0:
        rpc.init_rpc(
            "worker0",
            rank=rank,
   #         backend=rpc.BackendType.PROCESS_GROUP,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
   #         backend=rpc.BackendType.PROCESS_GROUP,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finisha
    rpc.shutdown()

if __name__=="__main__":
    world_size = 2
    rank=int(sys.argv[1])
    num_split=[8]

    print(f"{model_name}, {batch_size}, {num_split}")
    
    tik = time.time()
    run_worker(rank, world_size, num_split)
    tok = time.time()
    print(f"Total program execution time = {tok - tik}")