import os
import sys
import gc
import math
import threading
import psutil
import requests
import time
import argparse
import torch
from math import floor
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info
from torch.nn import functional as F
from transformers import AutoConfig, ViTFeatureExtractor, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTSelfAttention, ViTSelfOutput,ViTIntermediate, ViTOutput

#########################################################
#           Define Model Parallel Transformer           #
#########################################################
def fetch_rref_value(rref):
    return rref.local_value()

class TransformerShard(nn.Module):
    def __init__(self, rank, model_name, is_first, is_last, start_layer, end_layer, load_weight=True):
        super(TransformerShard, self).__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        print(f">>>> Model name {model_name}")
        self.rank = rank
        self.is_first = is_first
        self.is_last = is_last
        self.embeddings = None
        self.layernorm = None
        self.classifier = None
        self.start_layer = start_layer
        self.end_layer = end_layer

        self.load_weight = load_weight
        self._lock = threading.Lock()
        self.total_time = 0
        self.total_batch = 0

        ## operations/transformer layers set
        self.first_ops = nn.ModuleList()
        self.vit_layers = nn.ModuleList()
        self.last_ops = nn.ModuleList()

        ## weight file anme
        if self.model_name == 'google/vit-base-patch16-224':
            self.weights_file_name = 'ViT-B_16-224.npz'
        elif self.model_name == 'google/vit-large-patch16-224':
            self.weights_file_name = 'ViT-L_16-224.npz'
        elif self.model_name == 'google/vit-huge-patch14-224-in21k':
            self.weights_file_name = 'ViT-H_14.npz'
        if self.load_weight:
            print(f">>>> Load weight file f{self.weights_file_name}")
        self._make_layer()
        print(f"======= Finish Build TransformerShard{self.rank} ==========")
    
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
                print(f"    Load the {i%4}-th operation ({operators_list[(i-1)%4]}) for {math.ceil(i/4)-1}-th vit layer")
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
            print(f"    Load the {i%4}-th operation ({operators_list[(i-1)%4]}) for {math.ceil(i/4)-1}-th vit layer")
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

        if load_first == False and load_last == False:
            with torch.no_grad():
                if load_kernel == False:
                
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
                    print(f"memory {process.memory_info().rss // 1000000} MB")
                   
                elif kernel_id == 1:

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
                elif kernel_id == 2:
                    out_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
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
    def forward_pre(self, x):
        # if self.is_first:
        #     x = x_rref.to_here()
        # else:
        #     x, skip = x_rref.to_here()
        skip = x[1]
        x = x[0]
        with self._lock:
            start = time.time()
            if self.is_first:
                x = self.embeddings(x)
                skip = x

            for i, op in enumerate(self.first_ops):
                x, skip = self.forward_kernel(op, x, skip, (self.start_layer+i)%4)

            for layer in self.vit_layers:
                x = layer(x)[0]
                skip = x

            for i, op in enumerate(self.last_ops):
                # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with load_layer_weights()
                x, skip = self.forward_kernel(op, x, skip, (i+1)%4)

            if self.is_last:
                x = self.layernorm(x)
                x = self.classifier(x[:, 0, :])
            end = time.time()

        # self.total_time +=  (end - start)
        # self.total_batch += 1
        # print(f"Round {self.total_batch}: memory {process.memory_info().rss // 1000000} MB")
        # print(f"Shard{self.rank} finishes {self.total_batch} microbatch, time is {end -start}, total time is {self.total_time}")
        if self.is_last:
            return x
        return x, skip
    
    @staticmethod 
    @rpc.functions.async_execution
    def forward(self_rref, x_rref):
        self = self_rref.local_value()
        fut = rpc.rpc_async(x_rref.owner(), fetch_rref_value, args=(x_rref,))
        process = psutil.Process(os.getpid())
        self.total_batch  += 1
        print(f"Model1 start forward {self.total_batch} microbatch, memory {process.memory_info().rss / 1000000} MB")
        def bottom_half(future):
            y = self.forward_pre(future.wait())
            return y
        return fut.then(bottom_half)

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


#########################################################
#             Stitch Shards into one Module             #
#########################################################
class DistTransformer(nn.Module):
    def __init__(self, model_name, world_size, num_split):
        super(DistTransformer, self).__init__()
        self.world_size = world_size
        self.num_split = num_split
        self.rref_list = []
        for i in range(world_size):
            # Build Transformer Shard
            is_first = i == 0
            is_last = i == world_size-1
            rref = rpc.remote(f"worker{i}", TransformerShard, args=(i, model_name, is_first, is_last, partition[2*i], partition[2*i+1], True))
            self.rref_list.append(rref)

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            skip = torch.zeros(x.size())
            x = torch.stack((x, skip), 0)
            x_rref = RRef(x)
            for i in range(self.world_size-1):
                # x_rref = self.rref_list[i].remote().forward(x_rref)
                x_rref = rpc.remote(
                    self.rref_list[i].owner(),
                    # self.worker0,
                    TransformerShard.forward,
                    args=(self.rref_list[i], x_rref ),
                )
            # y_rref = self.rref_list[self.world_size-1].rpc_async().forward(x_rref)
            y_rref= rpc.rpc_async(
                    self.rref_list[self.world_size-1].owner(),
                    # self.worker1,
                    TransformerShard.forward,
                    args=(self.rref_list[self.world_size-1], x_rref),
                )

            # x_rref.to_here()
            out_futures.append(y_rref)
        return torch.cat(torch.futures.wait_all(out_futures))


#########################################################
#                   Run RPC Processes                   #
#########################################################

def run_master(model_name, world_size, split_size):
    print("Run mastering \n")
    latencies = []
    throughputs = []
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    ## for verification 
    # origin_model = ViTForImageClassification.from_pretrained(model_name)
    for si in range(len(split_size)):
        # print(f"Start Calcluate split size {split_size[si]}")
        model =  DistTransformer(model_name, world_size, split_size[si])
        inputs = feature_extractor(images=imgs, return_tensors="pt")
        tik = time.time()
        for i in range(num_batches):
            # generate random inputs and labels       
            outputs = model(inputs['pixel_values'])
            print(f"outputs is {outputs}")
            # predicted_class_idx = outputs[0].argmax(-1).item()
            # print("Predicted class:", origin_model.config.id2label[predicted_class_idx])
        ## Calculate time
        tok = time.time()
        latency = tok-tik
        throughput = num_batches*batch_size / latency
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
    print(f"\nBest split size is {split_size[best_choice]}, Execution time is {latencies[best_choice]}, throughput is {throughputs[best_choice]}\n")


def run_worker(model_name, rank, world_size, num_split):

    os.environ['MASTER_ADDR'] = args.addr #MASTER_ADDR #'10.52.3.175' #'127.0.0.1' # '172.30.0.21'
    os.environ['MASTER_PORT'] = args.port # MASTER_PORT
    os.environ["TP_SOCKET_IFNAME"] = args.socket_ifname #SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = args.socket_ifname #SOCKET_IFNAME

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=num_worker_threads,rpc_timeout=3000)

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
#         backend=rpc.BackendType.PROCESS_GROUP,
        world_size=world_size,
        rpc_backend_options=options
    )
    if rank == 0:
        run_master(model_name, world_size, num_split)

    # block until all rpcs finisha
    rpc.shutdown()

if __name__=="__main__":
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Runtime")
    parser.add_argument("rank", type=int, help="the rank for the current node")
    parser.add_argument("worldsize", type=int, help="the world size (the number of nodes)")
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224", choices=["google/vit-base-patch16-224", 
    "google/vit-large-patch16-224", "google/vit-huge-patch14-224-in21k"], help="the neural network model for loading")
    parser.add_argument("-pt", "--partition", default="1,48", help="the partition method")
    parser.add_argument("--addr", type=str, default="127.0.0.1", help="ip address for the master node")
    parser.add_argument("--port", type=str, default="29500", help="communication port for the master node")
    parser.add_argument("-s", "--socket-ifname", type=str, default="lo0", help="socket iframe name, use [ifconfig | ipaddress] to check")
    parser.add_argument("-p","--print", type=str, default = "None", choices=["full", "short", "default"], help="print the [full | short] tensor values")
    parser.add_argument("-t", "--threshold", default=1000, type=int, help="total number of array elements which trigger summarization rather than full repr")
    parser.add_argument("-n", "--num-batches", default=1, type=int, help="total number of batches")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-w", "--worker-threads", default=64, type=int, help="the number of worker threads for the communication backend")
    parser.add_argument("-sp", "--splits", default="8", help="the list of microbatch size")
    args = parser.parse_args()
    torch.set_printoptions(profile=args.print,threshold=args.threshold)  
    ## Force pytorch use CPU
    device = torch.device('cpu')
    # parallel_threads = 2
    # torch.set_num_threads(parallel_threads)
    # torch.set_num_interop_threads(parallel_threads)
    torch.set_grad_enabled(False)
    print(f"Use device: {device},  # parallel intra nodes threads: {torch.get_num_threads()}, # parallel inter nodes threads: {torch.get_num_interop_threads()}")
    process = psutil.Process(os.getpid())
    #########################################################
    #                 Configuration for Network             #
    #########################################################
    # *****  Define the World Size and partition Method ******#
    partition =   [int(i) for i in args.partition.split(',')]
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_worker_threads = args.worker_threads
    operators_list = ["LayerNorm + Attention", "Attention Output + residuel Connection", "LayerNorm + MLP-1", "MLP-2 + residuel Connection"]
    ## random data
    # img = torch.randn(3, 384, 384)
    ## ground truth: Egyptian cat
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    # image = Image.open('./images/panda.jpeg')
    imgs = [image for i in range(batch_size)]

    # ***********************  End  **************************#
    world_size = args.worldsize
    rank=args.rank
    num_split = [int(i) for i in args.splits.split(',')]
    model_name= args.model_name

    print(f"Model name is {model_name}, Batch size is {batch_size}, Split size is: {num_split}, \n Split method is {partition}, GLOO Threads is {num_worker_threads}")
    
    tik = time.time()
    run_worker(model_name, rank, world_size, num_split)
    tok = time.time()
    print(f"Total program execution time = {tok - tik}")