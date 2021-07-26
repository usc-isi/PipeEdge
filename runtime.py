import os
import sys
import gc
import threading
import requests
import time
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
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTIntermediate,ViTOutput, ViTAttention

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
        self.start_layer_is_half = False
        self.end_layer_is_half = False
        if(floor(start_layer) < start_layer):
            self.start_layer_is_half = True
        if(floor(end_layer) < end_layer):
            self.end_layer_is_half = True
        self.load_weight = load_weight
        self._lock = threading.Lock()
        self.total_time = 0
        self.total_batch = 0
        self.model = self._make_layer()
        print(f"======= Finish Build TransformerShard{self.rank} ==========")
    
    def _make_layer(self):
        build_layers = []
        ## first Shard
        if self.is_first:
            self.embeddings = ViTEmbeddings(self.config)

        ## Build [start_layer, end_layer]
        ## Judge start_layer has half layer
        vit_layer_start_id = floor(self.start_layer)
        self.layers = nn.ModuleList()
        if self.start_layer_is_half:
            self.layernorm_after = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self.intermediate = ViTIntermediate(self.config)
            self.output_layer = ViTOutput(self.config)
            self.layers = nn.ModuleList([self.layernorm_after])
            self.layers.append(self.intermediate)
            self.layers.append(self.output_layer)
            ## start layer = 2.5 means load the 3-th layer's mlp+norm layer
            vit_layer_start_id = floor(self.start_layer)+2
        
        ## NOTE: include self.end_layer
        vit_layer_end_id = floor(self.end_layer)+1

        for _ in range(vit_layer_start_id, vit_layer_end_id):
            self.layers.append(ViTLayer(self.config))

        # self.layers.extend([ViTLayer(self.config) for i in range(vit_layer_start_id, vit_layer_end_id)])

        if self.end_layer_is_half:
            self.layernorm_before = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self.attention = ViTAttention(self.config)
            self.layers.append(self.layernorm_before)
            self.layers.append(self.attention)
            

        ## last Shard
        if self.is_last:
            num_label = self.config.num_labels
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            if self.model_name == 'google/vit-huge-patch14-224-in21k':
                num_label = 21843
            self.classifier = nn.Linear(self.config.hidden_size, num_label) if self.config.num_labels > 0 else nn.Identity()

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
            weights_file_name = 'ViT-B_16-224.npz'
        elif self.model_name == 'google/vit-large-patch16-224':
            weights_file_name = 'ViT-L_16-224.npz'
        elif self.model_name == 'google/vit-huge-patch14-224-in21k':
            weights_file_name = 'ViT-H_14.npz'
        weights = np.load(weights_file_name)
        print(f">>>> Load weight file {weights_file_name}")

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
        

        ## Load Start half Layer
        if self.start_layer_is_half or self.end_layer_is_half:
            self.load_half_layer_weights(weights, floor(self.start_layer)+1, floor(self.end_layer)+1, self.start_layer_is_half, self.end_layer_is_half)
            

        ## Judge start whole VIT_Layer id:
        if self.start_layer_is_half:
            vit_layer_start_id = floor(self.start_layer)+2
            print(f">>>> Load {floor(self.start_layer)+2}'s MLP + LayerNorm layer for start half layer")
        else:
            vit_layer_start_id = self.start_layer
        

        for i, layer in enumerate(self.layers):
            if self.start_layer_is_half and i < 3:
                continue
            if self.end_layer_is_half and i > len(self.layers)-3:
                continue  
            if self.start_layer_is_half:
                vit_layer_id = vit_layer_start_id+i-3
            else:
                vit_layer_id = vit_layer_start_id+i
            layer = self.load_layer_weights(weights, vit_layer_id, layer)
            print(f">>>> Load {vit_layer_id}-th transformer layer")

        if self.end_layer_is_half:
            print(f">>>> Load {floor(self.end_layer)+1}'s Attention + LayerNorm layer for end  half layer")
            
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
        print(f"{id}, {ROOT}")
        with torch.no_grad():
            query_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # print(query_weight)
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
    
    def load_half_layer_weights(self, weights, start_id, end_id, is_start_half_layer, is_end_half_layer):
        self.hidden_size = self.config.hidden_size
        with torch.no_grad():
            if is_start_half_layer:
                ROOT = f"Transformer/encoderblock_{start_id}"
                FC_0 = "MlpBlock_3/Dense_0"
                FC_1 = "MlpBlock_3/Dense_1"
                MLP_NORM = "LayerNorm_2"

                self.layers[0].weight.copy_(torch.from_numpy(weights[os.path.join(ROOT, MLP_NORM, "scale")]))
                self.layers[0].bias.copy_(torch.from_numpy(weights[os.path.join(ROOT, MLP_NORM, "bias")]))

                mlp_weight_0 = torch.from_numpy(weights[os.path.join(ROOT, FC_0, "kernel")]).t()
                mlp_weight_1 = torch.from_numpy(weights[os.path.join(ROOT, FC_1, "kernel")]).t()
                mlp_bias_0 = torch.from_numpy(weights[os.path.join(ROOT, FC_0, "bias")]).t()
                mlp_bias_1 = torch.from_numpy(weights[os.path.join(ROOT, FC_1, "bias")]).t()

                self.layers[1].dense.weight.copy_(mlp_weight_0)
                self.layers[1].dense.bias.copy_(mlp_bias_0)
                self.layers[2].dense.weight.copy_(mlp_weight_1)
                self.layers[2].dense.bias.copy_(mlp_bias_1)

            if is_end_half_layer:
                ROOT = f"Transformer/encoderblock_{end_id}"
                ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
                ATTENTION_K = "MultiHeadDotProductAttention_1/key"
                ATTENTION_V = "MultiHeadDotProductAttention_1/value"
                ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
                ATTENTION_NORM = "LayerNorm_0"
                self.layers[-2].weight.copy_(torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_NORM, "scale")]))
                self.layers[-2].bias.copy_(torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_NORM, "bias")]))
                query_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                key_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                value_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                out_weight = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

                query_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_Q, "bias")]).view(-1)
                key_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_K, "bias")]).view(-1)
                value_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_V, "bias")]).view(-1)
                out_bias = torch.from_numpy(weights[os.path.join(ROOT, ATTENTION_OUT, "bias")]).view(-1)

                self.layers[-1].attention.query.weight.copy_(query_weight)
                self.layers[-1].attention.key.weight.copy_(key_weight)
                self.layers[-1].attention.value.weight.copy_(value_weight)
                self.layers[-1].output.dense.weight.copy_(out_weight)

                self.layers[-1].attention.query.bias.copy_(query_bias)
                self.layers[-1].attention.key.bias.copy_(key_bias)
                self.layers[-1].attention.value.bias.copy_(value_bias)
                self.layers[-1].output.dense.bias.copy_(out_bias)



    @torch.no_grad()
    def forward(self, x_rref):
        # print(x_rref)
        x = x_rref.to_here()
        # print(x)
        # del x_rref
        # gc.collect()
        ## dequen
        # if self.rank > 0:
        #     print(f"Before dequan, {x}")
        #     x = (x - 0) * 0.03
        #     print(f"After dequan {x}")
        if self.start_layer_is_half:
            start_id = 3
        else: start_id = 0
    
        if self.end_layer_is_half:
            end_id = len(self.layers)-3
        else: end_id = len(self.layers)-1 
        
        with self._lock:
            start = time.time()
            if self.is_first:
                x = self.embeddings(x)
                # print(f"After embeddings {x}")
            if self.start_layer_is_half:
                y = self.layers[0](x)
                y = self.layers[1](y)
                x = self.layers[2](y, x)
                # print(f"After first half {x}")
            # print(f"In {self.layers}")

            for i, layer in enumerate(self.layers):
                if i >= start_id and i <= end_id:
                    x = layer(x)[0]

            if self.end_layer_is_half:
                y = self.layers[-2](x)
                x = self.layers[-1](y,  head_mask=None,  output_attentions=False)[0] + x

            if self.is_last:
                x = self.layernorm(x)
                x = self.classifier(x[:, 0, :])
            gc.collect()
            end = time.time()
        self.total_time +=  (end - start)
        self.total_batch += 1
        print(f"Shard{self.rank} finishes {self.total_batch} microbatch, time is {end -start}, total time is {self.total_time}")
        
        
        ## quantization
        # max_v = torch.max(x)
        # min_v = torch.min(x)
        # print(f"tensor is {x}\n, max value is {max_v}, min value is {min_v}")
        # if self.rank < 1:
        #     print(f"Before quan {x}")
        #     xq = torch.quantize_per_tensor(x, scale = 0.03, zero_point = 0, dtype=torch.quint8).int_repr()
        #     print(f"After quan {xq}")
        #     return xq
        # else:
        #     return x
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
# 'google/vit-base-patch16-224'
# 'google/vit-large-patch16-224'
# 'google/vit-huge-patch14-224-in21k'
model_name= 'google/vit-base-patch16-224'
total_rank = 6
partition =   [0,0.5,0.5,1, 2,4, 5,8, 9,9.5,9.5,11] #[0,2, 3,5, 6,8, 9,11] #[0,2, 3,5, 6,8, 9,11] 
num_batches = 1
batch_size = 5

## random data
# img = torch.randn(3, 384, 384)
## ground truth: Egyptian cat
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open('./images/panda.jpeg')
imgs = [image for i in range(batch_size)]
# ***********************  End  **************************#

_shard_class = [f'TransformerShard{i+1}' for i in range(total_rank)]

rank = 0
shard_class_list = []
for _name in _shard_class:
    if rank == 0:
        globals()[_name] = _create_transformershard(_name, rank, model_name, True, False, partition[2*rank], partition[2*rank+1], True )
    elif rank == total_rank-1:
        globals()[_name] = _create_transformershard(_name, rank, model_name, False, True, partition[2*rank], partition[2*rank+1], True )
    else:
        globals()[_name] = _create_transformershard(_name, rank, model_name, False, False, partition[2*rank], partition[2*rank+1], True )
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
                    y_rref = self.rref_list[i].remote().forward(x_rref)
                elif i == total_rank-1:
                    z_rref = self.rref_list[i].rpc_async().forward(y_rref)
                     
                else:
                    y_rref = self.rref_list[i].remote().forward(y_rref)
                # rref_info = _rref_context_get_debug_info()
                # debug_info = _get_debug_info()
                # print(f"rref info {rref_info}")
                # print(f"debug info {debug_info}")
            
            y_rref.to_here()
            x_rref.to_here()
            del y_rref
            del x_rref
            gc.collect()
            out_futures.append(z_rref)
        return torch.cat(torch.futures.wait_all(out_futures))



#########################################################
#                   Run RPC Processes                   #
#########################################################

latencies = []
throughputs = []

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

def run_master(split_size):
    # put the two model parts on worker1 and worker2 respectively
    print("Run mastering \n")
    work_list = [f"worker{i}" for i in range(total_rank)]
    ## for verification 
    origin_model = ViTForImageClassification.from_pretrained(model_name)
    for si in range(len(split_size)):
        # print(f"Start Calcluate split size {split_size[si]}")
        model =  DistTransformer(model_name, split_size[si], work_list)
        inputs = feature_extractor(images=imgs, return_tensors="pt")
        tik = time.time()
        for i in range(num_batches):
            # generate random inputs and labels       
            outputs = model(inputs['pixel_values'])
            # del outputs
            # gc.collect()
            # print(f"outputs is {outputs}")
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
    print(f"\nBest split size is {split_size[best_choice]}, Execution time is {latencies[best_choice]}, throughput is {throughputs[best_choice]}\n")
    
   


def run_worker(rank, world_size, num_split):

    os.environ['MASTER_ADDR'] = '127.0.0.1' #'10.52.3.175' #'127.0.0.1' # '172.30.0.21'
    os.environ['MASTER_PORT'] = '29501'
    # os.environ["TP_SOCKET_IFNAME"] = "eth0"
    # os.environ["GLOO_SOCKET_IFNAME"] = 'eth0'

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16,rpc_timeout=3000)


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
    world_size = total_rank
    rank=int(sys.argv[1])
    num_split=[5]

    print(f"{model_name}, {batch_size}, {num_split}, split method is {partition}")
    
    tik = time.time()
    run_worker(rank, world_size, num_split)
    tok = time.time()
    print(f"Total program execution time = {tok - tik}")