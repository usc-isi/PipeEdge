"""Distributed pipeline driver application."""
import argparse
import gc
import logging
import os
import time
import numpy as np
from PIL import Image
import requests
import torch
from torch.distributed import rpc
from transformers import BertTokenizer, ViTFeatureExtractor
from transformer import BertDistRpcTransformer, ViTDistRpcTransformer

# torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(filename='runtime.log',level=logging.INFO)


#########################################################
#                   Run RPC Processes                   #
#########################################################
class DistRpcPipeline():
    """The singleton distributed RPC pipeline context manager."""

    def __init__(self, world_size, rank, num_rpc_worker_threads):
        self.world_size = world_size
        self.rank = rank
        self.num_rpc_worker_threads = num_rpc_worker_threads
        self._initialized = False

    def init(self):
        """Initialize the distributed context."""
        assert not self._initialized
        # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=self.num_rpc_worker_threads,
                                                  rpc_timeout=3000)
        rpc.init_rpc(f"worker{self.rank}",
                     rank=self.rank,
                     # backend=rpc.BackendType.PROCESS_GROUP,
                     world_size=self.world_size,
                     rpc_backend_options=options)
        self._initialized = True

    def shutdown(self):
        """Wait for all RPCs to finish and shutdown the distributed context."""
        assert self._initialized
        self._initialized = False
        rpc.shutdown()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *args):
        self.shutdown()

    def forward_model(self, model, inputs, split_size, num_batches=1):
        """Drive the distributed pipeline model with input data."""
        assert self._initialized
        ## for verification
        # origin_model = ViTForImageClassification.from_pretrained(model_name)
        for _ in range(num_batches):
            outputs = model(inputs, split_size=split_size)
            print(f"outputs is {outputs}")
            logging.info(f"outputs is {outputs}")
            del outputs
            gc.collect()
            # predicted_class_idx = outputs[0].argmax(-1).item()
            # print("Predicted class:", origin_model.config.id2label[predicted_class_idx])


if __name__=="__main__":
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Runtime")
    parser.add_argument("rank", type=int, help="the rank for the current node")
    parser.add_argument("worldsize", type=int, help="the world size (the number of nodes)")
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=["google/vit-base-patch16-224",
                                 "google/vit-large-patch16-224",
                                 "google/vit-huge-patch14-224-in21k",
                                 "bert-base-uncased",
                                 "bert-large-uncased"],
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str, help="the model file, if not in working directory")
    parser.add_argument("-pt", "--partition", default="1,48", help="the partition method")
    parser.add_argument("--addr", type=str, default="127.0.0.1", help="ip address for the master node")
    parser.add_argument("--port", type=str, default="29500", help="communication port for the master node")
    parser.add_argument("-s", "--socket-ifname", type=str, default="lo0", help="socket iframe name, use [ifconfig | ipaddress] to check")
    parser.add_argument("-p","--print", type=str, default = "None", choices=["full", "short", "default"], help="print the [full | short] tensor values")
    parser.add_argument("-t", "--threshold", default=1000, type=int, help="total number of array elements which trigger summarization rather than full repr")
    parser.add_argument("-n", "--num-batches", default=1, type=int, help="total number of batches")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-w", "--worker-threads", default=16, type=int, help="the number of worker threads for the communication backend")
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
    logging.info(f"Use device: {device},  # parallel intra nodes threads: {torch.get_num_threads()}, # parallel inter nodes threads: {torch.get_num_interop_threads()}")
    #########################################################
    #                 Configuration for Network             #
    #########################################################
    # *****  Define the World Size and partition Method ******#
    partition =   [int(i) for i in args.partition.split(',')]
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_worker_threads = args.worker_threads
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

    print(f"Model name is {model_name}, Batch size is {batch_size}, Split size is: {num_split}, \n Split method is {partition}, GLOO Threads is {num_worker_threads}")

    os.environ['MASTER_ADDR'] = args.addr # MASTER_ADDR
    os.environ['MASTER_PORT'] = args.port # MASTER_PORT
    os.environ["TP_SOCKET_IFNAME"] = args.socket_ifname # SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = args.socket_ifname # SOCKET_IFNAME

    tik = time.time()
    if model_name in ['bert-base-uncased', 'bert-large-uncased']:
        bert_inputs = np.load("bert_input.npz")['input']
        inputs_sentence = list(bert_inputs[0: batch_size])
        # print(len(inputs_sentence))
        tokenizer = BertTokenizer.from_pretrained(model_name)
        inputs = tokenizer(inputs_sentence, padding=True, truncation=True, return_tensors="pt")['input_ids']
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        inputs = feature_extractor(images=imgs, return_tensors="pt")['pixel_values']

    with DistRpcPipeline(world_size, rank, num_worker_threads) as pipeline:
        if rank == 0:
            print("Run mastering \n")
            # Create model shards on workers (requires distributed context to be initialized)
            if model_name in ['bert-base-uncased', 'bert-large-uncased']:
                model = BertDistRpcTransformer(model_name, model_file, world_size, partition)
            else:
                model = ViTDistRpcTransformer(model_name, model_file, world_size, partition)
            latencies = []
            throughputs = []
            for split_size in num_split:
                # print(f"Start calculate split size {split_size}")
                tik_ss = time.time()
                pipeline.forward_model(model, inputs, split_size, num_batches)
                tok_ss = time.time()
                latency = tok_ss - tik_ss
                throughput = num_batches * batch_size / latency
                latencies.append(latency)
                throughputs.append(throughput)
            best_choice = -1
            best_throughput  = -1
            for i, _ in enumerate(num_split):
                print(f"Split size {num_split[i]}, latency is {latencies[i]}, throughput is {throughputs[i]}")
                logging.info(f"Split size {num_split[i]}, latency is {latencies[i]}, throughput is {throughputs[i]}")
                if throughputs[i] > best_throughput:
                    best_throughput = throughputs[i]
                    best_choice = i
            print("\n---------------- Split output line ----------------")
            print(f"\nBest split size is {num_split[best_choice]}, Execution time is {latencies[best_choice]}, throughput is {throughputs[best_choice]}\n")
            logging.info(f"\nBest split size is {num_split[best_choice]}, Execution time is {latencies[best_choice]}, throughput is {throughputs[best_choice]}\n")

    tok = time.time()
    print(f"Total program execution time = {tok - tik}")
