"""Distributed pipeline driver application."""
import argparse
import gc
import logging
import os
import threading
import time
import numpy as np
from PIL import Image
import requests
import torch
from transformers import BertTokenizer, DeiTFeatureExtractor, ViTFeatureExtractor
from edgepipe.comm.rpc.transformers import (
    BertDistRpcTransformer, DeiTDistRpcTransformer, ViTDistRpcTransformer
)
from edgepipe.models.transformers.bert import BertTransformerShard
from edgepipe.models.transformers.deit import DeiTTransformerShard
from edgepipe.models.transformers.vit import ViTTransformerShard
import model_cfg
from pipeline import DistP2pPipelineStage, DistRpcPipeline

# torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(filename='runtime.log',level=logging.INFO)


class ThreadSafeCounter():
    """Thread-safe counter."""

    def __init__(self, value=0):
        self._value = value
        self._cond = threading.Condition()

    @property
    def value(self):
        """Current counter value."""
        with self._cond:
            val = self._value
            self._cond.notify_all()
        return val

    def add(self, quantity=1):
        """Add to counter atomically."""
        with self._cond:
            self._value += quantity
            self._cond.notify_all()

    def wait_gte(self, threshold):
        """Wait until counter >= threshold."""
        with self._cond:
            while self._value < threshold:
                self._cond.wait()


results_counter = ThreadSafeCounter()

## for verification
# origin_model = ViTForImageClassification.from_pretrained(model_name)
def handle_results(tensors):
    """Process result tensors"""
    print(f"outputs is {tensors}")
    logging.info(f"outputs is {tensors}")
    results_counter.add(len(tensors))
    del tensors
    gc.collect()
    # predicted_class_idx = tensors[0].argmax(-1).item()
    # print("Predicted class:", origin_model.config.id2label[predicted_class_idx])


def profile_split_sizes(split_sizes, num_batches, batch_size, callback):
    """Iterate over split_sizes and num_batches."""
    latencies = []
    throughputs = []
    for split_size in split_sizes:
        # print(f"Start calculate split size {split_size}")
        tik_ss = time.time()
        for _ in range(num_batches):
            callback(split_size)
        tok_ss = time.time()
        latency = tok_ss - tik_ss
        throughput = num_batches * batch_size / latency
        latencies.append(latency)
        throughputs.append(throughput)
    best_choice = -1
    best_throughput = -1
    for i, _ in enumerate(split_sizes):
        print(f"Split size {split_sizes[i]}, latency is {latencies[i]}, throughput is {throughputs[i]}")
        logging.info(f"Split size {split_sizes[i]}, latency is {latencies[i]}, throughput is {throughputs[i]}")
        if throughputs[i] > best_throughput:
            best_throughput = throughputs[i]
            best_choice = i
    print("\n---------------- Split output line ----------------")
    print(f"\nBest split size is {split_sizes[best_choice]}, Execution time is {latencies[best_choice]}, throughput is {throughputs[best_choice]}\n")
    logging.info(f"\nBest split size is {split_sizes[best_choice]}, Execution time is {latencies[best_choice]}, throughput is {throughputs[best_choice]}\n")


if __name__=="__main__":
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Runtime")
    parser.add_argument("rank", type=int, help="the rank for the current node")
    parser.add_argument("worldsize", type=int, help="the world size (the number of nodes)")
    parser.add_argument("-c", "--comm", type=str, default="rpc",
                        choices=["rpc", "p2p"],
                        help="the communication implementation")
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str, help="the model file, if not in working directory")
    parser.add_argument("-pt", "--partition", default="1,48", help="the partition method")
    parser.add_argument("-r", "--rank-order", type=str, default=None,
                        help="comma-delimited list of ranks in desired stage order; default: natural rank order until partitions are assigned")
    parser.add_argument("--addr", type=str, default="127.0.0.1", help="ip address for the master node")
    parser.add_argument("--port", type=str, default="29500", help="communication port for the master node")
    parser.add_argument("-s", "--socket-ifname", type=str, default="lo0", help="socket iframe name, use [ifconfig | ipaddress] to check")
    parser.add_argument("-p","--print", type=str, default = "None", choices=["full", "short", "default"], help="print the [full | short] tensor values")
    parser.add_argument("-t", "--threshold", default=1000, type=int, help="total number of array elements which trigger summarization rather than full repr")
    parser.add_argument("-n", "--num-batches", default=1, type=int, help="total number of batches")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-w", "--worker-threads", default=16, type=int, help="the number of worker threads for the 'rpc' communication backend")
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
    if args.rank_order is None:
        # use natural rank order
        stage_ranks = list(range(len(partition) // 2))
    else:
        stage_ranks = [int(i) for i in args.rank_order.split(',')]
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_worker_threads = args.worker_threads

    # ***********************  End  **************************#
    world_size = args.worldsize
    rank=args.rank
    num_split = [int(i) for i in args.splits.split(',')]
    model_name= args.model_name

    model_file = args.model_file
    if model_file is None:
        model_file = model_cfg.get_model_default_weights_file(model_name)

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
        if model_name in ['facebook/deit-base-distilled-patch16-224',
                          'facebook/deit-small-distilled-patch16-224',
                          'facebook/deit-tiny-distilled-patch16-224']:
            feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
        else:
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        ## random data
        # image = torch.randn(3, 384, 384)
        ## ground truth: Egyptian cat
        URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(URL, stream=True).raw)
        imgs = [image for i in range(batch_size)]
        inputs = feature_extractor(images=imgs, return_tensors="pt")['pixel_values']

    if args.comm == 'p2p':
        # Create model shard locally (doesn't require distributed context to be initialized)
        try:
            stage = stage_ranks.index(rank)
        except ValueError:
            # we're not assigned a stage at this time
            stage = None
        if stage is None:
            model = None
        else:
            layer_start = partition[2*stage]
            layer_end = partition[2*stage+1]
            is_first = layer_start == 1
            is_last = layer_end == model_cfg.get_model_layers(model_name)
            if model_name in ['bert-base-uncased', 'bert-large-uncased']:
                model = BertTransformerShard(stage, model_name, model_file, is_first, is_last,
                                             layer_start, layer_end, True)
            elif model_name in ['facebook/deit-base-distilled-patch16-224',
                                'facebook/deit-small-distilled-patch16-224',
                                'facebook/deit-tiny-distilled-patch16-224']:
                model = DeiTTransformerShard(stage, model_name, model_file, is_first, is_last,
                                             layer_start, layer_end, True)
            else:
                model = ViTTransformerShard(stage, model_name, model_file, is_first, is_last,
                                            layer_start, layer_end, True)
        stop_event = threading.Event()
        CMD_STOP = 100
        def handle_cmd(cmd):
            """Process received commands."""
            if cmd == CMD_STOP:
                print('handle_cmd: stop')
                stop_event.set()
            else:
                print(f'handle_cmd: Unknown command: {cmd}')
        # Initialize the distributed peer-to-peer context
        with DistP2pPipelineStage(world_size, rank, stage_ranks, stage, model, handle_results, handle_cmd) as stage_ctx:
            if stage == 0:
                def drive_pipeline(split_size):
                    """Feed the pipeline."""
                    # this call is asynchronous - wait for results to get end-to-end timings
                    start_count = results_counter.value
                    stage_ctx.enqueue_batch(inputs, split_size)
                    results_counter.wait_gte(start_count + len(inputs))
                profile_split_sizes(num_split, num_batches, batch_size, drive_pipeline)
                # will set stop_event on all other ranks
                stage_ctx.cmd_broadcast(CMD_STOP)
                stop_event.set()
            else:
                stop_event.wait()
    else:
        # Initialize the distributed RPC context
        with DistRpcPipeline(world_size, rank, num_worker_threads) as pipeline:
            if rank == stage_ranks[0]:
                print("Run mastering \n")
                # Create model shards on workers (requires distributed context to be initialized)
                if model_name in ['bert-base-uncased', 'bert-large-uncased']:
                    model = BertDistRpcTransformer(model_name, model_file, stage_ranks, partition)
                elif model_name in ['facebook/deit-base-distilled-patch16-224',
                                    'facebook/deit-small-distilled-patch16-224',
                                    'facebook/deit-tiny-distilled-patch16-224']:
                    model = DeiTDistRpcTransformer(model_name, model_file, stage_ranks, partition)
                else:
                    model = ViTDistRpcTransformer(model_name, model_file, stage_ranks, partition)
                def drive_pipeline(split_size):
                    """Feed the pipeline."""
                    # this call is synchronous - it won't return until it has the results
                    pipeline.forward_model(model, inputs, split_size, handle_results)
                profile_split_sizes(num_split, num_batches, batch_size, drive_pipeline)

    tok = time.time()
    print(f"Total program execution time = {tok - tik}")
