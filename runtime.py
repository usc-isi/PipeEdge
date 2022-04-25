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
import model_cfg
from pipeline import DistP2pPipelineStage, DistRpcPipeline

# torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(filename='runtime.log',level=logging.INFO)

## ground truth: Egyptian cat
IMG_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'

CMD_STOP = 100


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


def main():
    """Main function."""
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Runtime",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("rank", type=int, help="the rank for the current node")
    parser.add_argument("worldsize", type=int, help="the world size (the number of nodes)")
    parser.add_argument("-c", "--comm", type=str, default="rpc",
                        choices=["rpc", "p2p"],
                        help="the communication implementation")
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str, help="the model file, if not in working directory")
    parser.add_argument("-pt", "--partition", type=str,
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,48'; default: all layers in the model")
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
    model_name = args.model_name
    if args.partition is None:
        stage_layers = [(1, model_cfg.get_model_layers(model_name))]
    else:
        parts = [int(i) for i in args.partition.split(',')]
        stage_layers = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
    if args.rank_order is None:
        # use natural rank order
        stage_ranks = list(range(len(stage_layers)))
    else:
        stage_ranks = [int(i) for i in args.rank_order.split(',')]
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_worker_threads = args.worker_threads

    # ***********************  End  **************************#
    world_size = args.worldsize
    rank=args.rank
    num_split = [int(i) for i in args.splits.split(',')]

    model_file = args.model_file
    if model_file is None:
        model_file = model_cfg.get_model_default_weights_file(model_name)

    print(f"Model name is {model_name}, Batch size is {batch_size}, Split size is: {num_split},")
    print(f"Split method is {stage_layers}, GLOO Threads is {num_worker_threads}")

    os.environ['MASTER_ADDR'] = args.addr # MASTER_ADDR
    os.environ['MASTER_PORT'] = args.port # MASTER_PORT
    os.environ["TP_SOCKET_IFNAME"] = args.socket_ifname # SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = args.socket_ifname # SOCKET_IFNAME

    tik = time.time()
    if model_name in ['bert-base-uncased', 'bert-large-uncased']:
        with np.load("bert_input.npz") as bert_inputs:
            inputs_sentence = list(bert_inputs['input'][0: batch_size])
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
        image = Image.open(requests.get(IMG_URL, stream=True).raw)
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
            model = model_cfg.module_shard_factory(model_name, model_file, stage_layers[stage][0],
                                                   stage_layers[stage][1], stage)
        stop_event = threading.Event()
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
                model = model_cfg.dist_rpc_module_factory(model_name, model_file, stage_ranks, stage_layers)
                def drive_pipeline(split_size):
                    """Feed the pipeline."""
                    # this call is synchronous - it won't return until it has the results
                    pipeline.forward_model(model, inputs, split_size, handle_results)
                profile_split_sizes(num_split, num_batches, batch_size, drive_pipeline)

    tok = time.time()
    print(f"Total program execution time = {tok - tik}")


if __name__=="__main__":
    main()
