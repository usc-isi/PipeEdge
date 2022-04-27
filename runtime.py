"""Distributed pipeline driver application."""
import argparse
import gc
import logging
import os
import queue
import threading
import time
import numpy as np
from PIL import Image
import requests
import torch
from transformers import BertTokenizer, DeiTFeatureExtractor, ViTFeatureExtractor
from edgepipe.sched.scheduler import sched_pipeline
import model_cfg
from pipeline import DistP2pContext, DistP2pPipelineStage, DistRpcPipeline

# torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(filename='runtime.log',level=logging.INFO)

## ground truth: Egyptian cat
IMG_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'

CMD_STOP = 0
CMD_SCHED = 1


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


def parse_yaml_sched(sched, hosts):
    """Parse the YAML schedule into `stage_layers` and `stage_ranks`."""
    # list of single-entry maps
    assert isinstance(sched, list)
    if len(sched) == 0:
        raise RuntimeError("No viable schedule found")
    stage_layers = []
    stage_ranks = []
    for stage in sched:
        # dict with a single mapping: host -> [layer_start, layer_end]
        assert len(stage) == 1
        for host, layers in stage.items():
            assert len(layers) == 2
            stage_layers.append((int(layers[0]), int(layers[1])))
            if hosts:
                try:
                    stage_ranks.append(hosts.index(host))
                except ValueError:
                    print(f"Scheduling: host not found in hosts list: {host}")
                    raise
            else:
                try:
                    stage_ranks.append(int(host))
                except ValueError:
                    print(f"Scheduling: 'hosts' not specified, failed to parse as rank: {host}")
                    raise
    return stage_layers, stage_ranks


def get_pipeline_sched(world_size, hosts, partition, rank_order, comm, model_name, batch_size,
                       s_models_file, s_dev_types_file, s_dev_file):
    """Get the pipeline schedule."""
    if partition:
        # User specified the stage layers
        print("Scheduling: using user-defined partitioning")
        parts = [int(i) for i in partition.split(',')]
        stage_layers = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
        if rank_order:
            # User specified the stage ranks
            print("Scheduling: using user-defined rank ordering")
            stage_ranks = [int(i) for i in rank_order.split(',')]
        else:
            # Use natural rank order
            print("Scheduling: using natural rank ordering")
            stage_ranks = list(range(len(stage_layers)))
    elif rank_order:
        raise RuntimeError("Must specify partition with rank stage ordering")
    elif world_size <= 1:
        # Degenerate case: everything runs locally
        print("Scheduling: single-node execution (degenerate case)")
        stage_layers = [(1, model_cfg.get_model_layers(model_name))]
        stage_ranks = [0]
    else:
        # Compute the distributed schedule
        # Set membership constraints: hosts in "s_dev_file" <= "hosts" <= hosts in "world" context
        # Since "hosts" is an implicit (rather than explicit) host-to-rank mapping, we enforce:
        #   "hosts" == hosts in "world" context
        print("Scheduling: using scheduler algorithm")
        if hosts:
            hosts = hosts.split(',')
            if len(hosts) != world_size:
                raise RuntimeError("Specified hosts count != world size")
        # comm='rpc' is _presumed_ to not use additional buffers (or queues), so set buffers=1
        buffers = 2 if comm == 'p2p' else 1
        sched = sched_pipeline(model_name, buffers, buffers, batch_size,
                               models_file=s_models_file,
                               dev_types_file=s_dev_types_file,
                               dev_file=s_dev_file)
        stage_layers, stage_ranks = parse_yaml_sched(sched, hosts)
    print(f"Scheduling: stage-to-layer mapping: {stage_layers}")
    print(f"Scheduling: stage-to-rank mapping: {stage_ranks}")
    return stage_layers, stage_ranks


def load_inputs(model_name, batch_size):
    """Load inputs based on model."""
    if model_name in ['bert-base-uncased', 'bert-large-uncased']:
        with np.load("bert_input.npz") as bert_inputs:
            inputs_sentence = list(bert_inputs['input'][0: batch_size])
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
    return inputs


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
    parser.add_argument("--addr", type=str, default="127.0.0.1", help="ip address for the master node")
    parser.add_argument("--port", type=str, default="29500", help="communication port for the master node")
    parser.add_argument("-s", "--socket-ifname", type=str, default="lo0", help="socket iframe name, use [ifconfig | ipaddress] to check")
    parser.add_argument("-p","--print", type=str, default = "None", choices=["full", "short", "default"], help="print the [full | short] tensor values")
    parser.add_argument("-t", "--threshold", default=1000, type=int, help="total number of array elements which trigger summarization rather than full repr")
    parser.add_argument("-n", "--num-batches", default=1, type=int, help="total number of batches")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-w", "--worker-threads", default=16, type=int, help="the number of worker threads for the 'rpc' communication backend")
    parser.add_argument("-sp", "--splits", default="8", help="the list of microbatch size")
    usched = parser.add_argument_group('User-defined scheduling')
    usched.add_argument("-pt", "--partition", type=str,
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,24,25,48'; "
                             "single-node default: all layers in the model")
    usched.add_argument("-r", "--rank-order", type=str, default=None,
                        help="comma-delimited list of ranks in desired stage order; "
                             "default: natural rank order")
    asched = parser.add_argument_group('Automated scheduling')
    # This ordered list is a poor man's approach to map the hosts used in the scheduler's output
    # to the rank values needed by PyTorch's distributed framework.
    # Conceptually, we *could* require that the devices file refer to ranks rather than hosts.
    # However, that would:
    # (1) force our distributed implementation details on the more generic scheduler YAML file
    # (2) require the YAML file to be tailored to the current rank-to-host deployment scenario,
    #     but rank-to-host selection is an entirely arbitrary decision on the user's part.
    # With this, the user sets that mapping entirely from the command line; files remain unchanged.
    # All that said, we'll *try* to treat scheduler hosts output as ranks if this isn't specified.
    asched.add_argument("-H", "--hosts", type=str,
                        help="comma-delimited list of hosts in rank order; "
                             "required for automated scheduling")
    asched.add_argument("-sm", "--sched-models-file", default=None, type=str,
                        help="models YAML file for scheduler, e.g., models.yml")
    asched.add_argument("-sdt", "--sched-dev-types-file", default=None, type=str,
                        help="device types YAML file for scheduler, e.g., device_types.yml")
    asched.add_argument("-sd", "--sched-dev-file", default=None, type=str,
                        help="devices YAML file for scheduler, e.g., devices.yml; "
                             "devices in file should satisfy set membership constraint: "
                             "devices <= HOSTS")
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
    world_size = args.worldsize
    rank = args.rank
    os.environ['MASTER_ADDR'] = args.addr # MASTER_ADDR
    os.environ['MASTER_PORT'] = args.port # MASTER_PORT
    os.environ["TP_SOCKET_IFNAME"] = args.socket_ifname # SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = args.socket_ifname # SOCKET_IFNAME
    num_worker_threads = args.worker_threads
    # ***********************  End  **************************#

    model_name = args.model_name
    model_file = args.model_file
    if model_file is None:
        model_file = model_cfg.get_model_default_weights_file(model_name)
    batch_size = args.batch_size
    num_batches = args.num_batches
    num_split = [int(i) for i in args.splits.split(',')]

    # The master rank computes schedule, then:
    # (1) with comm='p2p': distributes it, then each stage initializes their own stage context
    # (2) with comm='rpc': the rank assigned to stage 0 instantiates the pipeline
    if rank == 0:
        stage_layers, stage_ranks = get_pipeline_sched(world_size, args.hosts, args.partition, args.rank_order,
                                                       args.comm, model_name, batch_size,
                                                       args.sched_models_file, args.sched_dev_types_file,
                                                       args.sched_dev_file)
        print(f"Stage layers: {stage_layers}")
        print(f"Stage ranks: {stage_ranks}")

    tik = time.time()
    if args.comm == 'p2p':
        stop_event = threading.Event()
        sched_q = queue.Queue()
        def handle_cmd(cmd, tensors):
            """Process received commands."""
            if cmd == CMD_STOP:
                print('handle_cmd: stop')
                stop_event.set()
            elif cmd == CMD_SCHED:
                print('handle_cmd: sched')
                assert isinstance(tensors, tuple)
                assert len(tensors) == 2 # stage_layers, stage_ranks
                sched_q.put((tensors[0].tolist(), tensors[1].tolist()))
            else:
                print(f'handle_cmd: Unknown command: {cmd}')
        # Initialize the distributed P2P context
        with DistP2pContext(world_size, rank, handle_cmd) as dist_ctx:
            # Send or receive the schedule
            if rank == 0:
                print("Broadcasting schedule")
                dist_ctx.cmd_broadcast(CMD_SCHED,
                                       (torch.tensor(stage_layers), torch.tensor(stage_ranks)))
            else:
                print("Waiting for schedule")
                stage_layers, stage_ranks = sched_q.get()
                print(f"Stage layers: {stage_layers}")
                print(f"Stage ranks: {stage_ranks}")
            # Create model shard locally
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
            # Initialize the stage context
            with DistP2pPipelineStage(stage_ranks, stage, model, handle_results) as stage_ctx:
                if stage == 0:
                    inputs = load_inputs(model_name, batch_size)
                    def drive_pipeline(split_size):
                        """Feed the pipeline."""
                        # this call is asynchronous - wait for results to get end-to-end timings
                        start_count = results_counter.value
                        stage_ctx.enqueue_batch(inputs, split_size)
                        results_counter.wait_gte(start_count + len(inputs))
                    profile_split_sizes(num_split, num_batches, batch_size, drive_pipeline)
                    # will set stop_event on all other ranks
                    dist_ctx.cmd_broadcast(CMD_STOP)
                    stop_event.set()
                else:
                    stop_event.wait()
    else:
        # Initialize the distributed RPC context
        # print(f"GLOO Threads: {num_worker_threads}")
        with DistRpcPipeline(world_size, rank, num_worker_threads) as pipeline:
            if rank == stage_ranks[0]:
                inputs = load_inputs(model_name, batch_size)
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
