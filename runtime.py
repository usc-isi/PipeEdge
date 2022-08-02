"""Distributed pipeline driver application."""
import argparse
import logging
import os
import queue
import sys
import threading
import time
from typing import Tuple
import numpy as np
from PIL import Image
import requests
import torch
from transformers import BertTokenizer, DeiTFeatureExtractor, ViTFeatureExtractor
from pipeedge.comm.p2p import DistP2pContext
from pipeedge.comm.rpc import DistRpcContext, tensorpipe_rpc_backend_options_factory
from pipeedge import models
from pipeedge.quantization.hook import forward_hook_quant_encode, forward_pre_hook_quant_decode
from pipeedge.sched.scheduler import sched_pipeline
import devices
import model_cfg
import monitoring

# torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(filename='runtime.log', level=logging.DEBUG)
console_hndlr = logging.StreamHandler(sys.stdout)
console_hndlr.setFormatter(logging.Formatter(fmt='%(message)s'))
console_hndlr.setLevel(logging.INFO)
logging.getLogger().addHandler(console_hndlr)
logger = logging.getLogger(__name__)

## ground truth: Egyptian cat
IMG_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'

CMD_STOP = 0
CMD_SCHED = 1


MONITORING_KEY_MODEL = 'shard'
MONITORING_KEY_OUTPUT = 'output'
MONITORING_KEY_QUANT_DECODE = 'quant_decode'
MONITORING_KEY_QUANT_ENCODE = 'quant_encode'
MONITORING_KEY_RECV = 'recv'
MONITORING_KEY_SEND = 'send'

def forward_pre_hook_monitor(_module, _inputs) -> None:
    """Register iteration start."""
    monitoring.iteration_start(MONITORING_KEY_MODEL)

def forward_hook_monitor(module, _inputs, outputs) -> None:
    """Register iteration completion."""
    # Measure work as the microbatch size
    n_items = models.get_microbatch_size(outputs, verify=True)
    # Measure accuracy as the number of layers processed
    n_layers = module.end_layer - module.start_layer + 1
    monitoring.iteration(MONITORING_KEY_MODEL, work=n_items, accuracy=n_layers)

def forward_pre_hook_quant_decode_start(_module, _inputs) -> None:
    """Register quantization decode start."""
    monitoring.iteration_start(MONITORING_KEY_QUANT_DECODE)

def forward_pre_hook_quant_decode_finish(module, inputs) -> None:
    """Register quantization decode completion."""
    # Measure work as the microbatch size, but quantization only does work if quant_bits > 0.
    quant_bits = module.quant_bits.tolist()[0]
    assert isinstance(inputs, tuple)
    n_items = models.get_microbatch_size(inputs[0], verify=True) if quant_bits > 0 else 0
    monitoring.iteration(MONITORING_KEY_QUANT_DECODE, work=n_items, accuracy=quant_bits)

def forward_hook_quant_encode_start(_module, _inputs, _outputs) -> None:
    """Register quantization encode start."""
    monitoring.iteration_start(MONITORING_KEY_QUANT_ENCODE)

def forward_hook_quant_encode_finish(module, inputs, _outputs) -> None:
    """Register quantization encode completion."""
    # Measure work as the microbatch size, but quantization only does work if quant_bits > 0.
    # If output tensors are encoded, they're collapsed s.t. we can't infer the microbatch size.
    # We'll rely on the inputs instead.
    quant_bits = module.quant_bits.tolist()[1]
    assert isinstance(inputs, tuple)
    n_items = models.get_microbatch_size(inputs[0], verify=True) if quant_bits > 0 else 0
    monitoring.iteration(MONITORING_KEY_QUANT_ENCODE, work=n_items, accuracy=quant_bits)

def p2p_pre_hook_monitor(key):
    """Register send/recv start."""
    monitoring.iteration_start(key)

def p2p_post_hook_monitor(tensors, key):
    """Register send/recv completion."""
    assert isinstance(tensors, tuple)
    # Measure work in total data size (MBits), which is a useful metric for data transfers.
    # We don't have enough context here to map tensor structure to a higher-level work concept.
    mbits = sum([t.numel() * t.numpy().dtype.itemsize for t in tensors]) * 8 / 1000000
    # Accuracy has no meaning here.
    monitoring.iteration(key, work=mbits)


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
# from transformers import ViTForImageClassification
# origin_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
def handle_results(tensors: torch.Tensor):
    """Process result tensors"""
    # Monitoring here is intended to measure time between results, NOT end-to-end pipeline latency.
    # Here we use a traditional Application Heartbeats approach, without an explicit start.
    # Therefore, an initial iteration should be reported prior to here to indicate the start (e.g.,
    # when the pipeline is initialized), otherwise metrics from the first report will be lost.
    # Measure work as the microbatch size.
    n_items = models.get_microbatch_size(tensors, verify=True)
    # Measure accuracy as the prediction confidence values.
    # Use softmax to get probability distribution for each tensor.
    prob_sum = torch.nn.functional.softmax(tensors, dim=-1).max(dim=-1)[0].sum().item()
    monitoring.iteration(MONITORING_KEY_OUTPUT, work=n_items, accuracy=prob_sum, safe=False)
    logger.info("outputs is %s", tensors)
    results_counter.add(n_items)
    # predicted_class_idx = tensors[0].argmax(-1).item()
    # logger.info("Predicted class: %s", origin_model.config.id2label[predicted_class_idx])


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
                    logger.error("Scheduling: host not found in hosts list: %s", host)
                    raise
            else:
                try:
                    stage_ranks.append(int(host))
                except ValueError:
                    logger.error("Scheduling: 'hosts' not specified, failed to parse as rank: %s",
                                  host)
                    raise
    return stage_layers, stage_ranks


def _get_default_quant(n_stages):
    return [0] * n_stages


def get_pipeline_sched(world_size, hosts, partition, quant, rank_order, comm, model_name,
                       microbatch_size, s_models_file, s_dev_types_file, s_dev_file):
    """Get the pipeline schedule."""
    if partition:
        # User specified the stage layers
        logger.info("Scheduling: using user-defined partitioning")
        parts = [int(i) for i in partition.split(',')]
        stage_layers = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
        if quant:
            # User specified quantization
            logger.info("Scheduling: using user-defined quantization")
            stage_quant = [int(i) for i in quant.split(',')]
        else:
            # No quantization by default
            logger.info("Scheduling: using default quantization")
            stage_quant = _get_default_quant(len(stage_layers))
        if rank_order:
            # User specified the stage ranks
            logger.info("Scheduling: using user-defined rank ordering")
            stage_ranks = [int(i) for i in rank_order.split(',')]
        else:
            # Use natural rank order
            logger.info("Scheduling: using natural rank ordering")
            stage_ranks = list(range(len(stage_layers)))
    elif quant:
        raise RuntimeError("Must specify partition with quantization")
    elif rank_order:
        raise RuntimeError("Must specify partition with rank stage ordering")
    elif world_size <= 1:
        # Degenerate case: everything runs locally
        logger.info("Scheduling: single-node execution (degenerate case)")
        stage_layers = [(1, model_cfg.get_model_layers(model_name))]
        stage_quant = _get_default_quant(len(stage_layers))
        stage_ranks = [0]
    else:
        # Compute the distributed schedule
        # Set membership constraints: hosts in "s_dev_file" <= "hosts" <= hosts in "world" context
        # Since "hosts" is an implicit (rather than explicit) host-to-rank mapping, we enforce:
        #   "hosts" == hosts in "world" context
        logger.info("Scheduling: using scheduler algorithm")
        if hosts:
            hosts = hosts.split(',')
            if len(hosts) != world_size:
                raise RuntimeError("Specified hosts count != world size")
        # Scheduler assumes 1 processing thread and accounts for those in/out buffers independently.
        # Then for both data receive and send, the design intent for the worst case is:
        # 1 buffer for in-flight data exchanges, 1 buffer for queued (P2P) or blocked (RPC) data.
        # So, let both 'in' and 'out' buffer counts = 2.
        # P2P enforces this with threads for recv/process/send, and queues between the 3 threads.
        # RPC threads each do recv/process/send for a microbatch, but are constrained in number (3).
        sched = sched_pipeline(model_name, 2, 2, microbatch_size,
                               models_file=s_models_file,
                               dev_types_file=s_dev_types_file,
                               dev_file=s_dev_file)
        stage_layers, stage_ranks = parse_yaml_sched(sched, hosts)
        # no quantization support yet for automated scheduling
        stage_quant = _get_default_quant(len(stage_layers))
    logger.info("Scheduling: stage-to-layer mapping: %s", stage_layers)
    logger.info("Scheduling: stage output quantization: %s", stage_quant)
    logger.info("Scheduling: stage-to-rank mapping: %s", stage_ranks)
    return stage_layers, stage_quant, stage_ranks


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


sched_q = queue.Queue()
stop_event = threading.Event()
def handle_cmd(cmd: int, tensors: Tuple[torch.Tensor, ...]):
    """Process received commands."""
    if cmd == CMD_STOP:
        logger.info("handle_cmd: stop")
        stop_event.set()
    elif cmd == CMD_SCHED:
        logger.info("handle_cmd: sched")
        sched_q.put(tuple(t.tolist() for t in tensors))
    else:
        logger.warning("handle_cmd: Unknown command: %s", cmd)


def main():
    """Main function."""
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Runtime",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument("rank", type=int, help="the rank for the current node")
    parser.add_argument("worldsize", type=int, help="the world size (the number of nodes)")
    # Device options
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="compute device type to use, with optional ordinal, "
                             "e.g.: 'cpu', 'cuda', 'cuda:1'")
    # Network configurations
    parser.add_argument("-s", "--socket-ifname", type=str, default="lo0",
                        help="socket interface name, use [ifconfig | ipaddress] to check")
    parser.add_argument("--addr", type=str, default="127.0.0.1",
                        help="ip address for the master node")
    parser.add_argument("--port", type=str, default="29500",
                        help="communication port for the master node")
    # Communication options
    parser.add_argument("-c", "--comm", type=str, default="rpc",
                        choices=["rpc", "p2p"],
                        help="the communication implementation")
    parser.add_argument("-w", "--worker-threads", default=16, type=int,
                        help="the number of worker threads for the 'rpc' communication backend")
    # Model options
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str,
                        help="the model file, if not in working directory")
    # Input options
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")
    # Scheduling options (grouped)
    usched = parser.add_argument_group('User-defined scheduling')
    usched.add_argument("-pt", "--partition", type=str,
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,24,25,48'; "
                             "single-node default: all layers in the model")
    usched.add_argument("-q", "--quant", type=str,
                        help="comma-delimited list of quantization bits to use after each stage")
    usched.add_argument("-r", "--rank-order", type=str, default=None,
                        help="comma-delimited list of ranks in desired stage order; "
                             "default: natural rank order")
    usched.add_argument("-D", "--data-rank", type=int, default=0,
                        help="rank where inputs are loaded and outputs are processed - must be "
                             "the same as stage=0 or not in the stage pipeline")
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

    if args.device is not None:
        devices.DEVICE = torch.device(args.device)
    logger.info("Device: %s", devices.DEVICE)
    if devices.DEVICE is not None and devices.DEVICE.type == 'cuda':
        torch.cuda.init()
    else:
        # Workaround for PyTorch RPC comm initialization automatically enabling CUDA
        # See: https://github.com/pytorch/pytorch/issues/80141
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # parallel_threads = 2
    # torch.set_num_threads(parallel_threads)
    # torch.set_num_interop_threads(parallel_threads)
    logger.debug("# parallel intra nodes threads: %d", torch.get_num_threads())
    logger.debug("# parallel inter nodes threads: %d", torch.get_num_interop_threads())

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
    ubatch_size = args.ubatch_size

    # The master rank computes schedule, then:
    # (1) with comm='p2p': distributes it, then each stage initializes their own stage context
    # (2) with comm='rpc': the rank assigned to stage 0 instantiates the pipeline
    if rank == 0:
        stage_layers, stage_quant, stage_ranks = \
            get_pipeline_sched(world_size, args.hosts, args.partition, args.quant, args.rank_order,
                               args.comm, model_name, ubatch_size,
                               args.sched_models_file, args.sched_dev_types_file,
                               args.sched_dev_file)
        data_rank = args.data_rank
        logger.info("Scheduling: data rank: %s", data_rank)

    monitoring.init(MONITORING_KEY_MODEL, work_type='tensors', acc_type='layers')
    monitoring.add_key(MONITORING_KEY_OUTPUT, work_type='classifications', acc_type='confidence')
    monitoring.add_key(MONITORING_KEY_QUANT_DECODE, work_type='tensors', acc_type='bits')
    monitoring.add_key(MONITORING_KEY_QUANT_ENCODE, work_type='tensors', acc_type='bits')
    tik = time.time()
    if args.comm == 'p2p':
        monitoring.add_key(MONITORING_KEY_RECV, work_type='Mbits')
        monitoring.add_key(MONITORING_KEY_SEND, work_type='Mbits')
        # Initialize the distributed P2P context
        with DistP2pContext(('gloo',), { 'world_size': world_size, 'rank': rank }, handle_cmd) \
            as dist_ctx:
            # Send or receive the schedule
            if rank == 0:
                logger.info("Broadcasting schedule")
                dist_ctx.cmd_broadcast(CMD_SCHED,
                                       (torch.tensor(stage_layers),
                                        torch.tensor(stage_quant),
                                        torch.tensor(stage_ranks),
                                        torch.tensor(data_rank)))
            else:
                logger.info("Waiting for schedule")
                stage_layers, stage_quant, stage_ranks, data_rank = sched_q.get()
                logger.info("Stage layers: %s", stage_layers)
                logger.info("Stage quant: %s", stage_quant)
                logger.info("Stage ranks: %s", stage_ranks)
                logger.info("Data rank: %s", data_rank)
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
                q_bits = torch.tensor((0 if stage == 0 else stage_quant[stage - 1], stage_quant[stage]))
                model.register_buffer('quant_bits', q_bits)
                model.register_forward_hook(devices.forward_hook_to_cpu)
                model.register_forward_hook(forward_hook_monitor)
                if stage != len(stage_ranks) - 1:
                    model.register_forward_hook(forward_hook_quant_encode_start)
                    model.register_forward_hook(forward_hook_quant_encode)
                    model.register_forward_hook(forward_hook_quant_encode_finish)
                if stage != 0:
                    model.register_forward_pre_hook(forward_pre_hook_quant_decode_start)
                    model.register_forward_pre_hook(forward_pre_hook_quant_decode)
                    model.register_forward_pre_hook(forward_pre_hook_quant_decode_finish)
                model.register_forward_pre_hook(forward_pre_hook_monitor)
                model.register_forward_pre_hook(devices.forward_pre_hook_to_device)
            # Initialize the stage context
            with model_cfg.dist_p2p_pipeline_stage_factory(stage_ranks, data_rank, rank, stage,
                                                           model, handle_results) as stage_ctx:
                stage_ctx.register_recv_pre_hook(p2p_pre_hook_monitor, (MONITORING_KEY_RECV,))
                stage_ctx.register_recv_post_hook(p2p_post_hook_monitor, (MONITORING_KEY_RECV,))
                stage_ctx.register_send_pre_hook(p2p_pre_hook_monitor, (MONITORING_KEY_SEND,))
                stage_ctx.register_send_post_hook(p2p_post_hook_monitor, (MONITORING_KEY_SEND,))
                if rank == data_rank:
                    inputs = load_inputs(model_name, batch_size)
                    tik_data = time.time()
                    # start results monitoring - see comments in handle_results
                    monitoring.iteration(MONITORING_KEY_OUTPUT, work=0, accuracy=0, safe=False)
                    # this call is asynchronous - wait for results to get end-to-end timings
                    start_count = results_counter.value
                    stage_ctx.enqueue_batch(inputs, ubatch_size)
                    results_counter.wait_gte(start_count + len(inputs))
                    tok_data = time.time()
                    latency = tok_data - tik_data
                    throughput = batch_size / latency
                    logger.info("Latency is %f, throughput is %f", latency, throughput)
                    # will set stop_event on all other ranks
                    dist_ctx.cmd_broadcast(CMD_STOP)
                    stop_event.set()
                else:
                    stop_event.wait()
    else:
        # Initialize the distributed RPC context
        logger.debug("GLOO Threads: %d", num_worker_threads)
        rpc_opts = tensorpipe_rpc_backend_options_factory(num_worker_threads=num_worker_threads)
        with DistRpcContext((f"worker{rank}",),
                            { 'world_size': world_size,
                              'rank': rank,
                              'rpc_backend_options': rpc_opts }
                           ) as dist_ctx:
            # Send or receive the schedule
            if rank == 0:
                logger.info("Broadcasting schedule")
                dist_ctx.cmd_broadcast(handle_cmd, CMD_SCHED,
                                       (torch.tensor(stage_layers),
                                        torch.tensor(stage_quant),
                                        torch.tensor(stage_ranks),
                                        torch.tensor(data_rank)))
            else:
                logger.info("Waiting for schedule")
                stage_layers, stage_quant, stage_ranks, data_rank = sched_q.get()
                logger.info("Stage layers: %s", stage_layers)
                logger.info("Stage quant: %s", stage_quant)
                logger.info("Stage ranks: %s", stage_ranks)
                logger.info("Data rank: %s", data_rank)
            if rank == data_rank:
                inputs = load_inputs(model_name, batch_size)
                # Create model shards on workers (requires distributed context to be initialized)
                pipeline = model_cfg.dist_rpc_pipeline_factory(model_name, model_file, stage_ranks,
                                                               stage_layers, data_rank,
                                                               handle_results)
                q_bits = [torch.tensor((0 if s == 0 else stage_quant[s - 1], stage_quant[s]))
                          for s in range(len(stage_quant))]
                pipeline.rpc_register_buffer('quant_bits', q_bits)
                pipeline.rpc_register_forward_hook(devices.forward_hook_to_cpu)
                pipeline.rpc_register_forward_hook(forward_hook_monitor)
                pipeline.rpc_register_forward_hook(forward_hook_quant_encode_start, last=False)
                pipeline.rpc_register_forward_hook(forward_hook_quant_encode, last=False)
                pipeline.rpc_register_forward_hook(forward_hook_quant_encode_finish, last=False)
                pipeline.rpc_register_forward_pre_hook(forward_pre_hook_quant_decode_start, first=False)
                pipeline.rpc_register_forward_pre_hook(forward_pre_hook_quant_decode, first=False)
                pipeline.rpc_register_forward_pre_hook(forward_pre_hook_quant_decode_finish, first=False)
                pipeline.rpc_register_forward_pre_hook(forward_pre_hook_monitor)
                pipeline.rpc_register_forward_pre_hook(devices.forward_pre_hook_to_device)
                tik_data = time.time()
                # start results monitoring - see comments in handle_results
                monitoring.iteration(MONITORING_KEY_OUTPUT, work=0, accuracy=0, safe=False)
                # this call is asynchronous - wait for results to get end-to-end timings
                start_count = results_counter.value
                pipeline.enqueue_batch(inputs, ubatch_size)
                results_counter.wait_gte(start_count + len(inputs))
                tok_data = time.time()
                latency = tok_data - tik_data
                throughput = batch_size / latency
                logger.info("Latency is %f, throughput is %f", latency, throughput)
    tok = time.time()
    monitoring.finish()
    logger.info("Total program execution time = %f", tok - tik)


if __name__=="__main__":
    main()
