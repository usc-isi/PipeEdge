"""Distributed pipeline driver application."""
import argparse
import logging
import os
import queue
import sys
import threading
import time
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DeiTFeatureExtractor, ViTFeatureExtractor
from pipeedge.comm.p2p import DistP2pContext
from pipeedge.comm.rpc import DistRpcContext, tensorpipe_rpc_backend_options_factory
from pipeedge import models
from pipeedge.quantization.basic_op import (
    compression_factor, tensor_encode_outerdim, tensor_decode_outerdim
)
from pipeedge.quantization.clamp_op import clamp_banner2019_gelu, clamp_banner2019_laplace
from pipeedge.sched.scheduler import sched_pipeline
import devices
import model_cfg
import monitoring
from utils import data, threads
from utils import quant as quantutil

logger = logging.getLogger(__name__)

## ground truth: Egyptian cat
IMG_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
IMG_LABEL_IDX = 285

CMD_STOP = 0
CMD_SCHED = 1

# A window period defines monitoring and configurability work intervals
WINDOW_SIZE = 10
ENV_WINDOW_SIZE: str = "WINDOW_SIZE"
def get_window_size() -> int:
    """Get the window size."""
    return int(os.getenv(ENV_WINDOW_SIZE, str(WINDOW_SIZE)))


ENV_SEND_CONSTRAINT: str = "SEND_CONSTRAINT"

ENV_ADAPTIVE_QUANT: str = "ADAPTIVE_QUANT"
ADAPTIVE_QUANT_HEURISTIC = "HEURISTIC"
ADAPTIVE_QUANT_HEURISTIC2 = "HEURISTIC2"
ADAPTIVE_QUANT_CONTROLLER = "CONTROLLER"

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
    n_layers = module.shard_config.layer_end - module.shard_config.layer_start + 1
    monitoring.iteration(MONITORING_KEY_MODEL, work=n_items, accuracy=n_layers)

def forward_hook_quant_encode(module, _input_arg, output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
    """encode tensor in the forward hook (after each module)"""
    monitoring.iteration_start(MONITORING_KEY_QUANT_ENCODE)
    if isinstance(output, torch.Tensor):
        output = (output,)
    assert isinstance(output, tuple)
    quant_bit = module.quant_bit.item()
    comm_tuple = []
    for tensor in output:
        assert isinstance(tensor, torch.Tensor)
        if quant_bit > 0:
            clamp = clamp_banner2019_laplace if tensor.min() < 0.2 else clamp_banner2019_gelu
            tensor = clamp(tensor, quant_bit)
        stacked_tensor = tensor_encode_outerdim(tensor, quant_bit)
        comm_tuple += stacked_tensor
    # Measure work as the microbatch size, but quantization only does work if quant_bit > 0.
    n_items = models.get_microbatch_size(output[0], verify=True) if quant_bit > 0 else 0
    monitoring.iteration(MONITORING_KEY_QUANT_ENCODE, work=n_items, accuracy=quant_bit)
    return tuple(comm_tuple)

def forward_pre_hook_quant_decode(_module, input_arg: Tuple[Tuple[torch.Tensor, ...]]):
    """decode tensor in the preforward hook (before each module)"""
    monitoring.iteration_start(MONITORING_KEY_QUANT_DECODE)
    assert isinstance(input_arg, tuple)
    assert len(input_arg) == 1
    # input_tensor: len=5x for x tensors encoded as: comm_tensor, input_shape, scale_factor, shift, quant_bit
    input_tensors = input_arg[0]
    assert isinstance(input_tensors, tuple)
    assert len(input_tensors)%5 == 0
    assert len(input_tensors) >= 5
    quant_bit = input_tensors[4][0].item() # assume the same quantization bitwidth for all items
    forward_tensor = []
    for i in range(len(input_tensors) // 5):
        input_tensor = input_tensors[i*5:i*5+5]
        batched_tensor = tensor_decode_outerdim(input_tensor)
        forward_tensor.append(batched_tensor)
    # Return value(s) should be wrapped in an outer tuple, like input_arg
    # The tuple will be unpacked when forward() is invoked, which must yield a single parameter
    if len(forward_tensor) == 1:
        # assume that the original result was a single tensor rather than a tuple w/ len=1
        outputs = tuple(forward_tensor)
    else:
        outputs = (tuple(forward_tensor),)
    # Measure work as the microbatch size, but quantization only does work if quant_bit > 0.
    n_items = models.get_microbatch_size(outputs, verify=True) if quant_bit > 0 else 0
    monitoring.iteration(MONITORING_KEY_QUANT_DECODE, work=n_items, accuracy=quant_bit)
    return outputs

def forward_hook_set_quant_bandwidth_heuristic(module, _inputs, outputs) -> None:
    """Set quantization bitwidth to satisfy module's `rate_constraint` (requires comm=p2p)."""
    with monitoring.get_locked_context(MONITORING_KEY_SEND) as mctx:
        tag = mctx.get_tag(key=MONITORING_KEY_SEND)
        window_size = mctx.get_window_size(key=MONITORING_KEY_SEND)
        bandwidth = mctx.get_window_perf(key=MONITORING_KEY_SEND)
        send_work = mctx.get_window_work(key=MONITORING_KEY_SEND)
    # Only adapt at window period intervals
    if tag > 0 and tag % window_size == 0:
        target_rate = module.rate_constraint.item()
        if target_rate > 0:
            ubatch_size = models.get_microbatch_size(outputs, verify=True)
            target_time = ubatch_size * window_size / target_rate
        else:
            target_time = float('inf')
        target_datasize = target_time * bandwidth
        quant_bit = module.quant_bit.item()
        if quant_bit > 0:
            compress_ratio = int(send_work * (32 / quant_bit) / target_datasize) + 1
        else:
            compress_ratio = int(send_work / target_datasize) + 1
        if compress_ratio <= 1:
            module.quant_bit = torch.tensor(0)
        elif compress_ratio <= 2:
            module.quant_bit = torch.tensor(16)
        elif compress_ratio <= 4:
            module.quant_bit = torch.tensor(8)
        elif compress_ratio <= 5:
            module.quant_bit = torch.tensor(6)
        elif compress_ratio <= 8:
            module.quant_bit = torch.tensor(4)
        else:
            module.quant_bit = torch.tensor(2)
        logger.info("Adaptive quantization (heuristic): bitwidth=%d", int(module.quant_bit))

def forward_hook_set_quant_bandwidth_heuristic_2(module, _inputs, outputs) -> None:
    """Set quantization bitwidth to satisfy module's `rate_constraint` (requires comm=p2p)."""
    with monitoring.get_locked_context(MONITORING_KEY_SEND) as mctx:
        tag = mctx.get_tag(key=MONITORING_KEY_SEND)
        window_size = mctx.get_window_size(key=MONITORING_KEY_SEND)
        bandwidth = mctx.get_window_perf(key=MONITORING_KEY_SEND)
    # Only adapt at window period intervals
    if tag > 0 and tag % window_size == 0:
        tensors = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
        assert isinstance(tensors, tuple)
        # Rate constraint is per-item, not per-ubatch; rate = 0 -> time = inf
        ubatch_size = models.get_microbatch_size(outputs, verify=True)
        ubatch_time = ubatch_size / module.rate_constraint
        ubatch_mbits = sum(t.numel() * t.numpy().dtype.itemsize for t in tensors) * 8 / 1000000
        src_bit = torch.tensor(tensors[0].numpy().dtype.itemsize * 8)
        quant_bit = quantutil.constrain_max_bitwidth(ubatch_time, ubatch_mbits, bandwidth, src_bit)
        # enforce min bitwidth = 2; quant_bit = src_bit -> quant_bit = 0
        module.quant_bit = max(torch.tensor(2), quant_bit) % src_bit
        logger.info("Adaptive quantization (heuristic2): bitwidth=%d", int(module.quant_bit))

# Largest bitwidths in range [2, 32] with unique discrete compressions
BITWIDTHS = [i for i in range(32, 1, -1)
             if int(compression_factor(i)) > int(compression_factor(i + 1))]
# Cannot keep controllers in a Module instance, so cache by module reference
_MODULE_QUANT_CONTROLLERS = {}
_MODULE_QUANT_CONTROLLERS_LOCK = threading.Lock()
def forward_hook_set_quant_controller(module, _inputs, outputs) -> None:
    """Set quantization bitwidth to to satisfy module's `rate_constraint` (requires comm=p2p)."""
    try:
        bw1 = module.bitwidth1.item()
        bw2 = module.bitwidth2.item()
        bw1_iters = module.bitwidth1_iters.item()
    except AttributeError:
        # only happens in the first window period, before we set all these buffers
        bw1 = bw2 = bw1_iters = 0
    with monitoring.get_locked_context(MONITORING_KEY_SEND) as mctx:
        tag = mctx.get_tag(key=MONITORING_KEY_SEND)
        window_size = mctx.get_window_size(key=MONITORING_KEY_SEND)
        heartrate = mctx.get_window_heartrate(key=MONITORING_KEY_SEND)
    # Only adapt at window period intervals
    if tag > 0 and tag % window_size == 0:
        with _MODULE_QUANT_CONTROLLERS_LOCK:
            if module not in _MODULE_QUANT_CONTROLLERS:
                # quant_bit = 0 -> bw_start = bw_max
                bw_start = module.quant_bit.item() or max(BITWIDTHS)
                _MODULE_QUANT_CONTROLLERS[module] = \
                    quantutil.AdaptiveBitwidthPerformanceController(0, BITWIDTHS, bw_start)
        bw_ctlr = _MODULE_QUANT_CONTROLLERS[module]
        # set the reference value on the controller (usually doesn't change)
        bw_ctlr.reference = module.rate_constraint.item()
        ubatch_size = models.get_microbatch_size(outputs, verify=True)
        send_rate = heartrate * ubatch_size
        bw1, bw2, bw1_iters = bw_ctlr(send_rate, window_size)
        module.register_buffer('bitwidth1', torch.tensor(bw1), persistent=False)
        module.register_buffer('bitwidth2', torch.tensor(bw2), persistent=False)
        logger.info("Adaptive quantization (controller): bitwidth1=%d (iters=%d), bitwidth2=%d",
                    bw1, bw1_iters, bw2)
    bitwidth = bw1 if bw1_iters > 0 else bw2
    # max bitwidth implies no quantization (quant_bit = 0)
    module.quant_bit = torch.tensor(bitwidth % max(BITWIDTHS))
    module.register_buffer('bitwidth1_iters', torch.tensor(max(0, bw1_iters - 1)), persistent=False)


def p2p_pre_hook_monitor(key: str) -> None:
    """Register send/recv start."""
    monitoring.iteration_start(key)

def p2p_post_hook_monitor(tensors: Tuple[torch.Tensor, ...], key: str) -> None:
    """Register send/recv completion."""
    assert isinstance(tensors, tuple)
    # Measure work in total data size (MBits), which is a useful metric for data transfers.
    # We don't have enough context here to map tensor structure to a higher-level work concept.
    mbits = sum(t.numel() * t.numpy().dtype.itemsize for t in tensors) * 8 / 1000000
    # Accuracy has no meaning here.
    monitoring.iteration(key, work=mbits)


results_counter = threads.ThreadSafeCounter()
label_queue = queue.Queue()

def handle_results(tensors: torch.Tensor) -> None:
    """Process result tensors"""
    # Monitoring here is intended to measure time between results, NOT end-to-end pipeline latency.
    # Here we use a traditional Application Heartbeats approach, without an explicit start.
    # Therefore, an initial iteration should be reported prior to here to indicate the start (e.g.,
    # when the pipeline is initialized), otherwise metrics from the first report will be lost.
    # Measure work as the microbatch size.
    n_items = models.get_microbatch_size(tensors, verify=True)
    if label_queue.empty():
        # RPC comm doesn't guarantee microbatch ordering, so it can't reliably check labels.
        # Measure accuracy as the prediction confidence values.
        # Use softmax to get probability distribution for each tensor.
        acc = torch.nn.functional.softmax(tensors, dim=-1).max(dim=-1)[0].sum().item()
    else:
        # Measure accuracy based on label (microbatch ordering must be enforced for correctness).
        ubatch_labels = label_queue.get()
        assert len(tensors) == len(ubatch_labels)
        pred = tensors.argmax(dim=1)
        acc = pred.eq(ubatch_labels).sum().item()
    monitoring.iteration(MONITORING_KEY_OUTPUT, work=n_items, accuracy=acc, safe=False)
    logger.info("outputs is %s", tensors)
    results_counter.add(n_items)


def parse_yaml_sched(sched: List[dict], hosts: Optional[List[str]]) -> \
    Tuple[List[Tuple[int, int]], List[int]]:
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


def get_pipeline_sched(world_size: int, hosts: Optional[List[str]],
                       partition: Optional[List[Tuple[int, int]]], quant: Optional[List[int]],
                       rank_order: Optional[List[int]], model_name: str, microbatch_size: int,
                       s_models_file: Optional[str], s_dev_types_file: Optional[str],
                       s_dev_file: Optional[str]) -> \
    Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Get the pipeline schedule: `stage_layers`, `stage_quant`, and `stage_ranks`."""
    def _get_default_quant(n_stages: int) -> List[int]:
        return [0] * n_stages
    if partition:
        # User specified the stage layers
        logger.info("Scheduling: using user-defined partitioning")
        stage_layers = partition
        if quant:
            # User specified quantization
            logger.info("Scheduling: using user-defined quantization")
            stage_quant = quant
        else:
            # No quantization by default
            logger.info("Scheduling: using default quantization")
            stage_quant = _get_default_quant(len(stage_layers))
        if rank_order:
            # User specified the stage ranks
            logger.info("Scheduling: using user-defined rank ordering")
            stage_ranks = rank_order
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


def load_dataset(dataset_cfg: dict, model_name: str, batch_size: int, ubatch_size: int) -> Dataset:
    """Load inputs based on model."""
    def _get_feature_extractor():
        if model_name in ['facebook/deit-base-distilled-patch16-224',
                          'facebook/deit-small-distilled-patch16-224',
                          'facebook/deit-tiny-distilled-patch16-224']:
            feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
        else:
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        return feature_extractor
    dataset_name = dataset_cfg['name']
    dataset_root = dataset_cfg['root']
    dataset_split = dataset_cfg['split']
    indices = dataset_cfg['indices']
    dataset_shuffle = dataset_cfg['shuffle']
    if dataset_name == 'CoLA':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = data.load_dataset_glue(tokenizer, 'cola', dataset_split, ubatch_size)
        dataset = data.load_dataset_subset(dataset, indices=indices, max_size=batch_size,
                                           shuffle=dataset_shuffle)
    elif dataset_name == 'ImageNet':
        if dataset_root is None:
            dataset_root = 'ImageNet'
            logging.info("Dataset root not set, assuming: %s", dataset_root)
        feature_extractor = _get_feature_extractor()
        dataset = data.load_dataset_imagenet(feature_extractor, dataset_root, split=dataset_split)
        dataset = data.load_dataset_subset(dataset, indices=indices, max_size=batch_size,
                                           shuffle=dataset_shuffle)
    elif model_name in ['bert-base-uncased', 'bert-large-uncased',
                        'textattack/bert-base-uncased-CoLA']:
        with np.load("bert_input.npz") as bert_inputs:
            inputs_sentence = list(bert_inputs['input'][:batch_size])
            labels = torch.tensor(bert_inputs['label'][:batch_size])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(inputs_sentence, padding=True, truncation=True, return_tensors="pt")['input_ids']
        dataset = data.RolloverTensorDataset(batch_size, inputs, labels)
    else:
        feature_extractor = _get_feature_extractor()
        ## random data
        # image = torch.randn(3, 384, 384)
        image = Image.open(requests.get(IMG_URL, stream=True, timeout=60).raw)
        inputs = feature_extractor(images=[image], return_tensors="pt")['pixel_values']
        dataset = data.RolloverTensorDataset(batch_size, inputs, torch.tensor([IMG_LABEL_IDX]))
    return dataset


sched_q = queue.Queue()
stop_event = threading.Event()
def handle_cmd(cmd: int, tensors: Tuple[torch.Tensor, ...]) -> None:
    """Process received commands."""
    if cmd == CMD_STOP:
        logger.info("handle_cmd: stop")
        stop_event.set()
    elif cmd == CMD_SCHED:
        logger.info("handle_cmd: sched")
        sched_q.put(tuple(t.tolist() for t in tensors))
    else:
        logger.warning("handle_cmd: Unknown command: %s", cmd)


def run_pipeline_p2p(world_size: int, rank: int, model_name: str, model_file: Optional[str],
                     batch_size: int, ubatch_size: int, partition: Optional[List[Tuple[int, int]]],
                     quant: Optional[List[int]], rank_order: Optional[List[int]], data_rank: int,
                     hosts: Optional[List[str]], dataset_cfg: dict,
                     sched_models_file: Optional[str], sched_dev_types_file: Optional[str],
                     sched_dev_file: Optional[str]) -> None:
    """Run the pipeline using P2P communication."""
    monitoring.init(MONITORING_KEY_MODEL, get_window_size(), work_type='tensors', acc_type='layers')
    monitoring.add_key(MONITORING_KEY_OUTPUT, work_type='classifications', acc_type='correct')
    monitoring.add_key(MONITORING_KEY_QUANT_DECODE, work_type='tensors', acc_type='bits')
    monitoring.add_key(MONITORING_KEY_QUANT_ENCODE, work_type='tensors', acc_type='bits')
    monitoring.add_key(MONITORING_KEY_RECV, work_type='Mbits')
    monitoring.add_key(MONITORING_KEY_SEND, work_type='Mbits')
    with DistP2pContext(('gloo',), { 'world_size': world_size, 'rank': rank }, handle_cmd) \
        as dist_ctx:
        # Send or receive the schedule
        if rank == 0:
            stage_layers, stage_quant, stage_ranks = \
                get_pipeline_sched(world_size, hosts, partition, quant, rank_order,
                                   model_name, ubatch_size, sched_models_file,
                                   sched_dev_types_file, sched_dev_file)
            logger.info("Scheduling: data rank: %s", data_rank)
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
            model.register_buffer('quant_bit', torch.tensor(stage_quant[stage]), persistent=False)
            send_constraint = float(os.getenv(ENV_SEND_CONSTRAINT, str(0)))
            model.register_buffer('rate_constraint', torch.tensor(send_constraint),
                                  persistent=False)
            model.register_forward_hook(devices.forward_hook_to_cpu)
            model.register_forward_hook(forward_hook_monitor)
            if stage != len(stage_ranks) - 1:
                quant_impl = os.getenv(ENV_ADAPTIVE_QUANT)
                if quant_impl == ADAPTIVE_QUANT_CONTROLLER:
                    model.register_forward_hook(forward_hook_set_quant_controller)
                elif quant_impl == ADAPTIVE_QUANT_HEURISTIC:
                    model.register_forward_hook(forward_hook_set_quant_bandwidth_heuristic)
                elif quant_impl == ADAPTIVE_QUANT_HEURISTIC2:
                    model.register_forward_hook(forward_hook_set_quant_bandwidth_heuristic_2)
                model.register_forward_hook(forward_hook_quant_encode)
            if stage != 0:
                model.register_forward_pre_hook(forward_pre_hook_quant_decode)
            model.register_forward_pre_hook(forward_pre_hook_monitor)
            model.register_forward_pre_hook(devices.forward_pre_hook_to_device)
        # Initialize the stage context
        with model_cfg.dist_p2p_pipeline_stage_factory(stage_ranks, data_rank, rank, stage, model,
                                                       handle_results) as stage_ctx:
            stage_ctx.register_recv_pre_hook(p2p_pre_hook_monitor, (MONITORING_KEY_RECV,))
            stage_ctx.register_recv_post_hook(p2p_post_hook_monitor, (MONITORING_KEY_RECV,))
            stage_ctx.register_send_pre_hook(p2p_pre_hook_monitor, (MONITORING_KEY_SEND,))
            stage_ctx.register_send_post_hook(p2p_post_hook_monitor, (MONITORING_KEY_SEND,))
            if rank == data_rank:
                dataset = load_dataset(dataset_cfg, model_name, batch_size, ubatch_size)
                data_loader = DataLoader(dataset, batch_size=ubatch_size)
                tik_data = time.time()
                # start results monitoring - see comments in handle_results
                monitoring.iteration(MONITORING_KEY_OUTPUT, work=0, accuracy=0, safe=False)
                # this call is asynchronous - wait for results to get end-to-end timings
                start_count = results_counter.value
                for ubatch, ubatch_labels in data_loader:
                    label_queue.put(ubatch_labels)
                    stage_ctx.enqueue_tensor(ubatch)
                results_counter.wait_gte(start_count + len(dataset))
                tok_data = time.time()
                latency = tok_data - tik_data
                throughput = batch_size / latency
                logger.info("Latency is %f, throughput is %f", latency, throughput)
                # will set stop_event on all other ranks
                dist_ctx.cmd_broadcast(CMD_STOP)
                stop_event.set()
            else:
                stop_event.wait()
    monitoring.finish()


def run_pipeline_rpc(world_size: int, rank: int, model_name: str, model_file: Optional[str],
                     batch_size: int, ubatch_size: int, partition: Optional[List[Tuple[int, int]]],
                     quant: Optional[List[int]], rank_order: Optional[List[int]], data_rank: int,
                     hosts: Optional[List[str]], dataset_cfg: dict,
                     sched_models_file: Optional[str], sched_dev_types_file: Optional[str],
                     sched_dev_file: Optional[str], rpc_num_worker_threads: int) -> None:
    """Run the pipeline using RPC communication."""
    monitoring.init(MONITORING_KEY_MODEL, get_window_size(), work_type='tensors', acc_type='layers')
    monitoring.add_key(MONITORING_KEY_OUTPUT, work_type='classifications', acc_type='confidence')
    monitoring.add_key(MONITORING_KEY_QUANT_DECODE, work_type='tensors', acc_type='bits')
    monitoring.add_key(MONITORING_KEY_QUANT_ENCODE, work_type='tensors', acc_type='bits')
    logger.debug("GLOO Threads: %d", rpc_num_worker_threads)
    rpc_opts = tensorpipe_rpc_backend_options_factory(num_worker_threads=rpc_num_worker_threads)
    with DistRpcContext((f"worker{rank}",),
                        { 'world_size': world_size,
                          'rank': rank,
                          'rpc_backend_options': rpc_opts }
                       ) as dist_ctx:
        # Send or receive the schedule
        if rank == 0:
            stage_layers, stage_quant, stage_ranks = \
                get_pipeline_sched(world_size, hosts, partition, quant, rank_order,
                                   model_name, ubatch_size, sched_models_file,
                                   sched_dev_types_file, sched_dev_file)
            logger.info("Scheduling: data rank: %s", data_rank)
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
            dataset = load_dataset(dataset_cfg, model_name, batch_size, ubatch_size)
            data_loader = DataLoader(dataset, batch_size=ubatch_size)
            # Create model shards on workers (requires distributed context to be initialized)
            pipeline = model_cfg.dist_rpc_pipeline_factory(model_name, model_file, stage_ranks,
                                                           stage_layers, data_rank, handle_results)
            pipeline.rpc_register_buffer('quant_bit', [torch.tensor(q) for q in stage_quant],
                                         persistent=False)
            pipeline.rpc_register_forward_hook(devices.forward_hook_to_cpu)
            pipeline.rpc_register_forward_hook(forward_hook_monitor)
            pipeline.rpc_register_forward_hook(forward_hook_quant_encode, last=False)
            pipeline.rpc_register_forward_pre_hook(forward_pre_hook_quant_decode, first=False)
            pipeline.rpc_register_forward_pre_hook(forward_pre_hook_monitor)
            pipeline.rpc_register_forward_pre_hook(devices.forward_pre_hook_to_device)
            tik_data = time.time()
            # start results monitoring - see comments in handle_results
            monitoring.iteration(MONITORING_KEY_OUTPUT, work=0, accuracy=0, safe=False)
            # this call is asynchronous - wait for results to get end-to-end timings
            start_count = results_counter.value
            for ubatch, _ in data_loader:
                pipeline.enqueue_tensor(ubatch)
            results_counter.wait_gte(start_count + len(dataset))
            tok_data = time.time()
            latency = tok_data - tik_data
            throughput = batch_size / latency
            logger.info("Latency is %f, throughput is %f", latency, throughput)
    monitoring.finish()


def init_env(device: Optional[str], net_addr: str, net_port: int, net_ifname: str) -> None:
    """Initialize the PyTorch environment."""
    # Device
    if device is not None:
        devices.DEVICE = torch.device(device)
    if devices.DEVICE is not None and devices.DEVICE.type == 'cuda':
        torch.cuda.init()
    else:
        # Workaround for PyTorch RPC comm initialization automatically enabling CUDA
        # See: https://github.com/pytorch/pytorch/issues/80141
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Parallelism
    # parallel_threads = 2
    # torch.set_num_threads(parallel_threads)
    # torch.set_num_interop_threads(parallel_threads)

    # Network
    os.environ['MASTER_ADDR'] = net_addr # MASTER_ADDR
    os.environ['MASTER_PORT'] = str(net_port) # MASTER_PORT
    os.environ["TP_SOCKET_IFNAME"] = net_ifname # SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = net_ifname # SOCKET_IFNAME


def main() -> None:
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
    parser.add_argument("--port", type=int, default=29500,
                        help="communication port for the master node")
    # Communication options
    parser.add_argument("-c", "--comm", type=str, default="p2p", choices=["p2p", "rpc"],
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
    # Dataset options
    dset = parser.add_argument_group('Dataset arguments')
    dset.add_argument("--dataset-name", type=str, choices=['CoLA', 'ImageNet'],
                      help="dataset to use")
    dset.add_argument("--dataset-root", type=str,
                      help="dataset root directory (e.g., for 'ImageNet', must contain "
                           "'ILSVRC2012_devkit_t12.tar.gz' and at least one of: "
                           "'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'")
    dset.add_argument("--dataset-split", default='train', type=str,
                      help="dataset split (depends on dataset), e.g.: train, val, validation, test")
    dset.add_argument("--dataset-indices-file", default=None, type=str,
                      help="PyTorch or NumPy file with precomputed dataset index sequence")
    dset.add_argument("--dataset-shuffle", type=bool, nargs='?', const=True, default=False,
                      help="dataset shuffle")
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

    if args.dataset_indices_file is None:
        indices = None
    elif args.dataset_indices_file.endswith('.pt'):
        indices = torch.load(args.dataset_indices_file)
    else:
        indices = np.load(args.dataset_indices_file)
    dataset_cfg = {
        'name': args.dataset_name,
        'root': args.dataset_root,
        'split': args.dataset_split,
        'indices': indices,
        'shuffle': args.dataset_shuffle,
    }

    if args.partition is None:
        partition = None
    else:
        parts = [int(i) for i in args.partition.split(',')]
        assert len(parts) % 2 == 0
        partition = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
    quant = None if args.quant is None else [int(i) for i in args.quant.split(',')]
    rank_order = None if args.rank_order is None else [int(i) for i in args.rank_order.split(',')]
    hosts = None if args.hosts is None else args.hosts.split(',')

    tik = time.time()
    init_env(args.device, args.addr, args.port, args.socket_ifname)
    logger.info("Device: %s", devices.DEVICE)
    logger.debug("# parallel intra nodes threads: %d", torch.get_num_threads())
    logger.debug("# parallel inter nodes threads: %d", torch.get_num_interop_threads())
    if args.comm == 'p2p':
        run_pipeline_p2p(args.worldsize, args.rank, args.model_name, args.model_file,
                         args.batch_size, args.ubatch_size, partition, quant, rank_order,
                         args.data_rank, hosts, dataset_cfg, args.sched_models_file,
                         args.sched_dev_types_file, args.sched_dev_file)
    else:
        run_pipeline_rpc(args.worldsize, args.rank, args.model_name, args.model_file,
                         args.batch_size, args.ubatch_size, partition, quant, rank_order,
                         args.data_rank, hosts, dataset_cfg, args.sched_models_file,
                         args.sched_dev_types_file, args.sched_dev_file, args.worker_threads)
    tok = time.time()
    logger.info("Total program execution time = %f", tok - tik)


if __name__=="__main__":
    logging.basicConfig(filename='runtime.log', level=logging.DEBUG)
    console_hndlr = logging.StreamHandler(sys.stdout)
    console_hndlr.setFormatter(logging.Formatter(fmt='%(message)s'))
    console_hndlr.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_hndlr)
    main()
