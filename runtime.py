"""Distributed pipeline driver application."""
import argparse
import logging
import os
import queue
import random
import sys
import threading
import time
from math import sqrt, floor, ceil
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import requests
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.datasets import ImageNet
from transformers import BertTokenizer, DeiTFeatureExtractor, ViTFeatureExtractor
from PyQt5.QtCore import QRunnable, pyqtSlot, pyqtSignal, QObject, QThreadPool
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QGridLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,)
import pyqtgraph as pg
from pipeedge.comm.p2p import DistP2pContext
from pipeedge.comm.rpc import DistRpcContext, tensorpipe_rpc_backend_options_factory
from pipeedge import models
from pipeedge.quantization.basic_op import tensor_encode_outerdim, tensor_decode_outerdim
from pipeedge.quantization.clamp_op import clamp_banner2019_gelu, clamp_banner2019_laplace
from pipeedge.sched.scheduler import sched_pipeline
import devices
import model_cfg
import monitoring


logger = logging.getLogger(__name__)

## ground truth: Egyptian cat
IMG_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
IMG_LABEL_IDX = 285

CMD_STOP = 0
CMD_SCHED = 1


MONITORING_KEY_MODEL = 'shard'
MONITORING_KEY_OUTPUT = 'output'
MONITORING_KEY_QUANT_DECODE = 'quant_decode'
MONITORING_KEY_QUANT_ENCODE = 'quant_encode'
MONITORING_KEY_RECV = 'recv'
MONITORING_KEY_SEND = 'send'

PLOT_DATAPOINT_NUMBER = 100
PLOT_TIME_INTERVAL = 0.5
TARGET_SEND_RATE = 12.0

monitoring_output_perf = []
monitoring_output_acc = []
monitoring_send_perf = []
monitoring_send_rate = []
monitoring_quant_bit = []
fig_titles = [ r"Pipeline Performance",
               r"Pipeline Accuracy",
               r"Send Bandwidth",
               r"Send Performance",
               r"Quant bitwidth"]
fig_yaxi_label = [ r"Images/sec",
               r"Top-1 correct %",
               r"Mbps",
               r"Images/sec",
               r"Bitwidth"]

def fetch_data_from_runtime():
    return (
        monitoring_output_perf,
        monitoring_output_acc,
        monitoring_send_perf,
        monitoring_send_rate,
        monitoring_quant_bit
    )

def forward_hook_bandwidth_detect(module, _inputs, outputs) -> None:
    with monitoring._monitor_ctx_lock:
        tag = monitoring._monitor_ctx.get_tag(key=MONITORING_KEY_SEND)
        window_size = monitoring._monitor_ctx.get_window_size(key=MONITORING_KEY_SEND)
        heartrate = monitoring._monitor_ctx.get_window_heartrate(key = MONITORING_KEY_SEND)
    send_rate = heartrate * module.microbatch_size.item()
    # Only adapt at window period intervals
    if tag > 0 and tag % window_size == 0:
        if send_rate < TARGET_SEND_RATE:
            compress_ratio = int(TARGET_SEND_RATE/send_rate)+1
            if compress_ratio <= 2:
                module.quant_bit = torch.tensor(16)
            elif compress_ratio <=4:
                module.quant_bit = torch.tensor(8)
            elif compress_ratio ==5:
                module.quant_bit = torch.tensor(6)
            elif compress_ratio <=8:
                module.quant_bit = torch.tensor(4)
            else:
                module.quant_bit = torch.tensor(2)
        else:
            module.quant_bit = torch.tensor(0)
    monitoring_send_rate.append(send_rate)
    monitoring_quant_bit.append(module.quant_bit.item() if module.quant_bit.item()!=0 else 32)

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
    quant_bits = module.quant_bit.item()
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
    quant_bits = module.quant_bit.item()
    assert isinstance(inputs, tuple)
    n_items = models.get_microbatch_size(inputs[0], verify=True) if quant_bits > 0 else 0
    monitoring.iteration(MONITORING_KEY_QUANT_ENCODE, work=n_items, accuracy=quant_bits)

def forward_hook_quant_encode(module, _input_arg, output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
    """encode tensor in the forward hook (after each module)"""
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
    return tuple(comm_tuple)

def forward_pre_hook_quant_decode(_module, input_arg: Tuple[Tuple[torch.Tensor, ...]]):
    """decode tensor in the preforward hook (before each module)"""
    assert isinstance(input_arg, tuple)
    assert len(input_arg) == 1
    # input_tensor: len=5x for x tensors encoded as: comm_tensor, input_shape, scale_factor, shift, quant_bit
    input_tensors = input_arg[0]
    assert isinstance(input_tensors, tuple)
    assert len(input_tensors)%5 == 0
    forward_tensor = []
    for i in range(len(input_tensors) // 5):
        input_tensor = input_tensors[i*5:i*5+5]
        batched_tensor = tensor_decode_outerdim(input_tensor)
        forward_tensor.append(batched_tensor)
    # Return value(s) should be wrapped in an outer tuple, like input_arg
    # The tuple will be unpacked when forward() is invoked, which must yield a single parameter
    if len(forward_tensor) == 1:
        # assume that the original result was a single tensor rather than a tuple w/ len=1
        return tuple(forward_tensor)
    return (tuple(forward_tensor),)

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
    if key == MONITORING_KEY_SEND:
        with monitoring._monitor_ctx_lock:
            perf = monitoring._monitor_ctx.get_window_perf(key=MONITORING_KEY_SEND)
        monitoring_send_perf.append(perf)



class ThreadSafeCounter:
    """Thread-safe counter."""

    def __init__(self, value: int=0):
        self._value = value
        self._cond = threading.Condition()

    @property
    def value(self) -> int:
        """Current counter value."""
        with self._cond:
            val = self._value
            self._cond.notify_all()
        return val

    def add(self, quantity: int=1) -> None:
        """Add to counter atomically."""
        with self._cond:
            self._value += quantity
            self._cond.notify_all()

    def wait_gte(self, threshold: int) -> None:
        """Wait until counter >= threshold."""
        with self._cond:
            while self._value < threshold:
                self._cond.wait()

    def reset(self, value: int=0) -> None:
        """Reset counter value."""
        with self._cond:
            self._value = value
            self._cond.notify_all()

# To wait to start data flow, reset to value=0; add to value to start
data_start_counter = ThreadSafeCounter(1)
# To pause data flow, reset to value=0; add to value to resume
data_pause_counter = ThreadSafeCounter(1)
# To stop data flow, add to value
data_stop_counter = ThreadSafeCounter(0)

def _unblock_pipeline():
    data_stop_counter.add()
    data_pause_counter.add()
    data_start_counter.add()

results_counter = ThreadSafeCounter()
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
    with monitoring._monitor_ctx_lock:
        perf = monitoring._monitor_ctx.get_window_perf(key=MONITORING_KEY_OUTPUT)
        acc = monitoring._monitor_ctx.get_window_accuracy(key=MONITORING_KEY_OUTPUT)
        count = monitoring._monitor_ctx.get_tag(key=MONITORING_KEY_OUTPUT)
        window_size = monitoring._monitor_ctx.get_window_size(key=MONITORING_KEY_OUTPUT)
    monitoring_output_perf.append(perf)
    acc_percentage = acc / min(count, window_size) / n_items * 100
    monitoring_output_acc.append(acc_percentage)


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


def _get_default_quant(n_stages: int) -> List[int]:
    return [0] * n_stages


def get_pipeline_sched(world_size: int, hosts: Optional[List[str]],
                       partition: Optional[List[Tuple[int, int]]], quant: Optional[List[int]],
                       rank_order: Optional[List[int]], model_name: str, microbatch_size: int,
                       s_models_file: Optional[str], s_dev_types_file: Optional[str],
                       s_dev_file: Optional[str]) -> \
    Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Get the pipeline schedule: `stage_layers`, `stage_quant`, and `stage_ranks`."""
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


def load_dataset_subset(dataset: Dataset, batch_size: int, shuffle: bool=False) -> Dataset:
    """Get a Dataset subset."""
    if shuffle:
        indices = [random.randint(0, len(dataset)) for _ in range(batch_size)]
    else:
        indices = list(range(batch_size))
    return Subset(dataset, indices)


def load_dataset_imagenet(model_name: str, root: str, split: str='train') -> Dataset:
    """Get the ImageNet dataset."""
    if model_name.startswith('facebook/deit'):
        feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    def transform(img):
        pixels = feature_extractor(images=img.convert('RGB'), return_tensors='pt')['pixel_values']
        # feature extractor expects a batch but we only have a single image, so drop the outer dim
        return pixels[0]
    return ImageNet(root, split=split, transform=transform)


def load_dataset(model_name: str, batch_size: int, indices: Optional[Sequence]=None) -> Dataset:
    """Load inputs based on model."""
    if model_name in ['bert-base-uncased', 'bert-large-uncased',
                      'textattack/bert-base-uncased-CoLA']:
        with np.load("bert_input.npz") as bert_inputs:
            inputs_sentence = list(bert_inputs['input'][:batch_size])
            labels = torch.tensor(bert_inputs['label'][:batch_size])
        tokenizer = BertTokenizer.from_pretrained(model_name)
        inputs = tokenizer(inputs_sentence, padding=True, truncation=True, return_tensors="pt")['input_ids']
        dataset = TensorDataset(inputs, labels)
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
        labels = torch.tensor([IMG_LABEL_IDX] * batch_size)
        dataset = TensorDataset(inputs, labels)
    if indices is not None:
        dataset = Subset(dataset, indices)
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
                     hosts: Optional[List[str]], sched_models_file: Optional[str],
                     sched_dev_types_file: Optional[str], sched_dev_file: Optional[str],
                     dataset_name: Optional[str]=None, dataset_root: Optional[str]=None,
                     dataset_split: Optional[str]=None, dataset_shuffle: bool=False,
                     dataloader_num_workers: int=0) -> None:
    """Run the pipeline using P2P communication."""
    monitoring.init(MONITORING_KEY_MODEL, work_type='tensors', acc_type='layers')
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
            q_bit = torch.tensor(stage_quant[stage])
            ubatch_size_tensor = torch.tensor(ubatch_size)
            model.register_buffer('quant_bit', q_bit)
            model.register_buffer('microbatch_size', ubatch_size_tensor)
            model.register_forward_hook(devices.forward_hook_to_cpu)
            model.register_forward_hook(forward_hook_monitor)
            if stage != len(stage_ranks) - 1:
                model.register_forward_hook(forward_hook_bandwidth_detect)
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
        with model_cfg.dist_p2p_pipeline_stage_factory(stage_ranks, data_rank, rank, stage, model,
                                                       handle_results) as stage_ctx:
            stage_ctx.register_recv_pre_hook(p2p_pre_hook_monitor, (MONITORING_KEY_RECV,))
            stage_ctx.register_recv_post_hook(p2p_post_hook_monitor, (MONITORING_KEY_RECV,))
            stage_ctx.register_send_pre_hook(p2p_pre_hook_monitor, (MONITORING_KEY_SEND,))
            stage_ctx.register_send_post_hook(p2p_post_hook_monitor, (MONITORING_KEY_SEND,))
            if rank == data_rank:
                if dataset_name == "ImageNet":
                    dataset = load_dataset_imagenet(model_name, dataset_root, split=dataset_split)
                    dataset = load_dataset_subset(dataset, batch_size, shuffle=dataset_shuffle)
                else:
                    dataset = load_dataset(model_name, batch_size)
                data_loader = DataLoader(dataset, batch_size=ubatch_size,
                                         num_workers=dataloader_num_workers)
                data_start_counter.wait_gte(1)
                tik_data = time.time()
                # start results monitoring - see comments in handle_results
                monitoring.iteration(MONITORING_KEY_OUTPUT, work=0, accuracy=0, safe=False)
                # this call is asynchronous - wait for results to get end-to-end timings
                start_count = results_counter.value
                total_items = 0
                for ubatch, ubatch_labels in data_loader:
                    data_pause_counter.wait_gte(1)
                    if data_stop_counter.value > 0:
                        break
                    label_queue.put(ubatch_labels)
                    stage_ctx.enqueue_tensor(ubatch)
                    total_items += len(ubatch)
                results_counter.wait_gte(start_count + total_items)
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
                     hosts: Optional[List[str]], sched_models_file: Optional[str],
                     sched_dev_types_file: Optional[str], sched_dev_file: Optional[str],
                     rpc_num_worker_threads: int) -> None:
    """Run the pipeline using RPC communication."""
    monitoring.init(MONITORING_KEY_MODEL, work_type='tensors', acc_type='layers')
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
            dataset = load_dataset(model_name, batch_size)
            data_loader = DataLoader(dataset, batch_size=ubatch_size)
            # Create model shards on workers (requires distributed context to be initialized)
            pipeline = model_cfg.dist_rpc_pipeline_factory(model_name, model_file, stage_ranks,
                                                           stage_layers, data_rank, handle_results)
            q_bit = [torch.tensor(stage_quant[s])
                      for s in range(len(stage_quant))]
            ubatch_size = torch.tensor(ubatch_size)
            pipeline.rpc_register_buffer('quant_bit', q_bit)
            pipeline.rpc_register_buffer('microbatch_size', ubatch_size)
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
    logger.info("Device: %s", devices.DEVICE)
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
    logger.debug("# parallel intra nodes threads: %d", torch.get_num_threads())
    logger.debug("# parallel inter nodes threads: %d", torch.get_num_interop_threads())

    # Network
    os.environ['MASTER_ADDR'] = net_addr # MASTER_ADDR
    os.environ['MASTER_PORT'] = str(net_port) # MASTER_PORT
    os.environ["TP_SOCKET_IFNAME"] = net_ifname # SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = net_ifname # SOCKET_IFNAME


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    '''
    # error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['result_callback'] = self.signals.result

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        self.fn(*self.args, **self.kwargs)

class MainWindow(QMainWindow):
    """GUI main window"""
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PipeEdge Performance Monitor")
        self.TOT_NUM_FIGS = len(fig_titles)
        self.COL_NUM_FIGS = floor(sqrt(self.TOT_NUM_FIGS))
        self.ROW_NUM_FIGS = ceil(self.TOT_NUM_FIGS/self.COL_NUM_FIGS)
        if self.COL_NUM_FIGS * self.ROW_NUM_FIGS == self.TOT_NUM_FIGS:
            self.fig_titles = fig_titles
            self.fig_yaxi_label = fig_yaxi_label
        else:
            self.fig_titles = fig_titles + \
                ["" for _ in range(self.COL_NUM_FIGS * self.ROW_NUM_FIGS - self.TOT_NUM_FIGS)]
            self.fig_yaxi_label = fig_yaxi_label + \
                ["" for _ in range(self.COL_NUM_FIGS * self.ROW_NUM_FIGS - self.TOT_NUM_FIGS)]
        self.labels = []
        self.graphWidgets = []
        self.plots = []
        pg.setConfigOptions(foreground=pg.mkColor(0.0))
        for i in range(self.ROW_NUM_FIGS):
            for j in range(self.COL_NUM_FIGS):
                tmp_label = QLabel()
                tmp_label.setText(self.fig_titles[i*self.COL_NUM_FIGS+j])
                tmp_label.setFont(QFont('Times', 16))
                tmp_widget = pg.PlotWidget() if self.fig_titles[i*self.COL_NUM_FIGS+j] != "" else QWidget()
                self.labels.append(tmp_label)
                self.graphWidgets.append(tmp_widget)

        pagelayout = QVBoxLayout()
        button_layout = QHBoxLayout()
        figs_layout = QGridLayout()
        config_layout = QHBoxLayout()

        # init the plot window
        self.results = [[0,] for _ in range(self.ROW_NUM_FIGS * self.COL_NUM_FIGS)]
        self.init_plot()
        for i in range(self.ROW_NUM_FIGS):
            for j in range(self.COL_NUM_FIGS):
                fig_layout = QVBoxLayout()
                fig_layout.addWidget(self.labels[i*self.COL_NUM_FIGS+j])
                fig_layout.addWidget(self.graphWidgets[i*self.COL_NUM_FIGS+j])
                figs_layout.addLayout(fig_layout, i, j)
        pagelayout.addLayout(figs_layout)
        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(config_layout)

        self.strt_btn = QPushButton("Start")
        self.strt_btn.pressed.connect(self.start_task)
        button_layout.addWidget(self.strt_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.pressed.connect(self.pause_task)
        button_layout.addWidget(self.pause_btn)

        self.stp_btn = QPushButton("Stop")
        self.stp_btn.pressed.connect(self.stop_task)
        button_layout.addWidget(self.stp_btn)

        config_layout.addWidget(QLabel("Set Rate Constraint (images/sec):"))
        self.target_rate_txtbox = QLineEdit()
        # self.target_rate_txtbox.editingFinished.connect(self.set_target_rate_task)
        config_layout.addWidget(self.target_rate_txtbox)
        target_rate_btn = QPushButton("Set")
        target_rate_btn.pressed.connect(self.set_target_rate_task)
        config_layout.addWidget(target_rate_btn)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.worker = Worker(self.poll_fig_data)
        self.worker.signals.result.connect(self.update_plot)
        self.threadpool = QThreadPool()
        self.threadpool.start(self.worker)

        self.end_thread = threading.Event()

    def closeEvent(self, event):
        _unblock_pipeline()
        self.end_thread.set()
        event.accept()

    def init_plot(self, x=0, y=0):
        # plot data: x, y values
        for i in range(self.ROW_NUM_FIGS):
            for j in range(self.COL_NUM_FIGS):
                if self.fig_titles[i*self.COL_NUM_FIGS+j] != "":
                    bottom_lbl_style = {'font-size': '18pt'}
                    left_lbl_style = {'font-size': '18pt'}
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].setBackground('w')
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].setXRange(0, PLOT_DATAPOINT_NUMBER, padding=0)
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].setLabel(axis='bottom', text='Microbatch', **bottom_lbl_style)
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].setLabel(axis='left', text=self.fig_yaxi_label[i*self.COL_NUM_FIGS+j], **left_lbl_style)
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].showGrid(x=True, y=True, alpha=0.2)
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].getAxis("bottom").setTickFont(QFont('Times', 16))
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].getAxis("left").setTickFont(QFont('Times', 16))
                    pen = pg.mkPen(color=(255, 0, 0), width=2)
                    self.plots.append(self.graphWidgets[i*self.COL_NUM_FIGS+j].plot([x], [y], pen=pen))

    def update_plot(self):
        window_size = monitoring.get_window_size()
        for i in range(self.ROW_NUM_FIGS):
            for j in range(self.COL_NUM_FIGS):
                if self.fig_titles[i*self.COL_NUM_FIGS+j] != "":
                    y = self.results[i*self.COL_NUM_FIGS+j] if len(self.results[i*self.COL_NUM_FIGS+j])!=0 else [0,]
                    # x must be derived from y in case of race condition, resulting unequal size of x and y
                    x = list(range(len(y)))
                    # slide in window size increments to reduce noise
                    idx_max = max(len(self.results[i*self.COL_NUM_FIGS+j]) - 1, 0)
                    x_max = int(ceil(idx_max / window_size)) * window_size
                    x_max = max(PLOT_DATAPOINT_NUMBER, x_max)
                    x_min = max(0, x_max - PLOT_DATAPOINT_NUMBER)
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].setXRange(x_min, x_max)
                    y_max = max(self.results[i*self.COL_NUM_FIGS+j][x_min:min(x_max, idx_max)], default=1)
                    self.graphWidgets[i*self.COL_NUM_FIGS+j].setYRange(0, y_max)
                    self.plots[i*self.COL_NUM_FIGS+j].setData(x, y)

    def poll_fig_data(self, result_callback):
        while True:
            time.sleep(PLOT_TIME_INTERVAL)
            if self.end_thread.is_set():
                break
            perf_data = fetch_data_from_runtime()
            for i in range(self.ROW_NUM_FIGS):
                for j in range(self.COL_NUM_FIGS):
                    if self.fig_titles[i*self.COL_NUM_FIGS+j] != "":
                        self.results[i*self.COL_NUM_FIGS+j] = perf_data[i*self.COL_NUM_FIGS+j]
            result_callback.emit(self.results)

    def start_task(self):
        data_start_counter.add()
        self.strt_btn.setEnabled(False)

    def pause_task(self):
        if self.pause_btn.text() == "Pause":
            data_pause_counter.reset()
            self.pause_btn.setText("Resume")
        else:
            data_pause_counter.add()
            self.pause_btn.setText("Pause")

    def stop_task(self):
        _unblock_pipeline()

    def set_target_rate_task(self):
        global TARGET_SEND_RATE
        text = self.target_rate_txtbox.text()
        try:
            TARGET_SEND_RATE = float(text)
        except ValueError:
            logger.error("Invalid float value: %s", text)
            self.target_rate_txtbox.setText(f"Use a valid float value!")


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
    parser.add_argument("-g", "--gui", action='store_true', help="use GUI")
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
    parser.add_argument("-c", "--comm", type=str, default="p2p",
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
    # Data loader options
    # num_workers > 0 is slow to break the processing for loop after data [sub]set completes
    parser.add_argument("--dataloader-num-workers", default=0, type=int,
                        help="dataloader worker threads (0 uses the main thread)")
    # Dataset options
    dset = parser.add_argument_group('Dataset arguments')
    dset.add_argument("--dataset-name", type=str, choices=['ImageNet'],
                      help="dataset to use")
    dset.add_argument("--dataset-root", type=str,
                      help="dataset root directory (e.g., for ImageNet, must contain "
                           "'ILSVRC2012_devkit_t12.tar.gz' and at least one of: "
                           "'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'")
    dset.add_argument("--dataset-split", default='train', type=str,
                      help="dataset split (depends on dataset), e.g.: train, val, test")
    dset.add_argument("--dataset-shuffle", default=False, type=bool,
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
    # Tuning the plot and demo configuration
    plt_cfg = parser.add_argument_group('Plot tuning')
    plt_cfg.add_argument("-ppn", "--plot-point-num", default=100, type=int,
                        help="the range of polt xaxi; recommand [100|1000] on device [cpu|cuda]")
    args = parser.parse_args()

    # Change the global value
    global PLOT_DATAPOINT_NUMBER
    PLOT_DATAPOINT_NUMBER = args.plot_point_num



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
    monitoring._PRINT_FIELDS_INSTANT = False

    def _run():
        if args.comm == 'p2p':
            run_pipeline_p2p(args.worldsize, args.rank, args.model_name, args.model_file,
                            args.batch_size, args.ubatch_size, partition, quant, rank_order,
                            args.data_rank, hosts, args.sched_models_file, args.sched_dev_types_file,
                            args.sched_dev_file, args.dataset_name, args.dataset_root,
                            args.dataset_split, args.dataset_shuffle, args.dataloader_num_workers)
        else:
            run_pipeline_rpc(args.worldsize, args.rank, args.model_name, args.model_file,
                            args.batch_size, args.ubatch_size, partition, quant, rank_order,
                            args.data_rank, hosts, args.sched_models_file, args.sched_dev_types_file,
                            args.sched_dev_file, args.worker_threads)

    t = threading.Thread(target=_run)
    t.start()

    # construct monitor panel
    if args.gui:
        data_start_counter.reset()
        app = QApplication([""])
        window = MainWindow()
        window.show()
        app.exec()

    t.join()

    tok = time.time()
    logger.info("Total program execution time = %f", tok - tik)



if __name__=="__main__":
    logging.basicConfig(filename='runtime.log', level=logging.DEBUG)
    console_hndlr = logging.StreamHandler(sys.stdout)
    console_hndlr.setFormatter(logging.Formatter(fmt='%(message)s'))
    console_hndlr.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_hndlr)
    main()