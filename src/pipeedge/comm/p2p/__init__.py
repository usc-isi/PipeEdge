"""Peer-to-peer communication module."""
import collections
import queue
import threading
import time
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from .. import DistCmdHandler, DistContext
from .util import DistRequestWaitDaemon

# Base tag values
TAG_BASE_DATA = 0
TAG_BASE_CMD = 10

# Offsets which are added to base values above
TAG_TENSOR_COUNT = 0
TAG_TENSOR_DTYPE = 1
TAG_TENSOR_SHAPE_LEN = 2
TAG_TENSOR_SHAPE = 3
TAG_TENSOR = 4

# Ordered set of torch types: https://pytorch.org/docs/stable/tensor_attributes.html
TORCH_TYPES = [ torch.float32,
                torch.float64,
                torch.complex64,
                torch.complex128,
                torch.float16,
                torch.bfloat16,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.bool ]
TORCH_TYPES_ENUM = collections.OrderedDict()
for i, t in enumerate(TORCH_TYPES):
    TORCH_TYPES_ENUM[t] = i


class DistP2pContext(DistContext):
    """The singleton distributed P2P context manager."""

    def __init__(self, ipg_args: tuple, ipg_kwargs: dict, cmd_cb: DistCmdHandler):
        super().__init__(ipg_args, ipg_kwargs)
        self._thread_cmd = CommandThread(cmd_cb)

    def init(self) -> None:
        """Initialize the distributed context and threads."""
        super().init()
        dist.init_process_group(*self._init_args, **self._init_kwargs)
        self._thread_cmd.start()

    def shutdown(self) -> None:
        """Shutdown threads and the distributed context."""
        super().shutdown()
        self._thread_cmd.stop()
        self._thread_cmd.join()
        dist.destroy_process_group()

    def cmd_broadcast(self, cmd: int, tensors: Optional[Tuple[torch.Tensor, ...]]=None) -> None:
        """Broadcast a command."""
        assert self._initialized
        if tensors is None:
            tensors = ()
        tensor_cmd = torch.tensor([cmd, len(tensors)], dtype=torch.int)
        reqs = []
        for dst in range(self._world_size):
            if dst != self._rank:
                reqs.append(dist.isend(tensor_cmd, dst=dst, tag=TAG_BASE_CMD))
                for tensor in tensors:
                    reqs += _send_tensor(tensor, dst, TAG_BASE_CMD, fn_send=dist.isend)
        for req in reqs:
            req.wait()


class ConditionQueue(queue.Queue):
    """A Queue with a public `condition: threading.Condition` variable for synchronization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.condition = threading.Condition()


def _send_tensor(tensor, dst, tag_base, fn_send=dist.send):
    # NOTE: could optimize by packing dtype and shape length into one message
    tensor_dtype = torch.tensor(TORCH_TYPES_ENUM[tensor.dtype], dtype=torch.int)
    tensor_shape_len = torch.tensor(len(tensor.shape), dtype=torch.int)
    tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)
    results = []
    results.append(fn_send(tensor=tensor_dtype, dst=dst, tag=tag_base+TAG_TENSOR_DTYPE))
    results.append(fn_send(tensor=tensor_shape_len, dst=dst, tag=tag_base+TAG_TENSOR_SHAPE_LEN))
    results.append(fn_send(tensor=tensor_shape, dst=dst, tag=tag_base+TAG_TENSOR_SHAPE))
    results.append(fn_send(tensor=tensor, dst=dst, tag=tag_base+TAG_TENSOR))
    return results


def _recv_tensor(src, tag_base):
    tensor_dtype = torch.zeros(1, dtype=torch.int)
    dist.recv(tensor=tensor_dtype, src=src, tag=tag_base+TAG_TENSOR_DTYPE)
    _tensor_dtype = TORCH_TYPES[int(tensor_dtype)]
    tensor_shape_len = torch.zeros(1, dtype=torch.int)
    dist.recv(tensor=tensor_shape_len, src=src, tag=tag_base+TAG_TENSOR_SHAPE_LEN)
    _tensor_shape_len = int(tensor_shape_len)
    tensor_shape = torch.zeros(_tensor_shape_len, dtype=torch.int)
    dist.recv(tensor=tensor_shape, src=src, tag=tag_base+TAG_TENSOR_SHAPE)
    _tensor_shape = [int(x) for x in tensor_shape] # list(map(lambda x: int(x), tensor_shape))
    tensor = torch.zeros(_tensor_shape, dtype=_tensor_dtype)
    dist.recv(tensor=tensor, src=src, tag=tag_base+TAG_TENSOR)
    return tensor


class AbstractTensorExchangeThread(threading.Thread):
    """Abstract tensor exchange thread."""

    def __init__(self):
        super().__init__()
        self._pre_hooks = []
        self._post_hooks = []

    def register_pre_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register hook with signature: `hook(*args)`."""
        self._pre_hooks.append((hook, args))

    # Python 3.7 type hinting doesn't support the real hook function signature, which is more like:
    # `Callable[[Tuple[torch.Tensor], ...], None]`
    def register_post_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register hook with signature: `hook(tensors, *args)`."""
        self._post_hooks.append((hook, args))

    def run(self):
        """Still-abstract thread run method."""
        raise NotImplementedError

    def _call_pre_hooks(self):
        for hook, args in self._pre_hooks:
            hook(*args)

    def _call_post_hooks(self, tensors):
        for hook, args in self._post_hooks:
            hook(tensors, *args)


class TensorSendThread(AbstractTensorExchangeThread):
    """Thread for sending tensors."""

    def __init__(self, queue_out: ConditionQueue, dst_rank: int):
        super().__init__()
        self._queue_out = queue_out
        self._dst_rank = dst_rank
        self._evt_stop_thread = threading.Event()

    def stop(self) -> None:
        """Direct the thread to stop."""
        with self._queue_out.condition:
            self._evt_stop_thread.set()
            self._queue_out.condition.notify_all()

    def run(self):
        """Dequeue tensors and send them."""
        while not self._evt_stop_thread.is_set():
            with self._queue_out.condition:
                while self._queue_out.empty():
                    if self._evt_stop_thread.is_set():
                        return
                    self._queue_out.condition.wait()
                tensors = self._queue_out.get(block=False)
                self._queue_out.condition.notify_all()
            # tensors come in pairs, except for the last stage results
            if isinstance(tensors, torch.Tensor):
                tensors = (tensors,)
            assert isinstance(tensors, tuple)
            assert len(tensors) > 0
            assert isinstance(tensors[0], torch.Tensor)
            _tensor_count = len(tensors)
            tensor_count = torch.tensor(_tensor_count, dtype=torch.int)
            dist.send(tensor=tensor_count, dst=self._dst_rank, tag=TAG_BASE_DATA+TAG_TENSOR_COUNT)
            # NOTE: could optimize by only sending dtype once (it's the same for all tensors)
            self._call_pre_hooks()
            for tensor in tensors:
                _send_tensor(tensor, self._dst_rank, TAG_BASE_DATA)
            self._call_post_hooks(tensors)


class TensorRecvThread(AbstractTensorExchangeThread):
    """Thread for receiving tensors."""

    def __init__(self, queue_in: ConditionQueue, src_rank: int):
        super().__init__()
        self._queue_in = queue_in
        self._src_rank = src_rank
        self._evt_stop_thread = threading.Event()

    def stop(self) -> None:
        """Direct the thread to stop."""
        self._evt_stop_thread.set()

    def run(self):
        """Receive tensors and enqueue them."""
        while True:
            tensor_count = torch.zeros(1, dtype=torch.int)
            ircv_req = dist.irecv(tensor=tensor_count, src=self._src_rank, tag=TAG_BASE_DATA+TAG_TENSOR_COUNT)
            ircv_req_t = DistRequestWaitDaemon(ircv_req)
            ircv_req_t.start()
            while ircv_req_t.is_alive():
                if self._evt_stop_thread.is_set():
                    return
                # TODO: we're basically spinning...
                time.sleep(0.1)
            _tensor_count = int(tensor_count)
            assert _tensor_count > 0
            tensors = ()
            self._call_pre_hooks()
            for _ in range(_tensor_count):
                tensor = _recv_tensor(self._src_rank, TAG_BASE_DATA)
                tensors += (tensor,)
            self._call_post_hooks(tensors)
            if _tensor_count == 1:
                # At this point, we don't know whether the original data type was a Tensor or Tuple[Tensor] w/ len=1.
                # We'd have to include that information in a separate message to know for sure.
                # For now, it works to reduce to the base case - just a single Tensor.
                tensors = tensors[0]
            # Blocks if queue is full, which then blocks receiving more tensors (as intended)
            # Worker thread must be running to avoid indefinite blocking
            with self._queue_in.condition:
                while self._queue_in.full():
                    self._queue_in.condition.wait()
                self._queue_in.put(tensors)
                self._queue_in.condition.notify_all()


class TensorWorkThread(threading.Thread):
    """Thread for processing tensors."""

    def __init__(self, queue_in: ConditionQueue, queue_out: ConditionQueue, callback: Callable):
        super().__init__()
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._callback = callback
        self._evt_stop_thread = threading.Event()

    def stop(self) -> None:
        """Direct the thread to stop."""
        with self._queue_in.condition:
            self._evt_stop_thread.set()
            self._queue_in.condition.notify_all()

    def run(self):
        """Dequeue, process, enqueue."""
        # Empty inbound queue before stopping
        while True:
            with self._queue_in.condition:
                while self._queue_in.empty():
                    if self._evt_stop_thread.is_set():
                        return
                    self._queue_in.condition.wait()
                tensor_in = self._queue_in.get(block=False)
                self._queue_in.condition.notify_all()
            tensor_out = self._callback(tensor_in)
            if tensor_out is not None:
                # Sender thread must be running to avoid indefinite blocking
                with self._queue_out.condition:
                    while self._queue_out.full():
                        self._queue_out.condition.wait()
                    self._queue_out.put(tensor_out)
                    self._queue_out.condition.notify_all()


class CommandThread(threading.Thread):
    """Thread for receiving commands."""

    def __init__(self, callback: DistCmdHandler):
        super().__init__()
        self._callback = callback
        self._evt_stop_thread = threading.Event()

    def stop(self) -> None:
        """Direct the thread to stop."""
        self._evt_stop_thread.set()

    def run(self):
        """Listen for commands."""
        while True:
            # contains (1) CMD enumeration and (2) an optional tensor count
            tensor_cmd = torch.zeros(2, dtype=torch.int)
            ircv_req = dist.irecv(tensor=tensor_cmd, tag=TAG_BASE_CMD)
            ircv_req_t = DistRequestWaitDaemon(ircv_req)
            ircv_req_t.start()
            while ircv_req_t.is_alive():
                if self._evt_stop_thread.is_set():
                    return
                # TODO: we're basically spinning...
                time.sleep(0.1)
            cmd = int(tensor_cmd[0])
            _tensor_count = int(tensor_cmd[1])
            tensors = ()
            for _ in range(_tensor_count):
                # it would be nice if we could restrict src to the prior request's src, but the
                # ircv_req "distributed request object" API doesn't document a src property to use
                tensor = _recv_tensor(None, TAG_BASE_CMD)
                tensors += (tensor,)
            self._callback(cmd, tensors)


class DistP2pPipelineStage:
    """
    The singleton distributed P2P pipeline stage context manager.

    Creates receiver, sender, worker, and results processing threads when their respective
    optional parameters are specified.
    Threads communicate with each other through data queues, where the exact configuration depends
    on which threads are requested.
    Parameters must be specified appropriately on each rank to form a functionally correct pipeline.

    Because there is (at most) one receiver thread, only one rank may specify `results_cb` and
    that rank must not have a `work_cb` (be a stage) in the middle of the work pipeline.
    If it's the first work stage, that rank must also be the data source feeding `enqueue_batch`
    (i.e., not receive inputs from a rank outside the work pipeline).
    If it's the last work stage, then `rank_dst` must be `None`, otherwise the results processing
    thread and sender thread would race for the data produced by the work thread.
    Otherwise, the rank specifying `results_cb` must not be in the work pipeline.

    Ranks that do nothing may specify `None` for all parameters.
    """

    def __init__(self, rank_src: Optional[int], rank_dst: Optional[int],
                 work_cb: Optional[Callable], results_cb: Optional[Callable[[Any], None]]):
        self._initialized = False
        self._queues = {}
        self._threads = {}
        self._create_stage(rank_src, rank_dst, work_cb, results_cb)

    def _create_stage(self, rank_src, rank_dst, work_cb, results_cb):
        self._queues['in'] = ConditionQueue(maxsize=1)
        self._queues['out'] = ConditionQueue(maxsize=1)
        self._queues['res'] = ConditionQueue(maxsize=1)

        if work_cb is None:
            # Short-circuit from the inbound queue (can relay data without a worker thread)
            self._queues['out'] = self._queues['in']
        else:
            self._threads['work'] = TensorWorkThread(self._queues['in'], self._queues['out'],
                                                     work_cb)

        if results_cb is not None:
            queue_res = self._queues['out'] if rank_dst is None else self._queues['res']
            self._threads['res'] = TensorWorkThread(queue_res, None, results_cb)

        if rank_dst is not None:
            self._threads['send'] = TensorSendThread(self._queues['out'], rank_dst)

        if rank_src is not None:
            queue_in = self._queues['in'] if results_cb is None else self._queues['res']
            self._threads['recv'] = TensorRecvThread(queue_in, rank_src)

    def init(self) -> None:
        """Initialize the distributed context and threads."""
        assert not self._initialized
        self._initialized = True
        for thr in self._threads.values():
            thr.start()

    def shutdown(self) -> None:
        """Shutdown threads and the distributed context."""
        assert self._initialized
        self._initialized = False
        for thr in self._threads.values():
            thr.stop()
            thr.join()

    def register_recv_pre_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register a pre hook for tensor receive with signature: `hook(*args)`."""
        thr = self._threads.get('recv')
        if thr is not None:
            thr.register_pre_hook(hook, args)

    def register_recv_post_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register a post hook for tensor receive with signature: `hook(tensors, *args)`."""
        thr = self._threads.get('recv')
        if thr is not None:
            thr.register_post_hook(hook, args)

    def register_send_pre_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register a pre hook for tensor send with signature: `hook(*args)`."""
        thr = self._threads.get('send')
        if thr is not None:
            thr.register_pre_hook(hook, args)

    def register_send_post_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register a post hook for tensor send with signature: `hook(tensors, *args)`."""
        thr = self._threads.get('send')
        if thr is not None:
            thr.register_post_hook(hook, args)

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *args):
        self.shutdown()

    def enqueue_batch(self, inputs: torch.Tensor, split_size: int) -> None:
        """Insert data into the pipeline."""
        assert self._initialized
        for input_chunk in iter(inputs.split(split_size, dim=0)):
            queue_in = self._queues['in']
            with queue_in.condition:
                while queue_in.full():
                    queue_in.condition.wait()
                queue_in.put(input_chunk)
                queue_in.condition.notify_all()
