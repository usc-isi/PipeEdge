"""Peer-to-peer communication module."""
import collections
import queue
import threading
import time
import torch
import torch.distributed as dist
from .. import DistContext
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

    def __init__(self, ipg_args: tuple, ipg_kwargs: dict, cmd_cb):
        super().__init__(ipg_args, ipg_kwargs)
        self._thread_cmd = CommandThread(cmd_cb)

    def init(self):
        """Initialize the distributed context and threads."""
        super().init()
        dist.init_process_group(*self._init_args, **self._init_kwargs)
        self._thread_cmd.start()

    def shutdown(self):
        """Shutdown threads and the distributed context."""
        super().shutdown()
        self._thread_cmd.stop()
        self._thread_cmd.join()
        dist.destroy_process_group()

    def cmd_broadcast(self, cmd, tensors=None):
        """Broadcast a command."""
        assert self._initialized
        if tensors is None:
            tensors = ()
        elif isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
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

    def register_pre_hook(self, hook, args):
        """Register hook with signature: `hook(*args)`."""
        self._pre_hooks.append((hook, args))

    def register_post_hook(self, hook, args):
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

    def __init__(self, queue_out, dst_rank):
        super().__init__()
        self._queue_out = queue_out
        self._dst_rank = dst_rank
        self._evt_stop_thread = threading.Event()

    def stop(self):
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

    def __init__(self, queue_in, src_rank):
        super().__init__()
        self._queue_in = queue_in
        self._src_rank = src_rank
        self._evt_stop_thread = threading.Event()

    def stop(self):
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

    def __init__(self, queue_in, queue_out, callback):
        super().__init__()
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._callback = callback
        self._evt_stop_thread = threading.Event()

    def stop(self):
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

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._evt_stop_thread = threading.Event()

    def stop(self):
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


class DistP2pPipelineStage():
    """The singleton distributed P2P pipeline stage context manager."""

    def __init__(self, stage_ranks, stage, work_cb, results_cb):
        self._stage = stage
        self._initialized = False
        self._queues = {}
        self._threads = {}
        if self._stage is not None:
            self._create_stage(stage_ranks, work_cb, results_cb)

    def _create_stage(self, stage_ranks, work_cb, results_cb):
        # stage 0 feeds `in` queue using `enqueue_batch()`; last stage sends results to stage 0
        # inputs are already loaded in memory, so no need to limit in-queue size on stage 0
        if self._stage == 0:
            self._queues['in'] = ConditionQueue(maxsize=0)
            # results thread must use a different queue than feeds the first model shard
            self._queues['res'] = ConditionQueue(maxsize=1)
            self._threads['res'] = TensorWorkThread(self._queues['res'], None, results_cb)
        else:
            self._queues['in'] = ConditionQueue(maxsize=1)

        if len(stage_ranks) > 1:
            rank_src = stage_ranks[(self._stage - 1)]
            rank_dst = stage_ranks[(self._stage + 1) % len(stage_ranks)]
            # create send/receive/command threads
            self._queues['out'] = ConditionQueue(maxsize=1)
            self._threads['send'] = TensorSendThread(self._queues['out'], rank_dst)
            if self._stage == 0:
                # stage 0's receiver thread gets results, so feeds a different queue
                self._threads['recv'] = TensorRecvThread(self._queues['res'], rank_src)
            else:
                self._threads['recv'] = TensorRecvThread(self._queues['in'], rank_src)
        else:
            # degenerate case: no send/receive/command threads; the out queue is the results queue
            self._queues['out'] = self._queues['res']

        # all stages do work
        self._threads['work'] = TensorWorkThread(self._queues['in'], self._queues['out'], work_cb)

    def init(self):
        """Initialize the distributed context and threads."""
        assert not self._initialized
        self._initialized = True
        for thr in self._threads.values():
            thr.start()

    def shutdown(self):
        """Shutdown threads and the distributed context."""
        assert self._initialized
        self._initialized = False
        for thr in self._threads.values():
            thr.stop()
            thr.join()

    def register_recv_pre_hook(self, hook, args):
        """Register a pre hook for tensor receive with signature: `hook(*args)`."""
        thr = self._threads.get('recv')
        if thr is not None:
            thr.register_pre_hook(hook, args)

    def register_recv_post_hook(self, hook, args):
        """Register a post hook for tensor receive with signature: `hook(tensors, *args)`."""
        thr = self._threads.get('recv')
        if thr is not None:
            thr.register_post_hook(hook, args)

    def register_send_pre_hook(self, hook, args):
        """Register a pre hook for tensor send with signature: `hook(*args)`."""
        thr = self._threads.get('send')
        if thr is not None:
            thr.register_pre_hook(hook, args)

    def register_send_post_hook(self, hook, args):
        """Register a post hook for tensor send with signature: `hook(tensors, *args)`."""
        thr = self._threads.get('send')
        if thr is not None:
            thr.register_post_hook(hook, args)

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *args):
        self.shutdown()

    def enqueue_batch(self, inputs, split_size):
        """Insert data into the front of the pipeline."""
        assert self._stage == 0
        assert self._initialized
        for input_chunk in iter(inputs.split(split_size, dim=0)):
            queue_in = self._queues['in']
            with queue_in.condition:
                while queue_in.full():
                    queue_in.condition.wait()
                queue_in.put(input_chunk)
                queue_in.condition.notify_all()
