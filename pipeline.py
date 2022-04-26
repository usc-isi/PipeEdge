"""Pipeline contexts."""
from torch.distributed import rpc
from edgepipe.comm import p2p


class DistRpcPipeline():
    """The singleton distributed RPC pipeline context manager."""

    def __init__(self, world_size, rank, num_rpc_worker_threads):
        self._world_size = world_size
        self._rank = rank
        self._num_rpc_worker_threads = num_rpc_worker_threads
        self._initialized = False

    def init(self):
        """Initialize the distributed context."""
        assert not self._initialized
        # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=self._num_rpc_worker_threads,
                                                  rpc_timeout=3000)
        rpc.init_rpc(f"worker{self._rank}",
                     rank=self._rank,
                     world_size=self._world_size,
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

    def forward_model(self, model, inputs, split_size, results_cb):
        """Drive the distributed pipeline model with input data."""
        assert self._initialized
        outputs = model(inputs, split_size=split_size)
        results_cb(outputs)


class DistP2pContext():
    """The singleton distributed P2P context manager."""

    def __init__(self, world_size, rank, cmd_cb):
        self._world_size = world_size
        self._rank = rank
        self._initialized = False
        self._thread_cmd = p2p.CommandThread(cmd_cb)

    def init(self):
        """Initialize the distributed context and threads."""
        assert not self._initialized
        self._initialized = True
        p2p.init(self._rank, self._world_size)
        self._thread_cmd.start()

    def shutdown(self):
        """Shutdown threads and the distributed context."""
        assert self._initialized
        self._initialized = False
        self._thread_cmd.stop()
        self._thread_cmd.join()
        p2p.shutdown()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *args):
        self.shutdown()

    def cmd_broadcast(self, cmd):
        """Broadcast a command."""
        assert self._initialized
        p2p.cmd_broadcast(cmd)


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
            self._queues['in'] = p2p.ConditionQueue(maxsize=0)
            # results thread must use a different queue than feeds the first model shard
            self._queues['res'] = p2p.ConditionQueue(maxsize=1)
            self._threads['res'] = p2p.TensorWorkThread(self._queues['res'], None, results_cb)
        else:
            self._queues['in'] = p2p.ConditionQueue(maxsize=1)

        if len(stage_ranks) > 1:
            rank_src = stage_ranks[(self._stage - 1)]
            rank_dst = stage_ranks[(self._stage + 1) % len(stage_ranks)]
            # create send/receive/command threads
            self._queues['out'] = p2p.ConditionQueue(maxsize=1)
            self._threads['send'] = p2p.TensorSendThread(self._queues['out'], rank_dst)
            if self._stage == 0:
                # stage 0's receiver thread gets results, so feeds a different queue
                self._threads['recv'] = p2p.TensorRecvThread(self._queues['res'], rank_src)
            else:
                self._threads['recv'] = p2p.TensorRecvThread(self._queues['in'], rank_src)
        else:
            # degenerate case: no send/receive/command threads; the out queue is the results queue
            self._queues['out'] = self._queues['res']

        # all stages do work
        self._threads['work'] = p2p.TensorWorkThread(self._queues['in'], self._queues['out'], work_cb)

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
