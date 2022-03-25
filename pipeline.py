"""Pipeline contexts."""
from torch.distributed import rpc

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
