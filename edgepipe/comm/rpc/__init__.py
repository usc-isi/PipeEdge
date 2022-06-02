"""RPC communication module."""
import threading
from typing import Any, Callable, List, Type, Union
import torch
from torch import nn
from torch.distributed import rpc
from .. import DistContext


class DistRpcContext(DistContext):
    """The singleton distributed RPC context manager."""

    def __init__(self, world_size, rank, num_rpc_worker_threads):
        super().__init__(world_size, rank)
        self._num_rpc_worker_threads = num_rpc_worker_threads

    def init(self):
        """Initialize the distributed context."""
        super().init()
        # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=self._num_rpc_worker_threads,
                                                  rpc_timeout=3000)
        rpc.init_rpc(f"worker{self._rank}",
                     rank=self._rank,
                     world_size=self._world_size,
                     rpc_backend_options=options)

    def shutdown(self):
        """Wait for all RPCs to finish and shutdown the distributed context."""
        super().shutdown()
        rpc.shutdown()

    def cmd_broadcast(self, remote_cmd_handler, cmd, tensors=None):
        """Broadcast a command."""
        assert self._initialized
        futs = []
        for rank in range(self._world_size):
            if rank != self._rank:
                fut = rpc.rpc_async(rank, remote_cmd_handler, args=(cmd, tensors))
                futs.append(fut)
        torch.futures.wait_all(futs)


class DistRpcPipelineStage(nn.Module):
    """Wrap a module that is not RPC-aware to manage threading and memory."""
    # NOTE: message ordering is NOT enforced!
    # pylint: disable=too-many-instance-attributes

    def __init__(self, module_cls: Type[nn.Module], module_args: tuple=None,
                 module_kwargs: dict=None):
        super().__init__()
        if module_args is None:
            module_args = ()
        if module_kwargs is None:
            module_kwargs = {}
        # _sem_fwd limits RPC threads in forward(), and thus data memory requirements (in + out).
        # _sem_mod limits Module thread parallelism, and thus processing memory requirements.
        # If each stage is configured for single-thread Module processing, then N=1.
        # Ideally, for _sem_mod value=N:
        # (1) N inputs are being or have been received (prior to forward() or waiting on _sem_mod)
        # (2) N inputs are processing (acquired _sem_mod)
        # (3) N outputs are sending or waiting to send (released _sem_mod)
        # More generally, however, the local stage may be backed up at any of these three steps,
        # depending on its performance relative to other stages and network conditions.
        self._sem_fwd = threading.Semaphore(value=3) # value = 3*N
        self._sem_mod = threading.Semaphore(value=1) # value = N
        self._module = module_cls(*module_args, **module_kwargs)
        self._next_rref = None
        self._results_cb = None
        # Redirect some Module methods to the wrapped module (extend as needed)
        self.register_buffer = self._module.register_buffer
        self.register_forward_hook = self._module.register_forward_hook
        self.register_forward_pre_hook = self._module.register_forward_pre_hook

    def set_next(self, stage_rref: rpc.RRef):
        """Set the RRef of the next pipeline stage."""
        self._next_rref = stage_rref

    def set_results_callback(self, results_cb: Callable[[Any], None]):
        """Set the results callback function on `next`, for use only by the last stage."""
        self._results_cb = results_cb

    def wait_for_ready(self):
        """Wait for this stage to be ready to receive data - MUST be called from previous stage."""
        # NOTE: This approach breaks down if the previous stage fails to send data afterward.
        self._sem_fwd.acquire() # pylint: disable=consider-using-with

    def forward(self, inputs: Any) -> None:
        """Wrap the module's callable method."""
        try:
            with self._sem_mod:
                outputs = self._module(inputs)
            if self._results_cb is None:
                # Sending must be asynchronous, otherwise we lose pipeline parallelism.
                # However, don't try to send until the next stage is ready.
                # If we were to initiate the async send (and then release _sem_fwd) too soon,
                # outbound data could get backlogged in this stage when the next stage is slow.
                self._next_rref.rpc_sync().wait_for_ready()
                self._next_rref.rpc_async().__call__(outputs)
            else:
                # There's no synchronization with the results handler, just send the data.
                rpc.rpc_sync(self._next_rref.owner(), self._results_cb, args=(outputs,))
        finally:
            # Now release so that another microbatch may be received.
            self._sem_fwd.release()


def pipeline_stage_factory(dest: Union[int, rpc.WorkerInfo, str], module_cls: Type[nn.Module],
                           module_args: tuple=None, module_kwargs: dict=None) -> rpc.RRef:
    """Create a `DistRpcPipelineStage` on a remote."""
    return rpc.remote(dest, DistRpcPipelineStage, args=(module_cls, module_args, module_kwargs))


class DistRpcPipeline(nn.Module):
    """A distributed RPC pipeline which links `DistRpcPipelineStage` RRefs."""

    def __init__(self, stage_rrefs: List[rpc.RRef], results_cb: Callable[[Any], None]):
        super().__init__()
        self._rref_list = stage_rrefs
        self._link_pipeline(results_cb)

    def rpc_register_buffer(self, name, tensors):
        """Add buffers to RPC modules."""
        assert len(tensors) == len(self._rref_list)
        futs = [rref.rpc_async().register_buffer(name, tensor)
                for rref, tensor in zip(self._rref_list, tensors)]
        torch.futures.wait_all(futs)

    def rpc_register_forward_pre_hook(self, hook, first=True):
        """Register forward pre hook."""
        rrefs = self._rref_list if first else self._rref_list[1:]
        hook_futures = [rref.rpc_async().register_forward_pre_hook(hook) for rref in rrefs]
        torch.futures.wait_all(hook_futures)

    def rpc_register_forward_hook(self, hook, last=True):
        """Register forward hook."""
        rrefs = self._rref_list if last else self._rref_list[:-1]
        hook_futures = [rref.rpc_async().register_forward_hook(hook) for rref in rrefs]
        torch.futures.wait_all(hook_futures)

    def _link_pipeline(self, results_cb: Callable[[Any], None]):
        n_stages = len(self._rref_list)
        futs = [self._rref_list[i].rpc_async().set_next(self._rref_list[(i + 1) % n_stages])
                for i in range(n_stages)]
        futs.append(self._rref_list[-1].rpc_async().set_results_callback(results_cb))
        torch.futures.wait_all(futs)

    def forward(self, inputs: Any, **kwargs) -> None:
        """Insert data into the front of the pipeline."""
        split_size = kwargs.get('split_size', len(inputs))
        for ubatch in iter(inputs.split(split_size, dim=0)):
            self._rref_list[0].rpc_sync().wait_for_ready()
            self._rref_list[0].rpc_async().__call__(ubatch)
