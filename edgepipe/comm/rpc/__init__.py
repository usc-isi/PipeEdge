"""RPC communication module."""
import torch
from torch import nn
from torch.distributed import rpc
from .. import DistContext
from ...quantization.hook import forward_hook_quant_encode, forward_pre_hook_quant_decode


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


def forward_pre_hook_rpc(_module, x):
    """Copy forward input data from the prior stage as needed."""
    return tuple(_x.to_here() for _x in x)


class DistRpcModule(nn.Module):
    """Parent class for distributed RPC modules."""

    def __init__(self):
        super().__init__()
        self._rref_list = []

    def set_quant_bits(self, stage_quant):
        """Set quantization bit configurations on each stage."""
        assert len(stage_quant) == len(self._rref_list)
        hook_futures = []
        for stage, rref in enumerate(self._rref_list):
            q_bits = torch.tensor((0 if stage == 0 else stage_quant[stage - 1], stage_quant[stage]))
            hook_futures.append(rref.rpc_async().register_buffer('quant_bits', q_bits))
        torch.futures.wait_all(hook_futures)

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

    def _register_hooks(self):
        """Register hooks."""
        self.rpc_register_forward_pre_hook(forward_pre_hook_rpc)
        self.rpc_register_forward_hook(forward_hook_quant_encode, last=False)
        self.rpc_register_forward_pre_hook(forward_pre_hook_quant_decode, first=False)

    def forward(self, xs, **kwargs):
        """Configure and run remote stages using RPC."""
        split_size = kwargs.get('split_size', len(xs))
        out_futures = []
        for x in iter(xs.split(split_size, dim=0)):
            x_rref = rpc.RRef(x)
            for rref in self._rref_list[:-1]:
                x_rref = rref.remote().__call__(x_rref)
            y_rref = self._rref_list[-1].rpc_async().__call__(x_rref)
            out_futures.append(y_rref)
        return torch.cat(torch.futures.wait_all(out_futures))
