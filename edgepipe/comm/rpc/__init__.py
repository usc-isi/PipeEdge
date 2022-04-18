"""RPC communication module."""
import torch
from torch import nn
from torch.distributed.rpc import RRef
from ...quantization.hook import forward_hook_quant_encode, forward_pre_hook_quant_decode


def forward_pre_hook_rpc(_module, x):
    """Copy forward input data from the prior stage as needed."""
    return tuple(_x.to_here() for _x in x)


class DistRpcModule(nn.Module):
    """Parent class for distributed RPC modules."""

    def __init__(self):
        super().__init__()
        self._rref_list = []

    def _register_hooks(self):
        """Register hooks."""
        hook_futures = [rref.rpc_async().register_forward_pre_hook(forward_pre_hook_rpc)
                        for rref in self._rref_list]
        torch.futures.wait_all(hook_futures)
        hook_encoder = [rref.rpc_async().register_forward_hook(forward_hook_quant_encode)
                        for rref in self._rref_list[:-1]]
        torch.futures.wait_all(hook_encoder)
        hook_decoder = [rref.rpc_async().register_forward_pre_hook(forward_pre_hook_quant_decode)
                        for rref in self._rref_list[1:]]
        torch.futures.wait_all(hook_decoder)


    def forward(self, xs, **kwargs):
        """Configure and run remote stages using RPC."""
        split_size = kwargs.get('split_size', len(xs))
        out_futures = []
        for x in iter(xs.split(split_size, dim=0)):
            x_rref = RRef(x)
            for rref in self._rref_list[:-1]:
                x_rref = rref.remote().__call__(x_rref)
            y_rref = self._rref_list[-1].rpc_async().__call__(x_rref)
            out_futures.append(y_rref)
        return torch.cat(torch.futures.wait_all(out_futures))
