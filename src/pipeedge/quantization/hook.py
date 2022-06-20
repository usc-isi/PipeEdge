"""Quant functions used to encode/decode tensors in hooks"""
from typing import Tuple, Union
from torch import Tensor
from .basic_op import tensor_encode, tensor_decode

def forward_hook_quant_encode(module, _input_arg, output: Union[Tensor, Tuple[Tensor, ...]]):
    """encode tensor in the forward hook (after each module)"""
    if isinstance(output, Tensor):
        output = (output,)
    assert isinstance(output, tuple)
    quant_bits = module.quant_bits.tolist()
    comm_tuple = []
    for tensor in output:
        assert isinstance(tensor, Tensor)
        comm_tuple += tensor_encode(tensor, quant_bits[1])
    return tuple(comm_tuple)


def forward_pre_hook_quant_decode(module, input_arg: Tuple[Tuple[Tensor, ...]]):
    """decode tensor in the preforward hook (before each module)"""
    assert isinstance(input_arg, tuple)
    assert len(input_arg) == 1
    # input_tensor: len=4x for x tensors encoded as: comm_tensor, input_shape, scale_factor, shift
    input_tensor = input_arg[0]
    assert isinstance(input_tensor, tuple)
    assert len(input_tensor) % 4 == 0
    quant_bits = module.quant_bits.tolist()
    forward_tensor = []
    for i in range(len(input_tensor) // 4):
        # comm_tensor: 1-dimensional tensor
        # input_shape: to restore the original shape of the tensor needed for forward
        # scale_factor: the max value of the original uncompressed tensor
        # shift: the min value of the original uncompressed tensor
        comm_tensor, input_shape, scale_factor, shift =\
            input_tensor[i*4], input_tensor[i*4+1], input_tensor[i*4+2], input_tensor[i*4+3]
        forward_tensor.append(
            tensor_decode(comm_tensor, input_shape, scale_factor, shift, quant_bits[0]))
    # Return value(s) should be wrapped in an outer tuple, like input_arg
    # The tuple will be unpacked when forward() is invoked, which must yield a single parameter
    if len(forward_tensor) == 1:
        # assume that the original result was a single tensor rather than a tuple w/ len=1
        return tuple(forward_tensor)
    return (tuple(forward_tensor),)
