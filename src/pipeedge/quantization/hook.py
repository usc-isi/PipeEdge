"""Quant functions used to encode/decode tensors in hooks"""
from base64 import encode
from typing import Tuple, Union
import torch
from torch import Tensor, stack
from .basic_op import tensor_encode, tensor_decode
import ipdb

def tensor_encode_outerdim(batched_tensor, quant_bit):
    """do quantization on each image in the micro-batched tensor with size [b,c,h,w]"""
    list_of_lists = [tensor_encode(t, quant_bit) for t in batched_tensor]
    return list(zip(*list_of_lists))

def tensor_decode_outerdim(input_tensor):
    """decode the encoded tensor with multiple images in one batch, each encoded image data is in length of 5"""
    quant_bit = input_tensor[4].item()
    batched_tensor_list = []
    data_tensors, input_shapes, scale_factors, shift_factors =\
        input_tensor[0], input_tensor[1], input_tensor[2], input_tensor[3]
    for idx, data_tensor in enumerate(data_tensors):
        batched_tensor_list.append(tensor_decode(data_tensor, input_shapes[idx], scale_factors[idx], shift_factors[idx], quant_bit))
    return stack(batched_tensor_list, 0)

def forward_hook_quant_encode(module, _input_arg, output: Union[Tensor, Tuple[Tensor, ...]]):
    """encode tensor in the forward hook (after each module)"""
    if isinstance(output, Tensor):
        output = (output,)
    assert isinstance(output, tuple)
    quant_bit = module.quant_bit.item()
    comm_tuple = []
    for tensor in output:
        assert isinstance(tensor, Tensor)
        encoded_tensors = tensor_encode_outerdim(tensor, quant_bit)
        stacked_tensors = [stack(t,0) for t in encoded_tensors] + [torch.tensor(quant_bit)]
        comm_tuple += stacked_tensors
    return tuple(comm_tuple)


def forward_pre_hook_quant_decode(module, input_arg: Tuple[Tuple[Tensor, ...]]):
    """decode tensor in the preforward hook (before each module)"""
    assert isinstance(input_arg, tuple)
    assert len(input_arg) == 1
    # input_tensor: len=4x for x tensors encoded as: comm_tensor, input_shape, scale_factor, shift
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
