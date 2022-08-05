"""Quant functions used to encode/decode tensors in hooks"""
from typing import Tuple, Union
from torch import Tensor, stack
from .basic_op import tensor_encode, tensor_decode


def per_batch_encode(batched_tensor, quant_bit):
    """do quantization on each image in the micro-batched tensor with size [b,c,h,w]"""
    data_tuple, shape_tuple, scale_tuple, shift_tuple = [],[],[],[]
    for sub_tensor in batched_tensor:
        data_tensor, shape, scale_factor, shift = tensor_encode(sub_tensor, quant_bit)
        data_tuple.append(data_tensor)
        shape_tuple.append(shape)
        scale_tuple.append(scale_factor)
        shift_tuple.append(shift)
    batched_data = stack(data_tuple,0)
    batched_shape = stack(shape_tuple,0)
    batched_scale = stack(scale_tuple,0)
    batched_shift = stack(shift_tuple,0)
    return [batched_data, batched_shape, batched_scale, batched_shift]


def forward_hook_quant_encode(module, _input_arg, output: Union[Tensor, Tuple[Tensor, ...]]):
    """encode tensor in the forward hook (after each module)"""
    if isinstance(output, Tensor):
        output = (output,)
    assert isinstance(output, tuple)
    quant_bits = module.quant_bits.tolist()
    comm_tuple = []
    for tensor in output:
        assert isinstance(tensor, Tensor)
        # comm_tuple += tensor_encode(tensor, quant_bits[1])
        comm_tuple += per_batch_encode(tensor, quant_bits[1])
    return tuple(comm_tuple)


def forward_pre_hook_quant_decode(module, input_arg: Tuple[Tuple[Tensor, ...]]):
    """decode tensor in the preforward hook (before each module)"""
    assert isinstance(input_arg, tuple)
    assert len(input_arg) == 1
    # input_tensor: len=4x for x tensors encoded as: comm_tensor, input_shape, scale_factor, shift
    input_tensor = input_arg[0]
    assert isinstance(input_tensor, tuple)
    assert len(input_tensor)%4 == 0
    quant_bits = module.quant_bits.tolist()
    forward_tensor = []
    for i in range(len(input_tensor) // 4):
        batched_tensor_list = []
        data_tensors, input_shapes, scale_factors, shift_factors =\
            input_tensor[i*4], input_tensor[i*4+1], input_tensor[i*4+2], input_tensor[i*4+3]
        for idx, data_tensor in enumerate(data_tensors):
            batched_tensor_list.append(tensor_decode(data_tensor, input_shapes[idx], scale_factors[idx], shift_factors[idx], quant_bits[0]))
        batched_tensor = stack(batched_tensor_list, 0)
        forward_tensor.append(batched_tensor)
    # Return value(s) should be wrapped in an outer tuple, like input_arg
    # The tuple will be unpacked when forward() is invoked, which must yield a single parameter
    if len(forward_tensor) == 1:
        # assume that the original result was a single tensor rather than a tuple w/ len=1
        return tuple(forward_tensor)
    return (tuple(forward_tensor),)
