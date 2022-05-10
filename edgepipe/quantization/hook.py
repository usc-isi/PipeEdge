"""Quant functions used to encode/decode tensors in hooks"""
from .basic_op import tensor_encode, tensor_decode

def forward_hook_quant_encode(module, input_tensor, output):
    """encode tensor in the forward hook (after each module)"""
    assert isinstance(output, tuple)
    quant_bits = module.quant_bits.tolist()
    comm_tuple = []
    for tensor in output:
        comm_tuple += tensor_encode(tensor, quant_bits[1])
    return tuple(comm_tuple)



def forward_pre_hook_quant_decode(module, input_tensor):
    """decode tensor in the preforward hook (before each module)"""
    # input x should be two tuples consists of (comm_tensor, input_shape, scale_factor, shift)
    # comm_tensor: 1-dimensional tensor
    # input_shape: to restore the original shape of the tensor needed for forward
    # scale_facto: the max value of the original uncomperssed tensor
    # shift: the min value of the original uncompressed tensor
    assert isinstance(input_tensor[0], tuple)
    input_tensor = input_tensor[0]
    quant_bits = module.quant_bits.tolist()
    forward_tensor = []
    assert len(input_tensor) % 4 == 0
    for i in range(len(input_tensor) // 4):
        comm_tensor, input_shape, scale_factor, shift =\
            input_tensor[i*4], input_tensor[i*4+1], input_tensor[i*4+2], input_tensor[i*4+3]
        forward_tensor.append(
            tensor_decode(comm_tensor, input_shape, scale_factor, shift, quant_bits[0]))
    return (tuple(forward_tensor),)
