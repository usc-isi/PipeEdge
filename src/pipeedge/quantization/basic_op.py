""" Basic operations used for Quantization """
from typing import List
import numpy as np
import torch

def _quant_op(input_data, bit, mode='original'):
    """
    The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    assert bit > 0
    assert np.all(input_data >= 0)
    assert np.all(input_data <= 1)

    # input should be in [0,1]
    # the res can be removed for further speed/memory improvement
    if mode == 'original':
        scale = (1 << bit) - 1
        res = np.around(scale * input_data)
        int_map = res.copy()
        int_map = int_map.astype(np.uint32)
        res /= scale
    elif mode == 'modified':
        scale = 1 << bit
        res = np.floor(scale * input_data)
        int_map = res.copy()
        int_map = int_map.astype(np.uint32)
        np.clip(res, 0, scale-1, res)
        res /= scale
    else:
        raise ValueError('mode should be either [original] or [modified]')

    assert np.all(res >= 0)
    assert np.all(res <= 1)
    return res, int_map


def _intmap_encode(int_map, bitwidth):
    """ compress the converted int_map to tesnor with fewer numbers"""
    # the int_map is assumed as a 4- or 3-dimensional np.array [b(optional),c,h,w]
    int_map = int_map.flatten()
    # enc_ratio is the number of original values compressed into one single int32 value
    enc_ratio = int(32/bitwidth)

    # store tensor into new_tensor
    # e.g. original tensor with 6 values: [0,1,2,3,4,5] (dtype=int32)
    # new tensor with 2 values: [3,2,1,0], [5,4,NULL,NULL] (enc_ratio=4, one int32 has 4 values)
    int_map_ext = np.append(int_map,
                            np.repeat(0, (enc_ratio - len(int_map) % enc_ratio) % enc_ratio))
    int_map_rs = np.reshape(int_map_ext, (-1, enc_ratio))
    bitshift = np.array([(i % enc_ratio) * bitwidth for i in range(enc_ratio)], dtype=np.uint32)
    int_map_shifted = np.left_shift(int_map_rs, bitshift)
    new_array = np.bitwise_or.reduce(int_map_shifted, axis=1, dtype=np.uint32)

    return new_array


def _intmap_decode(input_data, orig_shape, bitwidth):
    """ restore the compressed tensor """
    # the input is assumed as an 1-dimensional tensor / np.array
    # orig_shape represents the original tensor shape in format of tensor.shape [b(optional),c,h,w]
    enc_ratio = int(32/bitwidth)
    orig_size = np.prod(orig_shape, dtype=np.uint32)

    # restore tensor into original_tensor
    # e.g. new tensor with 2 values: [3,2,1,0], [5,4,NULL,NULL] (enc_ratio=4, one
    #    uint32 will contain 4 uint8 values)
    # original tensor should be 6 values: [0,1,2,3,4,5] (dtype=int32)
    data_exploded = np.repeat(input_data, enc_ratio)
    data_rs = np.reshape(data_exploded, (-1, enc_ratio))
    bitshift = np.array([(i % enc_ratio) * bitwidth for i in range(enc_ratio)], dtype=np.uint32)
    data_shifted = np.right_shift(data_rs, bitshift)
    data_flat = data_shifted.flatten()
    orig_tensor = np.bitwise_and(data_flat, 2**bitwidth-1)
    orig_tensor = orig_tensor[:orig_size]

    # # old version of implementation
    # for idx in range(orig_size):
    #     alt_idx = idx//enc_ratio
    #     alt_bitshift = (idx % enc_ratio) * bitwidth
    #     if (alt_bitshift+bitwidth)!=32:
    #         # mask the higher bits
    #         # e.g. decode the '2' in [3,2,1,0], need to mask the '3' in front of '2'
    #         significant_bits_masked_input = (input[alt_idx] % (1<<(alt_bitshift+bitwidth)))
    #     else:
    #         # e.g. '3' in [3,2,1,0] does not need mask, otherwise would be overflowed
    #         significant_bits_masked_input = input[alt_idx]
    #     orig_tensor[idx] = significant_bits_masked_input >> alt_bitshift

    return orig_tensor.reshape(orig_shape)


def _intmap2float(int_map, bitwidth):
    """ used to restore the tesnor from intmap to float """
    scale = (1 << bitwidth) - 1
    return (int_map/scale).astype(np.float32)

def _uint32_to_uint8(tensor):
    """ re-represent uint32 to uint8, since torch has no uint32 (does have uint8) """
    assert tensor.dtype == np.uint32
    return tensor.view('uint8')

def _uint8_to_uint32(tensor):
    """ restore the uint32 value from 4 uint8 values """
    assert tensor.dtype == np.uint8
    return tensor.view('uint32')


def compression_factor(quant_bit: torch.Tensor) -> torch.Tensor:
    """Compute the compression factor (data size improvement) for quantization bit widths > 0."""
    return torch.div(32, quant_bit)


def tensor_encode(input_data: torch.Tensor, quant_bit: int) -> List[torch.Tensor]:
    """
        The input to the encoder should be a torch.Tensor
        We first cast it to a np.array, then do everything else
    """
    quant_bit_tensor = torch.tensor(quant_bit, dtype = torch.int8)
    if quant_bit == 0:
        return [input_data, torch.tensor(input_data.shape), torch.tensor(1.0), torch.tensor(0.0),
                quant_bit_tensor]

    input_data = input_data.numpy()
    shape = input_data.shape
    # ensure the input is scaled to [0,1],
    shift = input_data.min()
    input_data = input_data - shift
    scale_factor = input_data.max()
    rescale_input = input_data/scale_factor
    # quant
    _, int_map = _quant_op(rescale_input, quant_bit)
    comm_tensor = _intmap_encode(int_map, quant_bit)
    # split uint32 into 4 uint8
    comm_tensor = _uint32_to_uint8(comm_tensor)
    # convert array to tensor for p2p communication
    comm_tensor = torch.tensor(comm_tensor, dtype = torch.uint8)
    shape = torch.tensor(shape, dtype = torch.int32)
    scale_factor = torch.tensor(scale_factor, dtype = torch.float32)
    shift = torch.tensor(shift, dtype = torch.float32)

    # scale_factor is needed to restore the tensor
    return [comm_tensor, shape, scale_factor, shift, quant_bit_tensor]


def tensor_decode(encodings: List[torch.Tensor]) -> torch.Tensor:
    """
        decode the compressed tensor with uint8 value
    """
    comm_tensor, input_shape, scale_factor, shift, quant_bit = encodings
    if quant_bit == 0:
        return comm_tensor

    # convert tensor to array for computation and splice uint8 to uint32
    assert isinstance(comm_tensor, torch.Tensor)
    comm_tensor = _uint8_to_uint32(comm_tensor.numpy())
    input_shape = input_shape.tolist()
    scale_factor = scale_factor.item()
    shift = shift.item()
    quant_bit = quant_bit.item()
    restore_int_map = _intmap_decode(comm_tensor, input_shape, quant_bit)
    restore_tensor = _intmap2float(restore_int_map, quant_bit)
    return torch.from_numpy((restore_tensor*scale_factor+shift).astype(np.float32))


def tensor_encode_outerdim(batched_tensor: torch.Tensor, quant_bit: int) -> List[torch.Tensor]:
    """do quantization on each image in the micro-batched tensor with size [b,c,h,w]"""
    list_of_lists = [tensor_encode(t, quant_bit) for t in batched_tensor]
    encoded_tensors = list(zip(*list_of_lists))
    return [torch.stack(t,0) for t in encoded_tensors]


def tensor_decode_outerdim(batched_encodings: List[torch.Tensor]) -> torch.Tensor:
    """decode the encoded tensor with multiple images in one batch, each encoded image data is in length of 5"""
    tensors = [tensor_decode(encodings) for encodings in zip(*batched_encodings)]
    return torch.stack(tensors, 0)
