# pylint: disable
""" Test quantization mosule """
import unittest
import numpy as np
import torch
from pipeedge.quantization.basic_op import _quant_op, _intmap_encode, _intmap_decode, _intmap2float

def quant_func(input_shape, quant_bit):
    assert quant_bit >= 0
    assert quant_bit <= 32
    b,h,w = input_shape
    input_tensor = torch.rand(b,h,w, dtype=torch.float32) #8,768,3072
    input_array = input_tensor.numpy()
    output, int_map = _quant_op(input_array, quant_bit)
    comm_tensor = _intmap_encode(int_map, quant_bit)
    restore_int_map = _intmap_decode(comm_tensor, input_array.shape, quant_bit)
    restore_tensor = _intmap2float(restore_int_map, quant_bit)
    return np.all((restore_tensor-output) < 1e-6)

class TestQuantCorrect(unittest.TestCase):
    """test quant operations' correctness"""
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.test_bit_list = [1,2,3,4,5,6,8,16,32]
        self.input_shapes = [[8,10,10], [1, 100, 100], [8, 197, 786]]

    def test_correctness(self):
        for shape in self.input_shapes:
            for bit in self.test_bit_list:
                self.assertTrue(quant_func(shape, bit))
