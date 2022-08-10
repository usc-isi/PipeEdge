"""clamp functions used for clipping quantization methods"""
from multiprocessing.sharedctypes import Value
import torch
from torch import Tensor

_CLAMP_FACTOR_Laplace = {
            2: 2.83,
            3: 3.89,
            4: 5.03,
            5: 6.20,
            6: 7.41,
            8: 9.90
    }
_CLAMP_FACTOR_GeLU = {
            2: 3.897,
            3: 5.029,
            4: 6.205,
            5: 7.41,
            6: 8.646,
            8: 11.163
    }

def clamp_PTQ_GeLU(input_tensor: Tensor, bit: int):
    """clamp for GeLU layers"""
    # Special case for GeLU layer
    # Distribution after GeLU only has half of bell curve
    # Assuming mean = 0, and ignore the influence of negtive small values
    variance = 2* torch.pow(input_tensor, 2).sum()/torch.numel(input_tensor)
    dist_parameter = torch.sqrt(0.5*variance)
    optimal_clamp_range = _CLAMP_FACTOR_GeLU[bit] * dist_parameter
    # clamp
    result = torch.where(torch.abs(input_tensor)<optimal_clamp_range, input_tensor, optimal_clamp_range)

    return result


def clamp_PTQ_RB2019(input_tensor: Tensor, bit: int):
    """clamp a input tensor"""
    variance = torch.var(input_tensor, unbiased = False)
    dist_parameter = torch.sqrt(0.5*variance)
    optimal_clamp_range = _CLAMP_FACTOR_Laplace[bit] * dist_parameter
    # clamp
    result = torch.where(torch.abs(input_tensor)<optimal_clamp_range, input_tensor, optimal_clamp_range)

    return result
