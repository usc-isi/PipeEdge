"""clamp functions used for clipping quantization methods"""
import torch

_CLAMP_FACTOR_LAPLACE = {
    2: 2.83,
    3: 3.89,
    4: 5.03,
    5: 6.20,
    6: 7.41,
    8: 9.90,
    16: 20.27
}

_CLAMP_FACTOR_GELU = {
    2: 3.897,
    3: 5.029,
    4: 6.205,
    5: 7.41,
    6: 8.646,
    8: 11.163,
    16: 21.59
}

def bitwidths_banner2019_gelu() -> torch.Tensor:
    """Get available bitwidths."""
    return torch.tensor(list(_CLAMP_FACTOR_GELU.keys()))


def clamp_banner2019_gelu(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Like `clamp_banner2019_laplace` but modified for a GeLU layer output."""
    # Special case for GeLU layer
    # Distribution after GeLU only has half of bell curve
    # Assuming mean = 0, and ignore the influence of negtive small values
    variance = 2* torch.pow(tensor, 2).sum()/torch.numel(tensor)
    dist_parameter = torch.sqrt(0.5*variance)
    optimal_clamp_range = _CLAMP_FACTOR_GELU[bit] * dist_parameter
    result = torch.where(torch.abs(tensor)<optimal_clamp_range, tensor, optimal_clamp_range)
    return result


def bitwidths_banner2019_laplace() -> torch.Tensor:
    """Get available bitwidths."""
    return torch.tensor(list(_CLAMP_FACTOR_LAPLACE.keys()))


def clamp_banner2019_laplace(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Clamp tensor with a Laplace distribution - based on Banner et. al.'s NIPS 2019 paper."""
    # "Post training 4-bit quantization of convolutional networks for rapid-deployment"
    variance = torch.var(tensor, unbiased = False)
    dist_parameter = torch.sqrt(0.5*variance)
    optimal_clamp_range = _CLAMP_FACTOR_LAPLACE[bit] * dist_parameter
    result = torch.where(torch.abs(tensor)<optimal_clamp_range, tensor, optimal_clamp_range)
    return result
