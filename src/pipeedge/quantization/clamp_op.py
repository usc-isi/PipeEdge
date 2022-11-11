"""clamp functions used for clipping quantization methods"""
from scipy.special import lambertw
import torch


def _clamp_factor_gelu(bit: int) -> torch.Tensor:
    # scipy returns a float64, but we'll overflow first if we don't force it when bit>=31
    return lambertw(3 * torch.tensor(4, dtype=torch.float64)**(bit+1)).real


def clamp_banner2019_gelu(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Like `clamp_banner2019_laplace` but modified for a GeLU layer output."""
    # Special case for GeLU layer
    # Distribution after GeLU only has half of bell curve
    # Assuming mean = 0, and ignore the influence of negtive small values
    variance = 2* torch.pow(tensor, 2).sum()/torch.numel(tensor)
    dist_parameter = torch.sqrt(0.5*variance)
    alpha = _clamp_factor_gelu(bit).to(tensor) * dist_parameter
    return tensor.clamp(min=-alpha, max=alpha)


def _clamp_factor_laplace(bit: int) -> torch.Tensor:
    # scipy returns a float64, but we'll overflow first if we don't force it when bit>=32
    return lambertw(3 * torch.tensor(4, dtype=torch.float64)**bit).real


def clamp_banner2019_laplace(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Clamp tensor with a Laplace distribution - based on Banner et. al.'s NIPS 2019 paper."""
    # "Post training 4-bit quantization of convolutional networks for rapid-deployment"
    variance = torch.var(tensor, unbiased = False)
    dist_parameter = torch.sqrt(0.5*variance)
    alpha = _clamp_factor_laplace(bit).to(tensor) * dist_parameter
    return tensor.clamp(min=-alpha, max=alpha)
