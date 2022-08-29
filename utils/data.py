"""Data utilities."""
from typing import Tuple
import torch
from torch.utils.data import Dataset

class RolloverTensorDataset(Dataset[Tuple[torch.Tensor, ...]]):
    """Like `TensorDataset`, but rolls over when the requested length exceeds the actual length."""

    def __init__(self, length: int, *tensors: torch.Tensor):
        assert all(tensors[0].size(0) == t.size(0) for t in tensors), \
               "Size mismatch between tensors"
        self.length = length
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index % len(t)] for t in self.tensors)

    def __len__(self):
        return self.length
