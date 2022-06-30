"""Models module."""
from torch import nn

class ModuleShard(nn.Module):
    """Abstract parent class for module shards."""
    # pylint: disable=abstract-method

    def __init__(self, stage: int, start_layer: int, end_layer: int):
        super().__init__()
        self.stage = stage
        self.start_layer = start_layer
        self.end_layer = end_layer
