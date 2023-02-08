"""Hook functions for lz4 compression/decompression."""
import logging
import os
import pickle
import shutil
import subprocess
import numpy as np
import torch

# User-specified lz4, e.g., to switch implementations for binary diversification
ENV_LZ4_BINARY: str = 'LZ4_BINARY'

logger = logging.getLogger(__name__)

def _get_lz4_binary() -> str:
    lz4 = os.getenv('ENV_LZ4_BINARY')
    if lz4 is None:
        lz4 = shutil.which('lz4')
    if lz4 is None:
        raise RuntimeError('Failed to find lz4')
    return lz4

def forward_pre_lz4_decompress(_module, inputs):
    """Decompresss `inputs` to their original type using a lz4 subprocess and pickle."""
    assert isinstance(inputs, tuple)
    assert len(inputs) == 1
    assert isinstance(inputs[0], torch.Tensor)
    lz4 = _get_lz4_binary()
    bytes_in = inputs[0].numpy().tobytes()
    args = [lz4, '-cd']
    logger.info("%s: decompress", lz4)
    bzip_dec = subprocess.run(args, capture_output=True, input=bytes_in, check=True)
    return (pickle.loads(bzip_dec.stdout),)

def forward_hook_lz4_compress(_module, _inputs, outputs):
    """Compresss `outputs` into a uint8 tensor using pickle and a lz4 subprocess."""
    lz4 = _get_lz4_binary()
    bytes_out = pickle.dumps(outputs)
    args = [lz4, '-cz']
    logger.info("%s: compress", lz4)
    bzip_com = subprocess.run(args, capture_output=True, input=bytes_out, check=True)
    return torch.tensor(np.frombuffer(bzip_com.stdout, dtype=np.uint8))
