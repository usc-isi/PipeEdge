"""Scheduling module.

In PipeEdge, layers are usually specified to be in range [1, num_layers] for legacy reasons.
In these functions, layer shards are specified in range [0, num_layers - 1].
For reference, see: src-native/schedule.cpp (which uses range [1, num_layers]).
"""

_DTYPE_BYTES_SIZES = {
    # Currently only supporting float32
    'torch.float32': 4,
}
def _dtype_bytes(dtype: str) -> int:
    """Get the number of bytes for a single value."""
    return _DTYPE_BYTES_SIZES[dtype]


def ubatch_bytes(n_params: int, ubatch_size: int, dtype: str='torch.float32') -> int:
    """Compute the bytes required for a microbatch buffer."""
    return n_params * ubatch_size * _dtype_bytes(dtype)


def mem_bytes(yml_model: dict, layer_l: int, layer_r: int, dtype: str, ubatch_size: int,
              data_buffers_in: int=2, data_buffers_out: int=2) -> int:
    """Estimate the memory required for a complete stage."""
    assert len(yml_model['mem_MB']) == len(yml_model['parameters_out'])
    assert layer_l >= 0
    assert layer_l <= layer_r
    assert layer_r < len(yml_model['mem_MB'])
    # Memory used by model weights
    mem_bytes_model = sum(yml_model['mem_MB'][layer_l: layer_r + 1]) * 1024 * 1024
    # Memory for model in/out data buffers
    params_in = yml_model['parameters_in'] if layer_l == 0 else \
        yml_model['parameters_out'][layer_l - 1]
    dat_bytes_in = ubatch_bytes(params_in, ubatch_size, dtype=dtype)
    dat_bytes_out = ubatch_bytes(yml_model['parameters_out'][layer_r], ubatch_size, dtype=dtype)
    # Communication and processing memory buffer overheads: send/recv/queue/processing buffers
    # Temporary processing buffers not accounted for - that's a function of the model impl
    # data_buffers_{in,out}: 1 for in-flight data exchanges, 1 for queues (p2p comm only)
    mem_bytes_buffers = 0
    # Receive buffer (and maybe queue)
    if layer_l > 0:
        mem_bytes_buffers += dat_bytes_in * data_buffers_in
    # send buffer (and maybe queue)
    mem_bytes_buffers += dat_bytes_out * data_buffers_out
    # processing buffers (data not in queues or send/recv threads)
    mem_bytes_buffers += dat_bytes_in + dat_bytes_out
    # Sum it up
    return mem_bytes_model + mem_bytes_buffers


def computation_time(yml_model_profile: dict, layer_l: int, layer_r: int) -> float:
    """Compute the time in seconds required to process a layer range."""
    assert layer_l >= 0
    assert layer_l <= layer_r
    time_s = yml_model_profile['time_s']
    assert layer_r < len(time_s)
    return sum(time_s[layer_l: layer_r + 1])


def communication_time(yml_device_type: dict, data_bytes: int) -> float:
    """Compute communication time in seconds."""
    return communication_time_bw(yml_device_type['bw_Mbps'], data_bytes)


def communication_time_bw(bw_mbits_sec: float, data_bytes: int) -> float:
    """Compute communication time in seconds."""
    # Distinguish between bytes and bits
    bytes_sec = bw_mbits_sec * 1024 * 1024 / 8
    return data_bytes / bytes_sec
