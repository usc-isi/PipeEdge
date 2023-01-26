"""Reverse auction scheduling."""
from typing import List, Tuple, Type
from . import computation_time, mem_bytes

ShardBid: Type = Tuple[Tuple[int, int], float]
"""A shard bid has the form: `((start_layer, end_layer), cost)`."""

def bid_latency(yml_model: dict, yml_dev_type: dict, yml_dtm_profile: dict, ubatch_size: int,
                dtype: str='torch.float32') -> List[ShardBid]:
    """Bid for shards using latency as the cost metric."""
    bids = []
    dt_mem_bytes = yml_dev_type['mem_MB'] * 1024 * 1024
    for layer_l in range(yml_model['layers']):
        for layer_r in range(layer_l, yml_model['layers']):
            bytes_req = mem_bytes(yml_model, layer_l, layer_r, dtype, ubatch_size)
            if dt_mem_bytes > bytes_req:
                cost = computation_time(yml_dtm_profile, layer_l, layer_r)
                bids.append(((layer_l, layer_r), cost))
    return bids
