"""RPC transformers."""
from torch.distributed import rpc
from . import DistRpcModule
from ...models.transformers.deit import DeiTTransformerShard
from ...models.transformers.bert import BertTransformerShard
from ...models.transformers.vit import ViTTransformerShard


class BertDistRpcTransformer(DistRpcModule):
    """BERT distributed RPC transformer."""

    def __init__(self, model_name, model_file, stage_ranks, partition):
        super().__init__()
        for i, dst_rank in enumerate(stage_ranks):
            # Build Transformer Shard
            is_first = i == 0
            is_last = i == len(stage_ranks) - 1
            rref = rpc.remote(dst_rank, BertTransformerShard,
                              args=(i, model_name, model_file, is_first, is_last, partition[2*i],
                                    partition[2*i+1], True))
            self._rref_list.append(rref)
        self._register_hooks()


class DeiTDistRpcTransformer(DistRpcModule):
    """DeiT distributed RPC transformer."""

    def __init__(self, model_name, model_file, stage_ranks, partition):
        super().__init__()
        for i, dst_rank in enumerate(stage_ranks):
            is_first = i == 0
            is_last = i == len(stage_ranks) - 1
            rref = rpc.remote(dst_rank, DeiTTransformerShard,
                              args=(i, model_name, model_file, is_first, is_last, partition[2*i],
                                    partition[2*i+1], True))
            self._rref_list.append(rref)
        self._register_hooks()


class ViTDistRpcTransformer(DistRpcModule):
    """ViT distributed RPC transformer."""

    def __init__(self, model_name, model_file, stage_ranks, partition):
        super().__init__()
        for i, dst_rank in enumerate(stage_ranks):
            # Build Transformer Shard
            is_first = i == 0
            is_last = i == len(stage_ranks) - 1
            rref = rpc.remote(dst_rank, ViTTransformerShard,
                              args=(i, model_name, model_file, is_first, is_last, partition[2*i],
                                    partition[2*i+1], True))
            self._rref_list.append(rref)
        self._register_hooks()
