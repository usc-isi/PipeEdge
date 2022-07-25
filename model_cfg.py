"""Model configurations and default parameters."""
from typing import Any, Callable, List, Optional, Tuple
from torch.distributed import rpc as trpc
from pipeedge.comm import p2p, rpc
from pipeedge.models import ModuleShard
from pipeedge.models.transformers.bert import BertTransformerShard
from pipeedge.models.transformers.deit import DeiTTransformerShard
from pipeedge.models.transformers.vit import ViTTransformerShard
import devices

_MODEL_CONFIGS = {}

def _model_cfg_add(name, layers, weights_file, shard_module):
    _MODEL_CONFIGS[name] = {
        'name': name,
        'layers': layers,
        'weights_file': weights_file,
        'shard_module': shard_module,
    }

# Transformer blocks can be split 4 ways, e.g., where ViT-Base has 12 layers, we specify 12*4=48
_model_cfg_add('google/vit-base-patch16-224', 48, 'ViT-B_16-224.npz',
               ViTTransformerShard)
_model_cfg_add('google/vit-large-patch16-224', 96, 'ViT-L_16-224.npz',
               ViTTransformerShard)
_model_cfg_add('google/vit-huge-patch14-224-in21k', 128, 'ViT-H_14.npz',
               ViTTransformerShard)
_model_cfg_add('bert-base-uncased', 48, 'BERT-B.npz',
               BertTransformerShard)
_model_cfg_add('bert-large-uncased', 96, 'BERT-L.npz',
               BertTransformerShard)
_model_cfg_add('facebook/deit-base-distilled-patch16-224', 48, 'DeiT_B_distilled.npz',
               DeiTTransformerShard)
_model_cfg_add('facebook/deit-small-distilled-patch16-224', 48, 'DeiT_S_distilled.npz',
               DeiTTransformerShard)
_model_cfg_add('facebook/deit-tiny-distilled-patch16-224', 48, 'DeiT_T_distilled.npz',
               DeiTTransformerShard)

def get_model_names() -> List[str]:
    """Get a list of available model names."""
    return list(_MODEL_CONFIGS.keys())

def get_model_dict(model_name: str) -> dict:
    """Get a model's key/value properties - modify at your own risk."""
    return _MODEL_CONFIGS[model_name]

def get_model_layers(model_name: str) -> int:
    """Get a model's layer count."""
    return _MODEL_CONFIGS[model_name]['layers']

def get_model_default_weights_file(model_name: str) -> str:
    """Get a model's default weights file name."""
    return _MODEL_CONFIGS[model_name]['weights_file']

def module_shard_factory(model_name: str, model_file: str, layer_start: int, layer_end: int,
                         stage: int) -> ModuleShard:
    """Get a shard instance on the globally-configured `devices.DEVICE`."""
    # This works b/c all shard implementations have the same constructor interface
    is_first = layer_start == 1
    is_last = layer_end == get_model_layers(model_name)
    module = _MODEL_CONFIGS[model_name]['shard_module']
    shard = module(stage, model_name, model_file, is_first, is_last, layer_start, layer_end, True)
    shard.to(device=devices.DEVICE)
    return shard

def _dist_rpc_pipeline_stage_factory(*args, **kwargs) -> rpc.DistRpcPipelineStage:
    """Get a `rpc.DistRpcPipelineStage` instance on the globally-configured `devices.DEVICE`."""
    stage = rpc.DistRpcPipelineStage(*args, **kwargs)
    stage.module_to(device=devices.DEVICE)
    return stage

def dist_rpc_pipeline_factory(model_name: str, model_file: str, stage_ranks: List[int],
                              stage_layers: List[Tuple[int, int]], results_to: int,
                              results_cb: Callable[[Any], None]) -> rpc.DistRpcPipeline:
    """Get an RPC pipeline instance."""
    # This works b/c all shard implementations have the same constructor interface
    module = _MODEL_CONFIGS[model_name]['shard_module']
    stage_rrefs = []
    assert len(stage_ranks) > 0
    assert len(stage_ranks) == len(stage_layers)
    for i, (dst_rank, layers) in enumerate(zip(stage_ranks, stage_layers)):
        is_first = i == 0
        is_last = i == len(stage_ranks) - 1
        module_args = (i, model_name, model_file, is_first, is_last, layers[0], layers[1], True)
        rref = trpc.remote(dst_rank, _dist_rpc_pipeline_stage_factory, args=(module,),
                           kwargs={ 'module_args': module_args })
        stage_rrefs.append(rref)
    return rpc.DistRpcPipeline(stage_rrefs, results_to, results_cb)

def dist_p2p_pipeline_stage_factory(stage_ranks: List[int], data_rank: int, rank: int,
                                    stage: Optional[int], module: Optional[ModuleShard],
                                    handle_results_cb: Callable[[Any], None]) \
    -> p2p.DistP2pPipelineStage:
    """Get a P2P pipeline stage instance."""
    if rank == data_rank:
        if stage is None:
            # We're data_rank w/out a module shard
            rank_src = stage_ranks[-1]
            rank_dst = stage_ranks[0]
            work_cb = None
        else:
            # We're simultaneously data_rank and a pipeline stage
            # In this case, the current p2p design requires that we must be the first stage
            assert stage == 0
            # Degenerate case when we're both data_rank and the only stage
            rank_src = stage_ranks[-1] if len(stage_ranks) > 1 else None
            rank_dst = stage_ranks[1] if len(stage_ranks) > 1 else None
            work_cb = module
        # While the handle_results_cb parameter isn't optional, we should assert it anyway.
        # If None, DistP2pPipelineStage would loop results back to its input queue, then the first
        # module shard would try to process the results tensors, which it would fail to unpack.
        # It wouldn't be obvious from the error that the real problem was handle_results_cb=None.
        assert handle_results_cb is not None
        results_cb = handle_results_cb
    elif stage is None:
        # We're completely idle
        rank_src = None
        rank_dst = None
        work_cb = None
        results_cb = None
    else:
        # We're not data_rank, but we have a module shard (possibly first and/or last stage)
        rank_src = data_rank if stage == 0 else stage_ranks[(stage - 1)]
        rank_dst = data_rank if stage == len(stage_ranks) - 1 else stage_ranks[(stage + 1)]
        work_cb = module
        results_cb = None
    return p2p.DistP2pPipelineStage(rank_src, rank_dst, work_cb, results_cb)
