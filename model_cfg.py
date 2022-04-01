"""Model configurations and default parameters."""
from edgepipe.comm.rpc.transformers import (
    BertDistRpcModule, DeiTDistRpcModule, ViTDistRpcModule
)
from edgepipe.models.transformers.bert import BertTransformerShard
from edgepipe.models.transformers.deit import DeiTTransformerShard
from edgepipe.models.transformers.vit import ViTTransformerShard

_MODEL_CONFIGS = {}

def _model_cfg_add(name, layers, weights_file, shard_module, rpc_module):
    _MODEL_CONFIGS[name] = {
        'name': name,
        'layers': layers,
        'weights_file': weights_file,
        'shard_module': shard_module,
        'rpc_module': rpc_module,
    }

# Transformer blocks can be split 4 ways, e.g., where ViT-Base has 12 layers, we specify 12*4=48
_model_cfg_add('google/vit-base-patch16-224', 48, 'ViT-B_16-224.npz',
               ViTTransformerShard, ViTDistRpcModule)
_model_cfg_add('google/vit-large-patch16-224', 96, 'ViT-L_16-224.npz',
               ViTTransformerShard, ViTDistRpcModule)
_model_cfg_add('google/vit-huge-patch14-224-in21k', 128, 'ViT-H_14.npz',
               ViTTransformerShard, ViTDistRpcModule)
_model_cfg_add('bert-base-uncased', 48, 'BERT-B.npz',
               BertTransformerShard, BertDistRpcModule)
_model_cfg_add('bert-large-uncased', 96, 'BERT-L.npz',
               BertTransformerShard, BertDistRpcModule)
_model_cfg_add('facebook/deit-base-distilled-patch16-224', 48, 'DeiT_B_distilled.npz',
               DeiTTransformerShard, DeiTDistRpcModule)
_model_cfg_add('facebook/deit-small-distilled-patch16-224', 48, 'DeiT_S_distilled.npz',
               DeiTTransformerShard, DeiTDistRpcModule)
_model_cfg_add('facebook/deit-tiny-distilled-patch16-224', 48, 'DeiT_T_distilled.npz',
               DeiTTransformerShard, DeiTDistRpcModule)

def get_model_names():
    """Get a list of available model names."""
    return list(_MODEL_CONFIGS.keys())

def get_model_dict(model_name):
    """Get a model's key/value properties - modify at your own risk."""
    return _MODEL_CONFIGS[model_name]

def get_model_layers(model_name):
    """Get a model's layer count."""
    return _MODEL_CONFIGS[model_name]['layers']

def get_model_default_weights_file(model_name):
    """Get a model's default weights file name."""
    return _MODEL_CONFIGS[model_name]['weights_file']

def module_shard_factory(model_name, model_file, layer_start, layer_end, stage):
    """Get a shard instance."""
    # This works b/c all shard implementations have the same constructor interface
    is_first = layer_start == 1
    is_last = layer_end == get_model_layers(model_name)
    module = _MODEL_CONFIGS[model_name]['shard_module']
    return module(stage, model_name, model_file, is_first, is_last, layer_start, layer_end, True)

def dist_rpc_module_factory(model_name, model_file, stage_ranks, partition):
    """Get a RPC instance."""
    # This works b/c all dist RPC implementations have the same constructor interface
    module = _MODEL_CONFIGS[model_name]['rpc_module']
    return module(model_name, model_file, stage_ranks, partition)
