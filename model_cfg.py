"""Model configurations and default parameters."""

_MODEL_CONFIGS = {}

def _model_cfg_add(name, layers, file):
    _MODEL_CONFIGS[name] = {
        'name': name,
        'layers': layers,
        'file': file,
    }

# Transformer blocks can be split 4 ways, e.g., where ViT-Base has 12 layers, we specify 12*4=48
_model_cfg_add('google/vit-base-patch16-224', 48, 'ViT-B_16-224.npz')
_model_cfg_add('google/vit-large-patch16-224', 96, 'ViT-L_16-224.npz')
_model_cfg_add('google/vit-huge-patch14-224-in21k', 128, 'ViT-H_14.npz')
_model_cfg_add('bert-base-uncased', 48, 'BERT-B.npz')
_model_cfg_add('bert-large-uncased', 96, 'BERT-L.npz')
_model_cfg_add('facebook/deit-base-distilled-patch16-224', 48, 'DeiT_B_distilled.npz')
_model_cfg_add('facebook/deit-small-distilled-patch16-224', 48, 'DeiT_S_distilled.npz')
_model_cfg_add('facebook/deit-tiny-distilled-patch16-224', 48, 'DeiT_T_distilled.npz')

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
    return _MODEL_CONFIGS[model_name]['file']
