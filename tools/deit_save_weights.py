import numpy as np
import torch


def save_weights(model, file_name):
    state_dict = model.state_dict()
    weights = {}
    for k, v in state_dict.items():
        print(k)
        weights[k] = v
    np.savez(file_name, **weights)

if __name__=="__main__":
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224',
                           pretrained=True)
    save_weights(model, 'DeiT_B_distilled.npz')
    model = torch.hub.load('facebookresearch/deit:main', 'deit_small_distilled_patch16_224',
                           pretrained=True)
    save_weights(model, 'DeiT_S_distilled.npz')
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224',
                           pretrained=True)
    save_weights(model, 'DeiT_T_distilled.npz')
