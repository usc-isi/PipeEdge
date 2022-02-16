from transformers import ViTFeatureExtractor, ViTModel,ViTForImageClassification
from PIL import Image
import requests
import torch
import time
import numpy as np


def save_weights(model, file_name):

    state_dict = model.state_dict()
    weights = {}
    for k, v in state_dict.items():
        print(k)
        weights[k] = v
    np.savez(file_name, **weights)

if __name__=="__main__":
    pth_file = "../deit_base_structure_40_82.22.pth"
    net = torch.load(pth_file,map_location=torch.device('cpu'))
    weights = {}
    for key,value in net["model"].items():
        print(key,value.size(),sep="   ")
        weights[key] = value
        np.savez("SViT_B.npz", **weights)
    # model_b = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # model_l = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')

    # model_h = ViTForImageClassification.from_pretrained('google/vit-huge-patch14-224-in21k')
    # save_weights(model_b, "ViT_B.npz")
    # save_weights(model_l, "ViT_L.npz")
    # save_weights(model_h, "ViT_H.npz")

    
    