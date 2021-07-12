import os
import sys
import gc
import threading
import time
import torch
from functools import wraps
from typing import Optional
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.nn import functional as F
from transformers import AutoConfig, ViTFeatureExtractor, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer
from PIL import Image
import requests
from runtime import TransformerBase

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

#########################################################
#                   Test Transformers                   #
#########################################################
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
print(feature_extractor)
print(model)
print("patch_embedding.projection weight is",model.vit.embeddings.patch_embeddings.projection.weight)
inputs = feature_extractor(images=image, return_tensors="pt")
print("inputs weight is", inputs)
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

#########################################################
#                   Test runtime model                  #
#########################################################
model_name = "google/vit-base-patch16-224"
is_first = True
is_last = True
start_layer = 0
end_layer = 12
load_weight = True
model = TransformerBase(model_name, is_first, is_last, start_layer, end_layer, load_weight)
# print(model)
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(inputs['pixel_values'])
# logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = outputs .argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])