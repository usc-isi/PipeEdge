from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import time
import torch
import torch.nn as nn
import gc

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
num_threads = 2
batch_size = 7
measure_times = 1
layer_id = 11
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)
model_config = 'google/vit-base-patch16-224'
num_layers = 24
if model_config == 'google/vit-large-patch16-224':
    num_layers = 24
elif model_config == 'google/vit-base-patch16-224':
    num_layers = 12
else:
    num_layers = 32
# number_layers = 12
print(f"number threads  {num_threads}, # layers {num_layers}, batch size {batch_size}, # measure {measure_times}")
image = Image.open(requests.get(url, stream=True).raw)

img = torch.randn(3, 384, 384)
imgs = [image for i in range(batch_size)]
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_config)
print(torch.get_num_threads(),torch.get_num_interop_threads())

inputs = feature_extractor(images=imgs, return_tensors="pt")


class MeasureViTModel(nn.Module):
    def __init__(self):
        super(MeasureViTModel, self).__init__()
        # self.config = config
        
        self.model = ViTForImageClassification.from_pretrained(model_config)
        self.embedding =  list(self.model.vit.children())[0]
        self.layers = []
        for i in range(num_layers):
            self.layers.append(self.model.vit.encoder.layer[i])
        # self.model.vit.encoder.layer = nn.Sequential(*[self.model.vit.encoder.layer[i] for i in range(3, 4)])
        self.layer_id = layer_id
        self.attention = list(self.model.vit.encoder.layer[layer_id].children())[0]
        self.mlp_1 = list(self.model.vit.encoder.layer[layer_id].children())[1]
        self.mlp_2 = list(self.model.vit.encoder.layer[layer_id].children())[2]
        self.layer_norm_in = list(self.model.vit.encoder.layer[layer_id].children())[3:]
        self.layer_norm_in  = nn.Sequential(*self.layer_norm_in)
        self.layer_norm = list(self.model.vit.children())[-1]
        self.classifier = list(self.model.children())[-1]
        print(self.attention, self.mlp_1, self.mlp_2, self.layer_norm_in)
        self.time = [0 for i in range(num_layers+3)]
        self.check_time = 0
        self.measure_time = measure_times
        del self.model

    def forward(self, pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        # return self.classifier(self.layer_norm(self.model(x)[0]))
        for i in range(self.measure_time):
            start = time.time()
            x = self.embedding(pixel_values)
            # print(x)
            mid = time.time()
            self.time[0] += (mid - start)
            # print(mid-start)
            for i in range(num_layers):
                if i ==self.layer_id:
                    mid2 = time.time()
                    # print("before", x.shape)
                    x = self.attention(x)[0]
                    # print("after", x.shape)
                    midm1 = time.time()
                    x_t = self.mlp_1(x)
                    x = self.mlp_2(x_t, x)
                    x = self.layer_norm_in(x)
                    mid3 = time.time()
                    print(f"Layer {self.layer_id}: attention: {midm1 - mid2}, mlp:{mid3 - midm1}, total is {mid3- mid2}")
                
                transformer_layer = self.layers[i]
                # print(transformer_layer)
                mid2 = time.time()
                x = transformer_layer(x)[0]
                mid3 = time.time()
                # print(mid3 - mid2)
                self.time[i+1] += (mid3-mid2)
                print(f"Layer {i}, time is {(mid3-mid2)}")
            mid4 = time.time()
            x = self.layer_norm(x)
            mid5 = time.time()
            self.time[num_layers+1] += (mid5 - mid4)
            mid6 = time.time()
            x = self.classifier(x)
            mid7 = time.time()
            self.time[num_layers+2] += (mid7 - mid6)
            gc.collect()
        for i in range(num_layers+3):
            self.time[i] = self.time[i]/self.measure_time
            self.check_time += self.time[i]
            print(self.time[i])
        print(self.time)
        print(f"check time is {self.check_time}")
        return self.time


for layer_id in range(0,12):
    vit_measure = MeasureViTModel()
    time_statics = vit_measure(**inputs)


