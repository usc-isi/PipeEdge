from transformers import ViTFeatureExtractor, ViTModel,ViTForImageClassification
from PIL import Image
import requests
import torch
import time
import numpy as np
import cProfile
## Force pytorch use CPU
device = torch.device('cpu')
# parallel_threads = 2
# torch.set_num_threads(parallel_threads)
# torch.set_num_interop_threads(parallel_threads)
torch.set_grad_enabled(False)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
batch_size = 64
num_batch = 1
images = [image for i in range(batch_size)]
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')


inputs = feature_extractor(images=images, return_tensors="pt")
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
print(model)
def forward():

    for i in range(1):
        outputs = model(**inputs)
        print(f"Start processing {i} batch") 
        print(outputs.logits) 
    return outputs

def main():
    print(f"Use device: {device},  # parallel intra nodes threads: {torch.get_num_threads()}, # parallel inter nodes threads: {torch.get_num_interop_threads()}")

    # print(inputs)
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # state_dict = model.state_dict()
    # weights = {}
    # for k, v in state_dict.items():
    #     print(k)
    #     weights[k] = v
    #     print(v.shape)
        # if k == "encoder.layer.6.output.dense.bias":
        #     print(v)
    # np.savez("ViT_B.npz", **weights)
    start = time.time()
    # outputs = model(**inputs)
    cProfile.run("forward()", 'restats_b')

    end = time.time()
    print(f"Exec time is {end-start}, throughput is {(num_batch*batch_size)/(end-start)}")
    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape)

if __name__=="__main__":
    main()
    