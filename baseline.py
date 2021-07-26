from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import torch
import time
## Force pytorch use CPU
device = torch.device('cpu')
# parallel_threads = 2
# torch.set_num_threads(parallel_threads)
# torch.set_num_interop_threads(parallel_threads)
torch.set_grad_enabled(False)
print(f"Use device: {device},  # parallel intra nodes threads: {torch.get_num_threads()}, # parallel inter nodes threads: {torch.get_num_interop_threads()}")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
batch_size = 1
num_batch = 1
images = [image for i in range(batch_size)]
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')


inputs = feature_extractor(images=images, return_tensors="pt")
print(inputs)
start = time.time()
for i in range(num_batch):
    print(f"Start processing {i} batch")  
    outputs = model(**inputs)
    # print(outputs)
end = time.time()
print(f"Exec time is {end-start}, throughput is {(num_batch*batch_size)/(end-start)}")
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
