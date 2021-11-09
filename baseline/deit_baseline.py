import torch
# check you have the right version of timm
import timm
import time 
import numpy as np
import argparse
# assert timm.__version__ == "0.3.2"

# now load it with torchhub
def measure_throughput(model_name, batchsize):
    model = torch.hub.load('facebookresearch/deit:main', model_name , pretrained=True)
    # model.load_state_dict(torch.load('../deit_base_structure_40_82.22.pth', map_location=torch.device('cpu'))['model'])
    model.eval()
    # state_dict = model.state_dict()
    # weights = {}
    # for k, v in state_dict.items():
    #     print(k, v.size())
    #     weights[k] = v
        # if k == "blocks.0.attn.qkv.weight":
        #     print(v[0, 0])

    x = torch.rand(batchsize,3,224,224)
    # print(model)
    start = time.time()
    model(x)
    end = time.time()
    print(f"Throughput is {batchsize/(end-start)}")
    # np.savez("../DeiT_B_Distilled.npz", **weights)
    # print(model)
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="deit baseline")
    parser.add_argument("-m", "--model-name", type=str, default='deit_tiny_distilled_patch16_224', choices=['deit_small_distilled_patch16_224', "DeiT-Base", 
    'deit_base_distilled_patch16_224'], help="the neural network model for loading")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")    
    args = parser.parse_args()
    model_name = args.model_name
    batchsize = args.batch_size
    measure_throughput(model_name, batchsize)


