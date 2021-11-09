import torch
import numpy as np

pth_file = "../deit_base_structure_40_82.22.pth"
net = torch.load(pth_file,map_location=torch.device('cpu'))
for key,value in net["model"].items():
    print(key,value.size(),sep="   ")