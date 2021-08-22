import torch
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
import torch.nn as nn

import os
import psutil
import gc

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.Sequential(*[nn.Linear(1000, 1000) for _ in range(2)])

    def forward(self, x):
        return self.nets(x)


options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=1, rpc_timeout=300)
rpc.init_rpc("worker0", world_size=1, rank=0, rpc_backend_options=options)

model_rref = rpc.remote("worker0", MyModel)

process = psutil.Process(os.getpid())
for i in range(10):
    with dist_autograd.context() as context_id:
        futs = []
        for _ in range(20):
            y = model_rref.rpc_sync().forward(torch.zeros(10000, 1000))
            futs.append(model_rref.rpc_async().forward(y))

        outputs = torch.cat(torch.futures.wait_all(futs))
        dist_autograd.backward(context_id, [outputs.sum()])

    del futs
    del outputs
    gc.collect()
    print(f"Round {i}: memory {process.memory_info().rss // 1000000} MB")

rpc.shutdown()

gc.collect()
print(f"Final: memory {process.memory_info().rss // 1000000} MB")