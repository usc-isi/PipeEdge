# EdgePipe
Requirement:
PyTorch > 1.8.1

Install [transformers](https://huggingface.co/transformers/installation.html) library with:
```sh
pip install transformers
```


## Usage

1. Change the model_name, world_size, total_rank, **partition** method in runtime.py
2. Change the MASTER_ADDR, MASTER_PORT, TP_SOCKET_IFNAME, GLOO_SOCKET_IFNAME in runtime.py 
3. Activate pytorch enviroment
4. Download weight file from [Google Cloud](https://console.cloud.google.com/storage/browser/vit_models;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
5. In node with Rank i, use command:

```sh
python runtime.py i
```
eg. in rank 0:

```sh
python runtime.py 0
```

**Note: support operation level partition**
For example:

ViT-Base model has 12 layers, the range is [1, 12*4] = [1, 48]

Even partitioning for 2 nodes:
```
partition = [1, 24,   25, 48]
```
Uneven partitioning for 2 nodes:

```
partition = [1, 47, 48,48]
```

For 4 nodes partition:

```
partition = [1, 4, 5,8, 9, 20, 21,48]
```



## TODO

- [x] Remove unneccessary code
- [x] Build layers in the node and reload the weight 
- [x] Combine the function for different number of nodes
- [x] Test for correctness
- [x] Test for multiple nodes 
- [x] Test for vit-large model
- [x] Test for vit-huge model
- [x] Support fine-grained partitioning
- [x] Add baseline.py   
- [x] Operator-level partition method
- [ ] Edit profile script
- [ ] Import profile to Partition
- [ ] Edit Partition script
- [ ] Import Partition to Runtime
- [ ] Solve the memory leak problem ([RPC Framework Problem](https://github.com/pytorch/pytorch/issues/61920#issuecomment-886345414))


