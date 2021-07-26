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

**Note: support fine-grained partition with x.5**
eg. for two layers (layer 0 and layer 1), use:
```sh
partition = [0, 0.5, 0.5, 1]
```
The include layer for each node is [start_layer, end_layer], thus for even partitioning among 4 nodes, use:

```sh
partition = [0,2,  3,5,  6,8,  9,11]
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
- [ ] Edit profile script
- [ ] Edit Partition script
- [ ] Solve the memory leak problem ([RPC Framework Problem](https://github.com/pytorch/pytorch/issues/61920#issuecomment-886345414))


