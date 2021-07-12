# EdgePipe

## TODO

- [x] Combine the function for different number of nodes
- [ ] Test for multiple nodes 
- [ ] Test for vit-large model
- [ ] Test for vit-huge model
- [ ] Solve the memory leak problem

## Usage

1. Change the world_size, total_rank in runtime.py
2. Change the MASTER_ADDR, MASTER_PORT, TP_SOCKET_IFNAME, GLOO_SOCKET_IFNAME in runtime.py 
3. Activate pytorch enviroment
4. In node with Rank i, use command:

```sh
python runtime.py i
```
eg. in rank 0:

```sh
python runtime.py 0
```


