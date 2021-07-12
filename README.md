# EdgePipe

## TODO

- [x] Combine the function for different number of nodes

- [ ] Solve the memory leak problem

## Usage

1. Change the world_size, total_rank in runtime.py
2. Activate pytorch enviroment
3. In node with Rank i, use command:

```sh
python runtime.py i
```
eg. in rank 0:

```sh
python runtime.py 0
```


