## Usage
1. install [CNPY](https://github.com/rogersce/cnpy)
2. Compile with

```bash
g++  partition.cpp -o partition -O2 -L/usr/local -lcnpy -lz --std=c++17
```
3. usage:
```bash
./partition model_name device_name
```
eg.
```bash
./partition vit-base-patch16-224 
./partition vit-large-patch16-224 
```
