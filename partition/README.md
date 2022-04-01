# Pipeline Scheduler

The `sched-pipeline` application is written in C++, so it must be compiled.


## Prerequisites:

* Compiler with C++17 support
* CMake >= 3.8 (for C++17 support)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) >= 0.6.0

On MacOS:

```sh
brew install cmake yaml-cpp
```

On Debian or Debian-based Linux (including Ubuntu):

```sh
sudo apt-get install build-essential cmake libyaml-cpp-dev
```


## Building

From this directory:

```sh
mkdir build
cd build
cmake ..
cmake --build .
```


# Deprecated partition scheduler

To build the deprecated `partition` scheduler:

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
