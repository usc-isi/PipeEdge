# EdgePipe

EdgePipe (aka PipeEdge) is an inference framework that pipelines neural network (e.g., transformer) model shards on distributed devices.
It includes an automatic partition scheduler which maps model layers to devices to optimize throughput.


## Prerequisites

System dependencies:

* Python >= 3.7
* Compiler with C++17 support
* CMake >= 3.8 (for C++17 support)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) >= 0.6.0

On MacOS:

```sh
brew install cmake yaml-cpp
```

On Debian (>= buster) or Debian-based Linux (including Ubuntu >= 20.04):

```sh
sudo apt-get install build-essential cmake libyaml-cpp-dev
```

We recommend using a Python virtual environment (`virtualenv`), e.g., on Debian-based Linux:

```sh
sudo apt-get install python3-venv
```

or directly with a system-installed `pip`:

```sh
pip3 install virtualenv
```

Create and activate the virtualenv:

```sh
python3 -m venv .venv
. .venv/bin/activate
```

Install the development package and Python package dependencies with:

```sh
pip install -U pip
pip install -e .
```

Download ViT weight files from [Google Cloud](https://console.cloud.google.com/storage/browser/vit_models;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false), e.g.:

```sh
wget https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz
wget https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz
```

Download BERT and DeiT weight files, too:

```sh
python tools/bert_save_weights.py
python tools/deit_save_weights.py
```


## Usage

For full usage help, run:

```sh
python runtime.py -h
```

To run with default parameters (using ViT-Base) on a single node:

```sh
python runtime.py 0 1
```

To run on multiple nodes, e.g., with 2 stages and even partitioning, on rank 0:

```sh
python runtime.py 0 2 -pt 1,24,25,48
```

and on rank 1:

```sh
python runtime.py 1 2 -pt 1,24,25,48
```

### Partitioning

For example, the ViT-Base model has 12 layers, so the range is [1, 12*4] = [1, 48].

An even partitioning for 2 nodes is:
```
partition = [1,24,25,48]
```

An uneven partitioning for 2 nodes could be:
```
partition = [1,47,48,48]
```

A partitioning for 4 nodes could be:
```
partition = [1,4,5,8,9,20,21,48]
```


## Automatic Partition Scheduling

In summary, the `sched-pipeline` scheduling application uses three input YAML files to map model partitions to devices (hosts).
Automated profiling helps produce two of these files; the third lists available hosts and is straightforward to create for your deployment environment.
For detailed instructions and documentation, see [README_Profiler.md](README_Profiler.md) and [README_Scheduler.md](README_Scheduler.md).

Point `runtime.py` to the YAML files using the options `-sm/--sched-models-file`, `--sdt/--sched-dev-types-file`, and `-sd/--sched-dev-file`.
The runtime passes these through to the previously compiled scheduler application, along with other configurations like the model name and microbatch size.
Then map the hosts specified in the third YAML file to the distributed ranks in your runtime using the `-H/--hosts` option.
Do not specify the `-pt/--partition` option, which is for manually specifying the schedule and takes precedence over automated scheduling.
