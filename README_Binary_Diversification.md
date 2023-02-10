# Binary Diversification Demonstration

We demonstrate applying binary diversification in PipeEdge by using a diversified LZ4 application to compress/decompress tensors communicated between stages.

The instructions below are for an Intel 64-bit CPU system running Ubuntu 18.04 LTS.
See the end of the document for Debian 10 (buster) modifications.


## LZ4

First install prerequisites (e.g., `gcc-multilib` is needed to produce 32-bit binaries):

```sh
sudo apt-get install -y build-essential gcc-multilib git
```

Now build `lz4` as a 32-bit binary without position-independent code:

```sh
git clone https://github.com/lz4/lz4.git
cd lz4
git checkout v1.9.4
make CFLAGS="-m32" LDFLAGS="-no-pie"
```


## Binary Diversification

We use tools from the paper:
W. K. Wong et al., "Deceiving Deep Neural Networks-Based Binary Code Matching with Adversarial Programs," 2022 IEEE International Conference on Software Maintenance and Evolution (ICSME), Limassol, Cyprus, 2022, pp. 117-128, doi: 10.1109/ICSME55016.2022.00019.

Fetch the repository:

```sh
git clone https://github.com/wwkenwong/Deceiving-DNN-based-Binary-Matching.git
```

Then follow the README file's section `Installing requirements for Uroboros`, but skip the package `lib32bz2-1.0` to avoid the error `Unable to locate package lib32bz2-1.0`.


### Create a Python2 Virtual Environment

Create and activate a Python 2.7 virtual environment:

```sh
sudo apt-get install -y python2.7 python-virtualenv
python2 -m virtualenv uroboros
. uroboros/bin/activate
pip install capstone pandas pyelftools termcolor
```

A `requirements.txt` for reference:

```
capstone==4.0.2
numpy==1.16.6
pandas==0.24.2
pkg-resources==0.0.0
pyelftools==0.29
python-dateutil==2.8.2
pytz==2022.7.1
six==1.16.0
termcolor==1.1.0
```


### Create Diversified Binaries

First ensure that the above Python2 virtual environment is activated and your working directory is `Deceiving-DNN-based-Binary-Matching`.

Copy the compiled `lz4` binary to the `Deceiving-DNN-based-Binary-Matching` directory, then run:

```sh
python2 ./uroboros_automate-func-name.py lz4 -i 1 -o temp_bin -d 1 -m original -f $(pwd)/save_bin_folder/tmp --function LZ4_compress
```

The diversified binary is found in the directory `workdir_1`.

The script's `-d` option supports different diversification techniques (snipped from `Deceiving-DNN-based-Binary-Matching/config.py`):

```python
#  0: no operation
#  1: opaque obfuscation on basic blocks
#  2: reorder basic blocks
#  3: branch functions
#  4: split basic blocks
#  5: flatten basic blocks
#  6: reorder functions
#  7: insert garbage code
#  8: transform equivalent instructions
#  9: inline functions
# 10: merge basic blocks
```


## Integrating with PipeEdge

Copy the diversified `lz4` binary to a custom location on each system participating in a distributed PipeEdge pipeline where you want to compress/decompress tensor transfers.
On each system, set the environment variable `LZ4_BINARY` value to the *full path* of the diversified `lz4` binary.
(This also works without a diversified binary, e.g., `sudo apt-get install -y lz4` and set `LZ4_BINARY=/usr/bin/lz4`.)

To enable LZ4 compression on a rank when sending tensors, add the `--lz4-out` argument to `runtime.py`.
To enable LZ4 decompression on a rank when receiving tensors, add the `--lz4-in` argument to `runtime.py`.
Obviously, neighboring send/receive ranks need mirroring compression/decompression configurations.
If either of these arguments are used without configuring `LZ4_BINARY`, the runtime will look for the `lz4` binary on `PATH`.


# Debian 10 Support

The [Distributed Computing Testbed](https://www.dcomptb.net/) platforms run Debian 10 (buster).
Some minor changes are needed for the `Deceiving-DNN-based-Binary-Matching` code.

First, add the following git remote and checkout the `debian-buster` branch:

```sh
git remote add cimes https://github.com/cimes-isi/Deceiving-DNN-based-Binary-Matching.git
git fetch cimes
git checkout debian-buster
```

Then change the Uroboros package dependencies installation command (which could probably be reduced further):

```sh
sudo apt-get -y install gcc gperf bison libtool gcc-multilib python python-dev python-pip gawk build-essential libc6-i386 lib32z1 wget git tar
```

Everything else should be the same.
