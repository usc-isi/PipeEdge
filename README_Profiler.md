# Profiler

The `profiler.py` application profiles model properties (layer input/output data shapes and memory requirements) and device performance (layer execution times).
Profiling produces a YAML output file that is a dump of the configuration and measured results.

The utilities `profiler_results_to_models.py` and `profiler_results_to_device_types.py` can then convert these results into YAML formats that the scheduler requires (see [README_Scheduler.md](README_Scheduler.md)).


## Usage

By default, the profiler will measure all layers in the configured model and produce the file `profiler_results.yml`.

For detailed usage instructions, see the help output:

```sh
python profiler.py -h
```


### Example Results

An example `profiler_results.yml` snippet:

```YAML
batch_size: 8
dtype: torch.float32
layers: 48
model_name: google/vit-base-patch16-224
profile_data:
- layer: 1
  memory: 26.808319999999995
  shape_in:
  - [3, 224, 224]
  shape_out:
  - [197, 768]
  - [197, 768]
  time: 0.37087414264678953
- layer: 2
  memory: 18.927615999999986
  shape_in:
  - [197, 768]
  - [197, 768]
  shape_out:
  - [197, 768]
  time: 0.0655491828918457
- layer: 3
  memory: 26.488831999999988
  shape_in:
  - [197, 768]
  shape_out:
  - [197, 3072]
  - [197, 768]
  time: 0.2822323560714722
- layer: 4
  memory: 25.927679999999995
  shape_in:
  - [197, 3072]
  - [197, 768]
  shape_out:
  - [197, 768]
  time: 0.23820888996124268

# ...

- layer: 48
  memory: 33.062912
  shape_in:
  - [197, 3072]
  - [197, 768]
  shape_out:
  - [1000]
  time: 0.24390978813171388
```


## Scheduler Compatibility

To convert the profiler results YAML file to scheduler-compatible YAML files, use the `profiler_results_to_models.py` and `profiler_results_to_device_types.py` utilities.
By default, these will produce the files `models.yml` and `device_types.yml`, respectively.

For detailed usage instructions, see their help outputs:

```sh
python profiler_results_to_models.py -h
python profiler_results_to_device_types.py -h
```
