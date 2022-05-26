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
  memory: 26.509311999999994
  shape_in:
  - [3, 224, 224]
  shape_out:
  - [197, 768]
  - [197, 768]
  time: 0.3705007314682007
- layer: 2
  memory: 19.140608
  shape_in:
  - [197, 768]
  - [197, 768]
  shape_out:
  - [197, 768]
  - [197, 768]
  time: 0.06647565364837646
- layer: 3
  memory: 26.67724799999999
  shape_in:
  - [197, 768]
  - [197, 768]
  shape_out:
  - [197, 3072]
  - [197, 768]
  time: 0.2820276737213135
- layer: 4
  memory: 26.07923199999999
  shape_in:
  - [197, 3072]
  - [197, 768]
  shape_out:
  - [197, 768]
  time: 0.2384713411331177

# ...

- layer: 48
  memory: 33.312768000000005
  shape_in:
  - [197, 3072]
  - [197, 768]
  shape_out:
  - [1000]
  time: 0.24462425708770752
```


## Scheduler Compatibility

To convert the profiler results YAML file to scheduler-compatible YAML files, use the `profiler_results_to_models.py` and `profiler_results_to_device_types.py` utilities.
By default, these will produce the files `models.yml` and `device_types.yml`, respectively.

For detailed usage instructions, see their help outputs:

```sh
python profiler_results_to_models.py -h
python profiler_results_to_device_types.py -h
```
