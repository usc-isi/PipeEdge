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
  memory: 26.857472
  shape_in:
  - [3, 224, 224]
  shape_out:
  - [197, 768]
  - [197, 768]
  time: 0.36892950534820557
- layer: 2
  memory: 18.890751999999992
  shape_in:
  - [197, 768]
  - [197, 768]
  shape_out:
  - [197, 768]
  - [197, 768]
  time: 0.06506834030151368
- layer: 3
  memory: 26.488832000000002
  shape_in:
  - [197, 768]
  - [197, 768]
  shape_out:
  - [197, 3072]
  - [197, 768]
  time: 0.28465914726257324
- layer: 4
  memory: 25.706496000000016
  shape_in:
  - [197, 3072]
  - [197, 768]
  shape_out:
  - [197, 768]
  - [197, 768]
  time: 0.23929953575134277

# ...

- layer: 48
  memory: 32.669696
  shape_in:
  - [197, 3072]
  - [197, 768]
  shape_out:
  - [1000]
  time: 0.24404664039611818
```


## Scheduler Compatibility

To convert the profiler results YAML file to scheduler-compatible YAML files, use the `profiler_results_to_models.py` and `profiler_results_to_device_types.py` utilities.
By default, these will produce the files `models.yml` and `device_types.yml`, respectively.

For detailed usage instructions, see their help outputs:

```sh
python profiler_results_to_models.py -h
python profiler_results_to_device_types.py -h
```
