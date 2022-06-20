"""Produce scheduler-compatible models YAML file from profiling results."""
import argparse
import sys
import numpy as np
import yaml
from pipeedge.sched import yaml_files, yaml_types
import model_cfg


def save_models_yml(file, model_name, num_layers, parameters_in, parameters_out, mem,
                    overwrite_model=False):
    """Save a YAML models file. Extends file if it exists."""
    # map of model names to yaml_model values
    models = yaml_files.yaml_models_load(file)

    # Check if model already exists
    if model_name in models:
        if overwrite_model:
            print(f"Overwriting existing model: {file}: {model_name}: {models[model_name]}")
        else:
            print(f"Model already exists: {file}: {model_name}: {models[model_name]}")
            return False

    # Create or overwrite entry
    models[model_name] = yaml_types.yaml_model(num_layers, parameters_in, parameters_out, mem)
    yaml_files.yaml_save(models, file)
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Produce scheduler-compatible models YAML file from profiling results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--results-yml", type=str, default="profiler_results.yml",
                        help="profiler results input YAML file")
    parser.add_argument("-o", "--models-yml", type=str, default="models.yml",
                        help="models output YAML file")
    parser.add_argument("-f", "--overwrite", action='store_true',
                        help="overwrite existing YAML model entries")
    args = parser.parse_args()

    with open(args.results_yml, 'r', encoding='utf-8') as yfile:
        # map of profiling configurations and results
        results = yaml.safe_load(yfile)

    layers = results['layers']
    model_name = results['model_name']
    profile_data = results['profile_data']
    if model_name in model_cfg.get_model_names():
        exp_layers = model_cfg.get_model_layers(model_name)
        if layers != exp_layers:
            print(f"Warning: expected and actual layer counts differ: {exp_layers} != {layers}")
    else:
        print(f"Warning: cannot verify layer count for unknown model: {model_name}: {layers}")
    if layers != len(profile_data):
        print(f'Declared layer count does not match profile data count: {layers} != {len(profile_data)}')
        sys.exit(1)
    if len(profile_data) == 0:
        print("Empty profile data!")
        sys.exit(1)
    shape_in = [r['shape_in'] for r in profile_data]
    parameters_in = int(sum([np.prod(s) for s in shape_in[0]]))
    shape_out = [r['shape_out'] for r in profile_data]
    parameters_out = []
    for shp in shape_out:
        parameters_out.append(int(sum([np.prod(s) for s in shp])))
    mem = [r['memory'] for r in profile_data]
    saved = save_models_yml(args.models_yml, model_name, layers, parameters_in, parameters_out, mem,
                            overwrite_model=args.overwrite)
    if not saved:
        sys.exit(1)


if __name__=="__main__":
    main()
