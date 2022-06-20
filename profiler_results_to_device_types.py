"""Produce scheduler-compatible device types YAML file from profiling results."""
import argparse
import sys
import yaml
from pipeedge.sched import yaml_files, yaml_types
import model_cfg


def is_dev_type_compatible(device_types, dev_type_name, mem, bwdth):
    """Test existing device type compatibility with current parameters."""
    assert dev_type_name in device_types
    dt_mem = device_types[dev_type_name]['mem_MB']
    dt_bwdth = device_types[dev_type_name]['bw_Mbps']
    if mem is not None and dt_mem != mem:
        print(f"Mismatch for existing device type: mem_MB: {dt_mem} != {mem}")
        return False
    if bwdth is not None and dt_bwdth != bwdth:
        print(f"Mismatch for existing device type: bw_Mbps: {dt_bwdth} != {bwdth}")
        return False
    return True


def is_model_profile_match(model_profile, dtype, batch_size):
    """Test if a model profile configuration matches other parameters."""
    # dtype+batch_size makes for a unique identifier
    return model_profile['dtype'] == dtype and model_profile['batch_size'] == batch_size


def save_device_types_yml(file, dev_type_name, mem, bwdth, model_name, dtype, batch_size, time_s,
                          overwrite_model=False):
    """Save a YAML device types file. Extends file if it exists."""
    # map of device type names to device_type values
    device_types = yaml_files.yaml_device_types_load(file)

    if dev_type_name in device_types:
        # Disallow overwriting properties - existing device type model profiles may depend on them
        if not is_dev_type_compatible(device_types, dev_type_name, mem, bwdth):
            return False
    else:
        if mem is None:
            print("New device type: must specify memory argument")
            return False
        if bwdth is None:
            print("New device type: must specify bandwidth argument")
            return False
        # Create the device type entry (model profile added below)
        device_types[dev_type_name] = yaml_types.yaml_device_type(mem, bwdth, {})

    # Device types support multiple model profiles
    if device_types[dev_type_name]['model_profiles'] is None:
        # new map with key: model name, value: list of profiles
        device_types[dev_type_name]['model_profiles'] = {}
    model_profiles = device_types[dev_type_name]['model_profiles']

    # Add or modify existing profile (if allowed)
    ymp = yaml_types.yaml_model_profile(dtype, batch_size, time_s)
    updated_in_place = False
    if not model_name in model_profiles:
        # new list of model profiles
        model_profiles[model_name] = []
    for idx, model_profile in enumerate(model_profiles[model_name]):
        if is_model_profile_match(model_profile, dtype, batch_size):
            if overwrite_model:
                print(f"Overwriting existing model profile: {file}: {dev_type_name}: {model_name}: {model_profile}")
                model_profiles[model_name][idx] = ymp
                updated_in_place = True
            else:
                print(f"Model profile already exists: {file}: {dev_type_name}: {model_name}: {model_profile}")
                return False
    if not updated_in_place:
        model_profiles[model_name].append(ymp)

    # Create or overwrite entry
    yaml_files.yaml_save(device_types, file)
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Produce scheduler-compatible device types YAML file from profiling results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dev_type", type=str, help="device type name")
    parser.add_argument("-i", "--results-yml", type=str, default="profiler_results.yml",
                        help="profiler results input YAML file")
    parser.add_argument("-o", "--dev-types-yml", type=str, default="device_types.yml",
                        help="device types output YAML file")
    parser.add_argument("-dtm", "--dev-type-mem", type=int,
                        help="memory in MB (required if not already in DEV_TYPES_YML)")
    parser.add_argument("-dtb", "--dev-type-bw", type=int,
                        help="bandwidth in Mbps (required if not already in DEV_TYPES_YML)")
    parser.add_argument("-f", "--overwrite", action='store_true',
                        help="overwrite existing YAML device type model profile entries")
    args = parser.parse_args()

    with open(args.results_yml, 'r', encoding='utf-8') as yfile:
        # map of profiling configurations and results
        results = yaml.safe_load(yfile)

    batch_size = results['batch_size']
    dtype = results['dtype']
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
    time_s = [r['time'] for r in profile_data]
    saved = save_device_types_yml(args.dev_types_yml, args.dev_type, args.dev_type_mem,
                                  args.dev_type_bw, model_name, dtype, batch_size, time_s,
                                  overwrite_model=args.overwrite)
    if not saved:
        sys.exit(1)


if __name__=="__main__":
    main()
