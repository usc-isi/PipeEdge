"""Generate device networks."""
import argparse
import yaml
import numpy as np

HOST_PREFIX = "mb"

def _rescale(arr, new_min, new_max):
    assert new_max > new_min
    old_min = min(arr)
    old_max = max(arr)
    assert old_max > old_min
    m = (new_max - new_min) / (old_max - old_min)
    b = new_min - m * old_min
    return m * arr + b

def _gen_random_network(size, dev_types, bw_min, bw_max, conn_prob):
    hostnames = [f'{HOST_PREFIX}-{i}' for i in range(size)]

    # Select device types randomly
    devices = { d: [] for d in dev_types }
    for host in hostnames:
        devices[np.random.choice(dev_types)].append(host)

    # Select bandwidths randomly from normal distribution of 1000 data points
    # TODO: This rescaling means there is always a min-bandwidth and max-bandwidth entry
    bw_dist = _rescale(np.random.standard_normal(1000), bw_min, bw_max)
    host_bandwidths = { host: int(np.random.choice(bw_dist)) for host in hostnames }

    # Now populate device neighbors
    device_neighbors = { h: {} for h in hostnames }
    for dev, dev_neighbors in device_neighbors.items():
        for neighbor in device_neighbors:
            if dev == neighbor:
                continue
            # Neighbor reporting doesn't have to be symmetric
            if np.random.random_sample() < conn_prob:
                bandwidth = min(host_bandwidths[dev], host_bandwidths[neighbor])
                dev_neighbors[neighbor] = { 'bw_Mbps': bandwidth }

    return devices, device_neighbors

def _main():
    parser = argparse.ArgumentParser(description="Random network generator",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--size", type=int, default=1, help="Network size")
    parser.add_argument("-p", "--permutations", type=int, default=1, help="Network permutations")
    parser.add_argument("-c", "--conn", type=float, default=1,
                        help="Connection probability in range [0, 1]")
    parser.add_argument("-b", "--bandwidth-min", type=int, default=0, help="Minimum bandwidth")
    parser.add_argument("-B", "--bandwidth-max", type=int, default=1000, help="Maximum bandwidth")
    parser.add_argument("-dt", "--dev-types-file", default='device_types.yml',
                        help="Device types YAML file to use for host generation")
    args = parser.parse_args()

    # device_types = ['MB', 'MB-0.7W', 'MB-1.0W', 'MB-1.5W']
    with open(args.dev_types_file, 'r', encoding='utf-8') as yfile:
        dev_types_yml = yaml.safe_load(yfile)
    device_types = list(dev_types_yml.keys())

    size = args.size
    for perm in range(args.permutations):
        devices, device_neighbors = _gen_random_network(size, device_types, args.bandwidth_min,
                                                        args.bandwidth_max, args.conn)
        # print(yaml.safe_dump(devices, default_flow_style=None, sort_keys=False))
        # print(yaml.safe_dump(device_neighbors, default_flow_style=None, sort_keys=False))
        with open(f'devices-{size}-{perm}.yml', 'w', encoding='utf-8') as yfile:
            yaml.safe_dump(devices, yfile, default_flow_style=None, encoding='utf-8')
        with open(f'device_neighbors_world-{size}-{perm}.yml', 'w', encoding='utf-8') as yfile:
            yaml.safe_dump(device_neighbors, yfile, default_flow_style=None, encoding='utf-8')

if __name__ == "__main__":
    _main()
