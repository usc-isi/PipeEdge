"""Generate device networks."""
import argparse
import yaml
import numpy as np
from scipy.stats import truncnorm

HOST_PREFIX = "mb"

# https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
def _get_truncated_normal(mean=0, stdev=1, low=0, upp=10):
    return truncnorm((low - mean) / stdev, (upp - mean) / stdev, loc=mean, scale=stdev)

def _gen_random_network(size, dev_types, bw_min, bw_max, bw_mean, bw_stdev, conn_prob):
    hostnames = [f'{HOST_PREFIX}-{i}' for i in range(size)]

    # Select device types randomly
    devices = { d: [] for d in dev_types }
    for host in hostnames:
        devices[np.random.choice(dev_types)].append(host)

    # Select bandwidths randomly from normal distribution of 1000 data points
    bw_dist = _get_truncated_normal(mean=bw_mean, stdev=bw_stdev, low=bw_min, upp=bw_max).rvs(1000)
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
    # The PipeEdge paper evaluated down to 5 Mbps, so use that as the default lower bound.
    parser.add_argument("-b", "--bandwidth-min", type=int, default=5, help="Minimum bandwidth")
    parser.add_argument("-B", "--bandwidth-max", type=int, default=1000, help="Maximum bandwidth")
    parser.add_argument("-bm", "--bandwidth-mean", type=int, default=500, help="Mean bandwidth")
    parser.add_argument("-bs", "--bandwidth-stdev", type=int, default=200, help="Bandwidth stdev")
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
                                                        args.bandwidth_max, args.bandwidth_mean,
                                                        args.bandwidth_stdev, args.conn)
        # print(yaml.safe_dump(devices, default_flow_style=None, sort_keys=False))
        # print(yaml.safe_dump(device_neighbors, default_flow_style=None, sort_keys=False))
        with open(f'devices-{size}-{perm}.yml', 'w', encoding='utf-8') as yfile:
            yaml.safe_dump(devices, yfile, default_flow_style=None, encoding='utf-8')
        with open(f'device_neighbors_world-{size}-{perm}.yml', 'w', encoding='utf-8') as yfile:
            yaml.safe_dump(device_neighbors, yfile, default_flow_style=None, encoding='utf-8')

if __name__ == "__main__":
    _main()
