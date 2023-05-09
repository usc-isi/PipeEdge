"""Generate device networks."""
import argparse
import yaml
import numpy as np
from scipy.stats import truncnorm

HOST_PREFIX = "mb"

# https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
def _get_truncated_normal(mean=0, stdev=1, low=0, upp=10):
    return truncnorm((low - mean) / stdev, (upp - mean) / stdev, loc=mean, scale=stdev)

def _gen_random_devices(size: int, dev_types: list):
    # First create hosts
    hostnames = [f'{HOST_PREFIX}-{i}' for i in range(size)]
    # Now select device types randomly
    devices = { d: [] for d in dev_types }
    for host in hostnames:
        devices[np.random.choice(dev_types)].append(host)
    return devices

def _gen_random_network(hostnames: list, bw_dist: list, conn_prob: float):
    assert 0 <= conn_prob <= 1
    # Select bandwidths randomly from a distribution
    bandwidths = np.random.choice(bw_dist, size=len(hostnames))
    # Now populate device neighbors symmetrically
    device_neighbors = { h: {} for h in hostnames }
    for dev_idx, (dev, dev_neighbors) in enumerate(device_neighbors.items()):
        for neighbor_idx, (neighbor, neighbor_neighbors) in enumerate(device_neighbors.items()):
            if dev_idx < neighbor_idx and np.random.random_sample() < conn_prob:
                bandwidth = int(min(bandwidths[dev_idx], bandwidths[neighbor_idx]))
                dev_neighbors[neighbor] = { 'bw_Mbps': bandwidth }
                neighbor_neighbors[dev] = { 'bw_Mbps': bandwidth }
    return device_neighbors

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
        devices = _gen_random_devices(size, device_types)
        bw_dist = _get_truncated_normal(mean=args.bandwidth_mean, stdev=args.bandwidth_stdev,
                                        low=args.bandwidth_min, upp=args.bandwidth_max).rvs(1000)
        hostnames = []
        for hosts in devices.values():
            hostnames.extend(hosts)
        device_neighbors = _gen_random_network(hostnames, bw_dist, args.conn)
        # print(yaml.safe_dump(devices, default_flow_style=None, sort_keys=False))
        # print(yaml.safe_dump(device_neighbors, default_flow_style=None, sort_keys=False))
        with open(f'devices-{size}-{perm}.yml', 'w', encoding='utf-8') as yfile:
            yaml.safe_dump(devices, yfile, default_flow_style=None, encoding='utf-8')
        with open(f'device_neighbors_world-{size}-{perm}.yml', 'w', encoding='utf-8') as yfile:
            yaml.safe_dump(device_neighbors, yfile, default_flow_style=None, encoding='utf-8')

if __name__ == "__main__":
    _main()
