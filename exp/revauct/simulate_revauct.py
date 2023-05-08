"""Simulate reverse auctions and scheduling."""
import argparse
import logging
import random
import time
import yaml
from pipeedge.sched import revauct

def _sim_bids(models: dict, device_types: dict, devices: dict, device_neighbors_world: dict,
              model: str, ubatch_size: int, dtype: str):
    """Simulate collecting bids."""
    yml_model = models[model]
    bids_in_rank_order = []
    for dev_type, hosts in devices.items():
        yml_dev_type = device_types.get(dev_type)
        if yml_dev_type is None:
            print(f'No matching device type: {dev_type}')
            bids_in_rank_order.extend([(host, ([], [], {})) for host in hosts])
            continue
        yml_dtm_profile = None
        for dtmp in yml_dev_type.get('model_profiles', {}).get(model, []):
            if dtmp['batch_size'] == ubatch_size and dtmp['dtype'] == dtype:
                yml_dtm_profile = dtmp
                break
        if yml_dtm_profile is None:
            print(f'No matching device type model profile: {dev_type}: {model}')
            bids_in_rank_order.extend([(host, ([], [], {})) for host in hosts])
            continue
        bids = revauct.bid_latency(yml_model, yml_dev_type, yml_dtm_profile, ubatch_size,
                                   dtype=dtype)
        shards = [b[0] for b in bids]
        costs = [b[1] for b in bids]
        for host in hosts:
            yml_dev_neighbors = device_neighbors_world[host]
            bids_in_rank_order.append((host, (shards, costs, yml_dev_neighbors)))
    return bids_in_rank_order

def _main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Reverse Auction Simulator",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Device config
    devcfg = parser.add_argument_group('Device configuration')
    devcfg.add_argument("-sm", "--sched-models-file", default='models.yml', type=str,
                        help="models YAML file for scheduler")
    devcfg.add_argument("-sdt", "--sched-dev-types-file", default='device_types.yml', type=str,
                        help="device types YAML file for scheduler")
    devcfg.add_argument("-sd", "--sched-devs", default='devices.yml', type=str,
                        help="devices YAML file for scheduler")
    devcfg.add_argument("-sdnw", "--sched-dev-neighbors-world",
                        default='device_neighbors_world.yml', type=str,
                        help="device neighbors world YAML file for scheduler")
    devcfg.add_argument("-sds", "--sched-dev-src", default='mb-0', type=str, help="source device")
    devcfg.add_argument("-sdd", "--sched-dev-dest", default='mb-0', type=str,
                        help="destination device")
    # Model options
    modcfg = parser.add_argument_group('Model configuration')
    modcfg.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        # choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    modcfg.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")
    # Scheduler options
    schcfg = parser.add_argument_group('Additional scheduler options')
    schcfg.add_argument("--filter-bids-chunk", type=int, default=1,
                        help="filter bids by chunk size")
    schcfg.add_argument("--filter-bids-largest", action='store_true',
                        help="filter bids by the largest chunks")
    schcfg.add_argument("-s", "--scheduler", default="latency_ordered",
                        choices=["latency_ordered", "throughput_ordered", "greedy_host_count"],
                        help="the scheduler to use")
    schcfg.add_argument("-d", "--dev-count", type=int, default=None,
                        help="the number of devices to consider")
    schcfg.add_argument("--no-strict-order", action='store_true',
                        help="disable strict ordering (total order still respected)")
    schcfg.add_argument("--strict-first", action='store_true',
                        help="require first device to be used")
    schcfg.add_argument("--strict-last", action='store_true',
                        help="require last device to be used")
    args = parser.parse_args()

    model = args.model_name
    ubatch_size = args.ubatch_size
    dtype = 'torch.float32'

    with open(args.sched_models_file, 'r', encoding='utf-8') as yfile:
        models = yaml.safe_load(yfile)
    with open(args.sched_dev_types_file, 'r', encoding='utf-8') as yfile:
        device_types = yaml.safe_load(yfile)
    with open(args.sched_devs, 'r', encoding='utf-8') as yfile:
        devices = yaml.safe_load(yfile)
    with open(args.sched_dev_neighbors_world, 'r', encoding='utf-8') as yfile:
        device_neighbors_world = yaml.safe_load(yfile)

    # Simulate collecting bids
    t_start = time.time()
    bids_in_rank_order = _sim_bids(models, device_types, devices, device_neighbors_world, model,
                                   ubatch_size, dtype)
    t_end = time.time()
    print(f"Reverse auction simulation time (sec): {t_end - t_start}")
    bid_data_by_host = \
        { b[0]: ({ tuple(lyrs): cost for lyrs, cost in zip(b[1][0], b[1][1]) }, b[1][2])
          for b in bids_in_rank_order }

    yml_model = models[model]
    host_src = args.sched_dev_src
    host_dest = args.sched_dev_dest
    scheduler = args.scheduler

    # Produce a schedule from bids

    if args.filter_bids_chunk > 1:
        bid_data_by_host = {
            h: (revauct.filter_bids_chunk(yml_model, b[0], chunk=args.filter_bids_chunk), b[1])
            for h, b in bid_data_by_host.items()
        }

    if args.filter_bids_largest:
        bid_data_by_host = { h: (revauct.filter_bids_largest(b[0]), b[1])
                             for h, b in bid_data_by_host.items() }

    dev_order = []
    for devs in devices.values():
        dev_order.extend(devs)
    random.shuffle(dev_order)
    dev_order = dev_order[:args.dev_count]
    strict_order = not args.no_strict_order
    strict_first = args.strict_first
    strict_last = args.strict_last
    print(f"Device order: {dev_order}")

    schedule = []
    pred = -1
    t_start = time.time()
    if scheduler == 'latency_ordered':
        schedule, pred = revauct.sched_optimal_latency_dev_order(
            yml_model, ubatch_size, dtype, bid_data_by_host, host_src, host_dest, dev_order,
            strict_order=strict_order, strict_first=strict_first, strict_last=strict_last)
        print(f"Latency prediction (sec): {pred}")
    elif scheduler == 'throughput_ordered':
        schedule, pred = revauct.sched_optimal_throughput_dev_order(
            yml_model, ubatch_size, dtype, bid_data_by_host, host_src, host_dest, dev_order,
            strict_order=strict_order, strict_first=strict_first, strict_last=strict_last)
        print(f"Throughput prediction (items/sec): {pred}")
    elif scheduler == "greedy_host_count":
        schedule = revauct.sched_greedy_host_count(yml_model, ubatch_size, dtype, bid_data_by_host,
                                                   host_src, host_dest)
    else:
        assert False
    t_end = time.time()
    print(f"Scheduler function runtime (sec): {t_end - t_start}")
    print(f"Schedule stages: {len(schedule)}")
    print("Schedule:")
    print(yaml.safe_dump(schedule, default_flow_style=None, sort_keys=False))

if __name__ == "__main__":
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    _main()
