"""Reverse auction distributed application."""
import argparse
import logging
import random
import time
from typing import List, Mapping, Optional, Tuple, Type
import yaml
import torch
from torch.distributed import rpc
from pipeedge.comm.rpc import DistRpcContext
from pipeedge.sched import revauct, yaml_files
import model_cfg
import runtime

Bid: Type = Tuple[List[Tuple[int, int]], List[float], Mapping[str, dict]]
"""A bid type: ([shard layer pairs], [shard costs], {neighbor: device_neighbors_type})."""

Schedule: Type = List[Mapping[str, List[int]]]
"""While not the cleanest type, this is compatible with the YAML one used in scheduler.py."""


logger = logging.getLogger(__name__)


# Local config isn't synchronized - do not modify after the distributed context is initialized.
_DEVICE_CFG = {}


def _find_profiles(model: str, ubatch_size: int, dtype: str) -> \
    Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """Find model, device type, and device type model-specific profiles."""
    # yml_model is a dict with keys:
    #  'layers' -> int
    #  'mem_MB' -> List[float] (length = 'layers')
    #  'parameters_in' -> int
    #  'parameters_out' -> List[int] (length = 'layers')
    try:
        yml_model = _DEVICE_CFG['yml_models'][model]
    except KeyError:
        logger.debug("No matching model profile for model=%s", model)
        return (None, None, None)
    # yml_dev_type is a dict with keys:
    #  'bw_Mbps' -> int
    #  'mem_MB' -> List[float] (length = yml_model['layers'])
    #  'model_profiles' -> Mapping[str, dev_type_model_profile]
    dev_type = _DEVICE_CFG['dev_type']
    try:
        yml_dev_type = _DEVICE_CFG['yml_dev_types'][dev_type]
    except KeyError:
        logger.debug("No matching device type profile for dev_type=%s", dev_type)
        return (yml_model, None, None)
    # yml_dtm_profile is a dict with keys (or None if no match found):
    #  'batch_size' -> int
    #  'dtype' -> torch.dtype (name as string)
    #  'time_s' -> List[float] (length = yml_model['layers'])
    yml_dtm_profile = None
    for dtmp in yml_dev_type.get('model_profiles', {}).get(model, []):
        if dtmp['batch_size'] == ubatch_size and dtmp['dtype'] == dtype:
            yml_dtm_profile = dtmp
            break
    if yml_dtm_profile is None:
        logger.debug("No matching device type model profile for model=%s, dev_type=%s, "
                     "ubatch_size=%d, dtype=%s", model, dev_type, ubatch_size, dtype)
    return (yml_model, yml_dev_type, yml_dtm_profile)


# This function is called over RPC, so the local _DEVICE_CFG must be populated first.
def revauct_bid_latency(model: str, ubatch_size: int, dtype: str='torch.float32') -> \
    Tuple[str, Bid]:
    """Respond to a reverse auction request with hostname and bid."""
    t_start = time.time()
    logger.debug("Received reverse auction request: model=%s, ubatch_size=%d", model, ubatch_size)
    yml_model, yml_dev_type, yml_dtm_profile = _find_profiles(model, ubatch_size, dtype)
    shards: List[Tuple[int, int]] = []
    costs: List[float] = []
    if yml_model is not None and yml_dev_type is not None and yml_dtm_profile is not None:
        bids = revauct.bid_latency(yml_model, yml_dev_type, yml_dtm_profile, ubatch_size,
                                   dtype=dtype)
        for bid in bids:
            shards.append(bid[0])
            costs.append(bid[1])
    host = _DEVICE_CFG['host']
    # yml_dev_neighbors is a dict with hostnames as keys and yaml_device_neighbors_type values
    # (yaml_device_neighbors_type a dict, currently with only a single key: 'bw_Mbps').
    yml_dev_neighbors = _DEVICE_CFG['yml_dev_neighbors_world'].get(host, {})
    logger.debug("Reverse auction bid time (ms): %f", 1000 * (time.time() - t_start))
    return (host, (shards, costs, yml_dev_neighbors))


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Pipeline Reverse Auction Scheduler",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument("rank", type=int, help="the rank for the current node")
    parser.add_argument("worldsize", type=int, help="the world size (the number of nodes)")
    # Network configurations
    netcfg = parser.add_argument_group('Network configuration')
    netcfg.add_argument("-s", "--socket-ifname", type=str, default="lo0",
                        help="socket interface name, use [ifconfig | ipaddress] to check")
    netcfg.add_argument("--addr", type=str, default="127.0.0.1",
                        help="ip address for the master node")
    netcfg.add_argument("--port", type=int, default=29500,
                        help="communication port for the master node")
    # Device config
    devcfg = parser.add_argument_group('Device configuration')
    devcfg.add_argument("-sm", "--sched-models-file", default='models.yml', type=str,
                        help="models YAML file for scheduler")
    devcfg.add_argument("-sdt", "--sched-dev-types-file", default='device_types.yml', type=str,
                        help="device types YAML file for scheduler")
    devcfg.add_argument("-sdnw", "--sched-dev-neighbors-world", default='device_neighbors_world.yml',
                        type=str,
                        help="device neighbors world YAML file for scheduler")
    devcfg.add_argument("-dt", "--dev-type", default=None, type=str, required=True,
                        help="this rank's device type (requires matching entry in YAML file)")
    devcfg.add_argument("-H", "--host", type=str,
                        help="this device's host name (requires matching entry in YAML file)")
    devcfg.add_argument("-D", "--data-host", type=str, default=None,
                        help="host where inputs are loaded and outputs are processed; "
                             "default: rank 0's host name")
    # Model options
    modcfg = parser.add_argument_group('Model configuration')
    modcfg.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    modcfg.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")
    # Scheduler options
    schcfg = parser.add_argument_group('Additional scheduler options')
    schcfg.add_argument("--filter-bids-chunk", type=int, default=1,
                        help="filter bids by chunk size")
    schcfg.add_argument("--filter-bids-largest", action='store_true',
                        help="filter bids by the largest chunks")
    schcfg.add_argument("-sch", "--scheduler", default="latency_ordered",
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

    # Populate device config - may contain more than just this device type and host - filter later
    _DEVICE_CFG['yml_models'] = yaml_files.yaml_models_load(args.sched_models_file)
    _DEVICE_CFG['yml_dev_types'] = yaml_files.yaml_device_types_load(args.sched_dev_types_file)
    _DEVICE_CFG['yml_dev_neighbors_world'] = \
        yaml_files.yaml_device_neighbors_world_load(args.sched_dev_neighbors_world)
    _DEVICE_CFG['dev_type'] = args.dev_type
    _DEVICE_CFG['host'] = args.host

    # Setup network
    runtime.init_env(None, args.addr, args.port, args.socket_ifname)

    # Run distributed reverse auction
    with DistRpcContext((f"worker{args.rank}",),
                        { 'world_size': args.worldsize,
                          'rank': args.rank }):
        if args.rank == 0:
            # We're the auctioneer (we're also a bidder, unless we skip rank=0 in the broadcast)
            # Make sure we have the profile info needed to schedule
            yml_model = _DEVICE_CFG['yml_models'][args.model_name]
            ubatch_size = args.ubatch_size
            dtype = 'torch.float32'
            # Collect bids
            logger.debug("Broadcasting reverse auction request")
            futs = []
            t_start = time.time()
            for rank in range(args.worldsize):
                fut = rpc.rpc_async(rank, revauct_bid_latency, args=(args.model_name, ubatch_size))
                futs.append(fut)
            bids_in_rank_order = torch.futures.wait_all(futs)
            logger.debug("Reverse auction total time (ms): %f", 1000 * (time.time() - t_start))
            logger.debug("Received bids in rank order: %s", bids_in_rank_order)
            bid_data_by_host = \
                { b[0]: ({ tuple(lyrs): cost for lyrs, cost in zip(b[1][0], b[1][1]) }, b[1][2])
                  for b in bids_in_rank_order }
            logger.debug("Received bids by host: %s", bid_data_by_host)

            # Maybe filter bids.
            if args.filter_bids_chunk > 1:
                bid_data_by_host = {
                    h: (revauct.filter_bids_chunk(yml_model, b[0], chunk=args.filter_bids_chunk),
                        b[1])
                    for h, b in bid_data_by_host.items()
                }

            if args.filter_bids_largest:
                bid_data_by_host = { h: (revauct.filter_bids_largest(b[0]), b[1])
                                     for h, b in bid_data_by_host.items() }

            # Shuffle device ordering and limit to specified device count.
            data_host = args.host if args.data_host is None else args.data_host
            dev_order = list(bid_data_by_host.keys())
            random.shuffle(dev_order)
            dev_order = dev_order[:args.dev_count]
            # Enforce that data host is first, if it's included at all.
            for idx, _ in enumerate(dev_order):
                if dev_order[idx] == data_host:
                    dev_order[0], dev_order[idx] = dev_order[idx], dev_order[0]
            logger.info("Device order: %s", dev_order)

            strict_order = not args.no_strict_order
            strict_first = args.strict_first
            strict_last = args.strict_last

            schedule = []
            pred = -1
            t_start = time.time()
            if args.scheduler == 'latency_ordered':
                schedule, pred = revauct.sched_optimal_latency_dev_order(
                    yml_model, ubatch_size, dtype, bid_data_by_host, data_host, data_host,
                    dev_order,
                    strict_order=strict_order, strict_first=strict_first, strict_last=strict_last)
                logger.info("Latency prediction (sec): %s", pred)
            elif args.scheduler == 'throughput_ordered':
                schedule, pred = revauct.sched_optimal_throughput_dev_order(
                    yml_model, ubatch_size, dtype, bid_data_by_host, data_host, data_host,
                    dev_order,
                    strict_order=strict_order, strict_first=strict_first, strict_last=strict_last)
                logger.info("Throughput prediction (items/sec): %s", pred)
            elif args.scheduler == "greedy_host_count":
                schedule = revauct.sched_greedy_host_count(yml_model, ubatch_size, dtype,
                                                           bid_data_by_host, data_host, data_host)
            else:
                assert False
            t_end = time.time()
            logger.info("Scheduler function runtime (sec): %s", t_end - t_start)
            logger.info("Schedule stages: %d", len(schedule))

            # PipeEdge scheduling/partitioning starts layer count at 1, so shift layer IDs
            sched_compat = [{ host: [l + 1 for l in layers] for host, layers in part.items()}
                            for part in schedule]

            # Finally, print the schedule.
            logger.info("Schedule:")
            logger.info(yaml.safe_dump(sched_compat, default_flow_style=None, sort_keys=False))


if __name__=="__main__":
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    main()
