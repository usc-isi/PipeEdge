"""Reverse auction distributed application."""
import argparse
import logging
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
    modcfg = parser.add_argument_group('Device configuration')
    modcfg.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    modcfg.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")
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
            # Collect bids
            logger.debug("Broadcasting reverse auction request")
            futs = []
            for rank in range(args.worldsize):
                fut = rpc.rpc_async(rank, revauct_bid_latency,
                                    args=(args.model_name, args.ubatch_size))
                futs.append(fut)
            bids_in_rank_order = torch.futures.wait_all(futs)
            logger.debug("Received bids in rank order: %s", bids_in_rank_order)
            bid_data_by_host = \
                { b[0]: ({ tuple(lyrs): cost for lyrs, cost in zip(b[1][0], b[1][1]) }, b[1][2])
                  for b in bids_in_rank_order }
            logger.debug("Received bids by host: %s", bid_data_by_host)
            # Schedule
            data_host = args.host if args.data_host is None else args.data_host
            schedule = revauct.sched_min_latencies(yml_model, args.ubatch_size, 'torch.float32',
                                                   bid_data_by_host, data_host, data_host)
            # PipeEdge scheduling/partitioning starts layer count at 1, so shift layer IDs
            sched_compat = [{ host: [l + 1 for l in layers] for host, layers in part.items()}
                            for part in schedule]
            print(yaml.safe_dump(sched_compat, default_flow_style=None, sort_keys=False))


if __name__=="__main__":
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    main()
