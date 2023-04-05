"""Reverse auction scheduling."""
import logging
from typing import List, Mapping, Tuple, Type
import networkx as nx
import numpy as np
from . import communication_time_bw, computation_time, mem_bytes, ubatch_bytes


logger = logging.getLogger(__name__)


ShardBid: Type = Tuple[Tuple[int, int], float]
"""A shard bid has the form: `((start_layer, end_layer), cost)`."""

DeviceBidData: Type = Tuple[Mapping[Tuple[int, int], float], Mapping[str, dict]]
"""A device's bids: (1) shard layer pairs mapped to their costs, (2) communication properties."""

def bid_latency(yml_model: dict, yml_dev_type: dict, yml_dtm_profile: dict, ubatch_size: int,
                dtype: str='torch.float32') -> List[ShardBid]:
    """Bid for shards using latency as the cost metric."""
    bids = []
    dt_mem_bytes = yml_dev_type['mem_MB'] * 1024 * 1024
    for layer_l in range(yml_model['layers']):
        for layer_r in range(layer_l, yml_model['layers']):
            bytes_req = mem_bytes(yml_model, layer_l, layer_r, dtype, ubatch_size)
            if dt_mem_bytes > bytes_req:
                cost = computation_time(yml_dtm_profile, layer_l, layer_r)
                bids.append(((layer_l, layer_r), cost))
    return bids

def filter_bids_chunk(yml_model: dict, bids: Mapping[Tuple[int, int], float], chunk: int=4) -> \
    Mapping[Tuple[int, int], float]:
    """Filter bids for shards that are multiples of a chunk size."""
    model_layers = yml_model['layers']
    bids_filt = {}
    for shard, cost in bids.items():
        # Tail shard may be smaller than chunk if chunk size doesn't evenly divide layer count
        if shard[0] % chunk == 0 and (shard[1] + 1 >= model_layers or (shard[1] + 1) % chunk == 0):
            bids_filt[shard] = cost
    return bids_filt

def filter_bids_largest(bids: Mapping[Tuple[int, int], float]) -> Mapping[Tuple[int, int], float]:
    """Filter bids by the largest shards for each start layer."""
    shards_largest = {} # map starting layer to (shard, cost)
    for shard, cost in bids.items():
        if shard[0] not in shards_largest:
            shards_largest[shard[0]] = (shard, cost)
        # Only keep the largest shard for this starting layer
        if shard[1] > shards_largest[shard[0]][0][0]:
            shards_largest[shard[0]] = (shard, cost)
    return { v[0]: v[1] for v in shards_largest.values() }

def sched_greedy_host_count(yml_model: dict, _ubatch_size: int, _dtype: str,
                            bids: Mapping[str, DeviceBidData], host_src: str, host_dest: str) -> \
    List[Mapping[str, List[int]]]:
    """
    Schedule for minimum device count: assumes full connectivity, ignores bandwidths.

    First schedules the largest supported shard starting at the first layer on the source host,
    then the largest supported shard with the last layer on the destination host.
    Finally, schedules the remaining layers with the largest supported shards on other hosts (if
    multiple hosts support a largest shard, choose the host with the lowest bid cost).
    It's possible that a complete pipeline won't be found, even if one is feasible.

    This approach tends to prioritize latency by reducing communication.
    The heuristic should work well if larger devices are also faster devices.
    """
    # Build a simpler LUT than for bid shards.
    max_lay_lut = { host: {} for host in bids } # { host: {start_layer: (max_end_layer, cost)}}
    for host, hbids in bids.items():
        for shard, cost in hbids[0].items():
            if max_lay_lut[host].get(shard[0], (-1, -1))[0] < shard[1]:
                max_lay_lut[host][shard[0]] = (shard[1], cost)
    # Now try to build a schedule.
    sched = []
    sched_ins_off = 0
    lay_start = 0
    lay_end = yml_model['layers'] - 1
    dev_scheduled = set()
    # Greedily schedule the first layer(s) on the src device.
    if host_src in max_lay_lut and lay_start in max_lay_lut[host_src]:
        lay_max = max_lay_lut[host_src][lay_start][0]
        sched.append({ host_src: [lay_start, lay_max] })
        dev_scheduled.add(host_src)
        lay_start = lay_max + 1
    # Greedily schedule the last layer(s) on the dest device.
    # The src device is not allowed to be given the last layers (unless it has the whole model).
    if host_dest in max_lay_lut and host_src != host_dest:
        lay_min = lay_end + 1
        for lay_s, (lay_e, _) in max_lay_lut[host_dest].items():
            if lay_e == lay_end:
                lay_min = min(lay_s, lay_min)
        if lay_min <= lay_end:
            sched.append({ host_dest: [lay_min, lay_end] })
            dev_scheduled.add(host_dest)
            lay_end = lay_min - 1
            sched_ins_off = 1
    # Greedily schedule the remaining layers.
    while lay_start <= lay_end:
        best = (None, -1, -1) # (device, layer_end, cost)
        # When multiple devices have the same max shard size, pick the one with the best cost.
        for dev in max_lay_lut:
            if dev not in dev_scheduled and lay_start in max_lay_lut[dev]:
                cand = max_lay_lut[dev][lay_start]
                if cand[0] > best[1] or (cand[0] == best[1] and cand[1] < best[2]):
                    best = (dev, cand[0], cand[1])
        if best[0] is None:
            # Game over, man, game over!
            return []
        dev, lay_max, _ = best
        sched.insert(len(sched) - sched_ins_off, { dev: [lay_start, lay_max] })
        dev_scheduled.add(dev)
        lay_start = lay_max + 1
    if host_dest not in sched[-1]:
        sched.append({ host_dest: [] })
    return sched


class _Device:
    """Scheduler device representation."""

    def __init__(self, name, devno):
        self.name = name
        self.devno = devno
        self.neighbors = {}
        self.bids = {}

    def __str__(self):
        return "Device:\n" + \
               f"  name: {self.name}\n  devno: {self.devno}\n" + \
               f"  neighbors: {[n.name for n in self.neighbors]}\n" + \
               f"  bids: {self.bids}"


class _Model:
    """Scheduler model representation."""

    def __init__(self, yml_model: dict):
        self.layers: int = yml_model['layers']
        self.mem_MB: List[float] = yml_model['mem_MB']
        self.parameters_in: int = yml_model['parameters_in']
        self.parameters_out: List[int] = yml_model['parameters_out']

    def __str__(self):
        return "Model\n" + \
               f"  layers: {self.layers}\n  mem_MB: {self.mem_MB}\n" + \
               f"  parameters_in: {self.parameters_in}\n  parameters_out: {self.parameters_out}"


def _bids_to_devices(bids: Mapping[str, DeviceBidData], host_src: str, host_dest: str) -> \
    List[_Device]:
    # First create the devices instances so we can reference them as neighbors
    dev_by_name = { host: _Device(host, devno) for devno, host in enumerate(bids) }
    dev_list = []
    for host, host_bid in bids.items():
        dev = dev_by_name[host]
        dev.bids = host_bid[0]
        # need Device references, not just neighbor host names
        # dict w/ key=hostname, value=dict (with one key: 'bw_Mbps'), e.g.:
        # {'mb-0': {'bw_Mbps': 1000}}
        bid_neighbors = host_bid[1]
        # Ignore neighboring devices that didn't submit their own bids
        dev_neighbors = { dev_by_name[neighbor_name]: bid_neighbors[neighbor_name]
                          for neighbor_name in bid_neighbors
                          if neighbor_name in dev_by_name }
        dev.neighbors = dev_neighbors
        dev_list.append(dev)
    if host_src not in dev_by_name:
        raise ValueError(f'Source host not in bids: {host_src}')
    if host_dest not in dev_by_name:
        raise ValueError(f'Destination host not in bids: {host_dest}')
    return (dev_list, dev_by_name[host_src], dev_by_name[host_dest])

def _devs_to_adj_matrix(dev_list: List[_Device]) -> np.ndarray:
    adj_matrix = np.zeros((len(dev_list), len(dev_list)))
    for dev in dev_list:
        for key, item in dev.neighbors.items():
            if dev.devno != key.devno:
                adj_matrix[dev.devno][key.devno] = item['bw_Mbps']
    return adj_matrix

# Return type isn't the cleanest, but is compatible with the YAML one used in scheduler.py
def sched_max_throughput(yml_model: dict, ubatch_size: int, dtype: str,
                         bids: Mapping[str, DeviceBidData], host_src: str, host_dest: str) -> \
    List[Mapping[str, List[int]]]:
    """Schedule for maximum throughput, accounting for computation and communication overlap."""
    model = _Model(yml_model)
    logger.debug("Created model: %s", model)

    # Populate device class instances with their bids
    dev_list, dev_src, dev_dest = _bids_to_devices(bids, host_src, host_dest)

    # Create adjacency matrix of bandwidths between devices.
    adj_matrix = _devs_to_adj_matrix(dev_list)
    logger.debug("Adjacency matrix: %s", adj_matrix)

    # Now do the scheduling
    sched = _schedule_best_time_overlap(model, ubatch_size, dtype, dev_src, dev_dest, dev_list,
                                        adj_matrix)
    if len(sched[0]) == 0:
        logger.debug("No possible paths.")
    else:
        logger.debug("The pipeline runs in: %f seconds.", sched[2])
        for stage, (dev, layers) in enumerate(zip(sched[0], sched[1])):
            logger.debug("Stage: %d\n  Device: %s\n  Layers: %s", stage, dev.name, layers)
    return [{ dev.name: list(layers) } for dev, layers in zip(sched[0], sched[1])]


NodeID: Type = Tuple[Tuple[Tuple[int, int], ...], Tuple[_Device, ...]]
"""Two tuples: (1) Layer ranges up to and including this node, (2) Devices assigned to (1)."""

def _shard_lut(dev: _Device) -> Mapping[int, List[Tuple[int, int]]]:
    """Create a lookup table of supported shards, keyed by their start layer."""
    shards = {}
    for bid in dev.bids:
        if bid[0] not in shards:
            shards[bid[0]] = []
        shards[bid[0]].append(bid)
    return shards

def _schedule_best_time_overlap(model: _Model, ubatch_size: int, dtype: str, dev_src: _Device,
                                dev_dest: _Device, devices: List[_Device], adj_matrix: np.ndarray) \
    -> Tuple[Tuple[_Device, ...], Tuple[Tuple[int, int], ...], float]:
    """Schedule for minimum latency (cost), accounting for computation and communication overlap."""
    # Create the shard lookup table for each device.
    dev_shards = { d: _shard_lut(d) for d in devices }
    # Search for feasible schedules, keeping track of the best.
    schedule = ((), (), float('inf'))
    for dev, shards in dev_shards.items():
        for shard in shards.get(0, []):
            graph = _build_graph(model, ubatch_size, dtype, dev_src, dev_dest, dev_shards,
                                 adj_matrix, dev, shard)
            for node_id_tail in graph.graph['tails']:
                path, cost = _shortest_path_with_cost(graph, graph.graph['root'], node_id_tail)
                if cost < schedule[2]:
                    # The final node ID contains the complete device and layer schedule.
                    schedule = (path[-1][1], path[-1][0], cost)
    return schedule

def _build_graph(model: _Model, ubatch_size: int, dtype: str, dev_src: _Device, dev_dest: _Device,
                 dev_shards: Mapping[_Device, Mapping[int, List[Tuple[int, int]]]],
                 adj_matrix: np.ndarray, dev: _Device, shard: Tuple[int, int]) -> nx.DiGraph:
    """Build a weighted directed graph, starting with the given device and shard."""
    graph = nx.DiGraph(tails=[])
    devices_seen = []
    if dev == dev_src:
        # This device is also the input data source device.
        node_id_src = None
    elif adj_matrix[dev_src.devno][dev.devno] > 0:
        # This device can't receive from the input data source device, so we're done - oh well.
        return graph
    else:
        node_id_src = (((),), (dev_src,))
        graph.add_node(node_id_src, weight=0)
        graph.graph['root'] = node_id_src
        # PipeEdge (currently) requires that if dev_src is assigned a shard, it must be first.
        # Since it's not first, claim it's been seen, o/w it might be assigned a shard somewhere
        # else in the pipeline.
        devices_seen.append(dev_src)
    _build_tree(model, ubatch_size, dtype, dev_src, dev_dest, dev_shards, adj_matrix, graph,
                devices_seen, node_id_src, dev, shard)
    return graph

def _build_tree(model: _Model, ubatch_size: int, dtype: str, dev_src: _Device, dev_dest: _Device,
                dev_shards: Mapping[_Device, Mapping[int, List[Tuple[int, int]]]],
                adj_matrix: np.ndarray, graph: nx.DiGraph, devices_seen: List[_Device],
                node_id_prev: NodeID, device: _Device, shard: Tuple[int, int]) -> None:
    """Recursively build the device/shard graph."""
    min_layer = shard[0]
    max_layer = shard[1]
    comp_time = device.bids[(min_layer, max_layer)]
    if node_id_prev is None:
        node_id = ((shard,), (device,))
    else:
        node_id = (node_id_prev[0] + (shard,), node_id_prev[1] + (device,))
    assert node_id not in graph.nodes
    graph.add_node(node_id, weight=comp_time)
    if node_id_prev is None:
        # We're also the data source
        graph.graph['root'] = node_id
    else:
        dev_prev = node_id_prev[1][-1]
        params = model.parameters_in if min_layer == 0 else model.parameters_out[min_layer - 1]
        comm_bytes = ubatch_bytes(params, ubatch_size, dtype=dtype)
        assert adj_matrix[dev_prev.devno][device.devno] > 0
        comm_time = communication_time_bw(adj_matrix[dev_prev.devno][device.devno], comm_bytes)
        graph.add_edge(node_id_prev, node_id, weight=comm_time)

    if max_layer < model.layers - 1:
        # Recursion: look for unseen devices that can handle the next layer(s).
        # If there aren't any or they aren't reachable, we can't form a complete pipeline - oh well.
        devices_seen.append(device)
        for dev, shards in dev_shards.items():
            # As we scale, it's faster to eliminate paths by checking device availability first.
            if adj_matrix[device.devno][dev.devno] <= 0 or dev in devices_seen:
                continue
            for next_shard in shards.get(max_layer + 1, []):
                _build_tree(model, ubatch_size, dtype, dev_src, dev_dest, dev_shards, adj_matrix,
                            graph, devices_seen, node_id, dev, next_shard)
        devices_seen.pop()
    else:
        # Termination: we scheduled all shards; now the output must reach the destination device.
        # PipeEdge doesn't currently support loops in the device pipeline, except to send results
        # back to the source device (since the source doesn't receive from any other device).
        if device == dev_dest:
            graph.graph['tails'].append(node_id)
        elif adj_matrix[device.devno][dev_dest.devno] > 0 and \
             (dev_dest not in devices_seen or dev_src == dev_dest):
            node_id_dest = (node_id[0] + ((),), node_id[1] + (dev_dest,))
            assert node_id_dest not in graph.nodes
            graph.add_node(node_id_dest, weight=0)
            comm_bytes = ubatch_bytes(model.parameters_out[-1], ubatch_size, dtype=dtype)
            comm_time = communication_time_bw(adj_matrix[device.devno][dev_dest.devno], comm_bytes)
            graph.add_edge(node_id, node_id_dest, weight=comm_time)
            graph.graph['tails'].append(node_id_dest)
        # else: We can't form a complete pipeline - oh well.

def _shortest_path_with_cost(graph: nx.DiGraph, source: NodeID, target: NodeID) -> \
    Tuple[List[NodeID], float]:
    # Cost overlaps compute time on source and communication time from source to target.
    target_comp_time = graph.nodes[target]["weight"]
    def calc_weight(src, targ, edge):
        comp_time = graph.nodes[src]["weight"]
        comm_time = edge["weight"]
        # Account for computation cost of the last device.
        # This must be done here (not externally) to get the true shortest path.
        tail_comp_time = target_comp_time if targ == target else 0
        return max(comp_time, comm_time) + tail_comp_time
    spath = nx.algorithms.shortest_path(graph, source=source, target=target, weight=calc_weight)
    # Networkx's shortest path functions don't give us the resulting cost, so we recompute them.
    if len(spath) == 1:
        # Device's compute time wasn't considered since there are no edges (calc_weight wasn't run).
        assert source == target
        cost = target_comp_time
    else:
        cost = sum(calc_weight(spath[i], spath[i + 1], graph[spath[i]][spath[i + 1]])
                   for i in range(len(spath) - 1))
    return (spath, cost)
