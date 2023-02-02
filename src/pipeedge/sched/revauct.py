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


def _bids_to_device_list(bids: Mapping[str, DeviceBidData]) -> List[_Device]:
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
    return dev_list

def _devs_to_adj_matrix(dev_list: List[_Device]) -> np.ndarray:
    adj_matrix = np.zeros((len(dev_list), len(dev_list)))
    for dev in dev_list:
        for key, item in dev.neighbors.items():
            if dev.devno != key.devno:
                adj_matrix[dev.devno][key.devno] = item['bw_Mbps']
    return adj_matrix

# Return type isn't the cleanest, but is compatible with the YAML one used in scheduler.py
def sched_min_latencies(yml_model: dict, ubatch_size: int, dtype: str,
                        bids: Mapping[str, DeviceBidData]) -> List[Mapping[str, List[int]]]:
    """Schedule for minimum latency (cost), accounting for computation and communication overlap."""
    model = _Model(yml_model)
    logger.debug("Created model: %s", model)

    # Populate device class instances with their bids
    dev_list = _bids_to_device_list(bids)

    # Create adjacency matrix of bandwidths between devices.
    adj_matrix = _devs_to_adj_matrix(dev_list)
    logger.debug("Adjacency matrix: %s", adj_matrix)

    # Get a mapping of layer pairs to devices that support those layer pairs
    layer_bid_devs = _layer_bids_to_devices(dev_list)

    # Now do the scheduling
    graphs = _create_sched_digraphs(layer_bid_devs, model, ubatch_size, dtype, adj_matrix)
    sched = _get_best_schedule(graphs, model)

    if len(sched[0]) == 0:
        logger.debug("No possible paths.")
    else:
        logger.debug("The pipeline runs in: %f seconds.", sched[2])
        for stage, (dev, layers) in enumerate(zip(sched[0], sched[1])):
            logger.debug("Stage: %d\n  Device: %s\n  Layers: %s", stage, dev.name, layers)
    return [{ dev.name: [layers[0], layers[-1]] }
            for dev, layers in zip(sched[0], sched[1])]


NodeID: Type = Tuple[Tuple[Tuple[int, int], ...], Tuple[_Device, ...]]
"""Two tuples: (1) Layer ranges up to and including this node, (2) Devices assigned to (1)."""

# Takes list of devices that bid plus the model and builds a dictionary whose key, value pair is
# the type of layer window possible and all devices that can support that window
def _layer_bids_to_devices(dev_list: List[_Device]) -> Mapping[Tuple[int, int], List[_Device]]:
    layer_bid_devs = {}
    for dev in dev_list:
        layer_bids = set(tuple(bid) for bid in dev.bids)
        for layer_bid in layer_bids:
            if layer_bid not in layer_bid_devs:
                layer_bid_devs[layer_bid] = []
            layer_bid_devs[layer_bid].append(dev)
    return layer_bid_devs

# Constructs all possible graphs where nodes are assigned layer windows and devices.
# All directed paths in a graph represent a possible layer/device schedule.
def _create_sched_digraphs(layer_bid_devs: Mapping[Tuple[int, int], List[_Device]], model: _Model,
                           ubatch_size: int, dtype: str, adj_matrix: np.ndarray) -> \
    List[nx.DiGraph]:
    # Create a directed graph where each node is an allowable layer assignment.
    path_generator_graph = nx.DiGraph()
    path_generator_graph.add_nodes_from(layer_bid_devs.keys())
    keys_max = [max(k) + 1 for k in layer_bid_devs]
    keys_min = [min(k) for k in layer_bid_devs]
    for key1, key1_max in zip(layer_bid_devs, keys_max):
        for key2, key2_min in zip(layer_bid_devs, keys_min):
            if key1_max == key2_min: # connect them
                path_generator_graph.add_edge(key1, key2)

    # All possible paths from nodes containing the first layer to nodes containing the last layer.
    keys_w_first_layer = [k for k in layer_bid_devs if 0 in k]
    keys_w_last_layer = [k for k in layer_bid_devs if model.layers - 1 in k]
    possible_paths = [nx.algorithms.all_shortest_paths(path_generator_graph, key1, key2)
                      for key2 in keys_w_last_layer
                      for key1 in keys_w_first_layer]

    graphs = []
    for possible_path in possible_paths:
        try:
            for path in possible_path:
                for dev in layer_bid_devs[path[0]]:
                    # Build a weighted directed graph for each device that can start this path
                    graph = nx.DiGraph(path=path, root=dev)
                    graphs.append(graph)
                    _build_tree(graph, path, 0, None, dev, [], layer_bid_devs, adj_matrix, model,
                                ubatch_size, dtype)
        except nx.exception.NetworkXNoPath:
            continue #print("Cannot get there from here") # path doesn't exist for this combination
    return graphs

# Recursively builds a tree from the node passed in to all its children
def _build_tree(graph: nx.DiGraph, path: List[Tuple[int, int]], path_idx: int, node_id_prev: NodeID,
                device: _Device, devices_seen: List[_Device],
                layer_bid_devs: Mapping[Tuple[int, int], List[_Device]], adj_matrix: np.ndarray,
                model: _Model, ubatch_size: int, dtype: str) -> None:
    # Add the graph node, then add a directed edge from the previous node (if there is one)
    min_layer = path[path_idx][0]
    max_layer = path[path_idx][-1]
    comp_time = device.bids[(min_layer, max_layer)]
    node_id = (tuple(path[:path_idx + 1]), tuple(devices_seen) + (device,))
    assert node_id not in graph.nodes
    graph.add_node(node_id, weight=comp_time)
    if node_id_prev is not None:
        dev_prev = node_id_prev[1][-1]
        params = model.parameters_in if min_layer == 0 else model.parameters_out[min_layer - 1]
        comm_bytes = ubatch_bytes(params, ubatch_size, dtype=dtype)
        assert adj_matrix[dev_prev.devno][device.devno] > 0
        comm_time = communication_time_bw(adj_matrix[dev_prev.devno][device.devno], comm_bytes)
        graph.add_edge(node_id_prev, node_id, weight=comm_time)

    # Recursion: look for unseen devices that can handle the next layer(s)
    if len(path) > path_idx + 1:
        devices_seen.append(device)
        for dev in layer_bid_devs[path[path_idx + 1]]:
            if adj_matrix[device.devno][dev.devno] > 0 and dev not in devices_seen:
                _build_tree(graph, path, path_idx + 1, node_id, dev, devices_seen, layer_bid_devs,
                            adj_matrix, model, ubatch_size, dtype)
        devices_seen.pop()

def _get_best_schedule(graphs: nx.DiGraph, model: _Model) -> \
    Tuple[Tuple[_Device, ...], Tuple[Tuple[int, int], ...], float]:
    shortest_paths = []
    for graph in graphs:
        shortest_paths.extend(_shortest_complete_paths(graph, model))
    best_sched = ((), (), float('inf')) # (device_tuple, layer_schedule_tuple, cost)
    for path, cost in shortest_paths:
        if cost < best_sched[2]:
            # The final path entry contains the device and layer info we need
            best_sched = (path[-1][1], path[-1][0], cost)
    return best_sched

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

def _shortest_complete_paths(graph: nx.DiGraph, model: _Model) -> List[Tuple[List[NodeID], float]]:
    # There's one source node and >= 0 target nodes that have the last model layer.
    shortest_paths = []
    # reconstruct source node ID
    node_id_src = ((graph.graph['path'][0],), (graph.graph['root'],))
    for node_id_targ in graph.nodes:
        targ_last_layer = node_id_targ[0][-1][-1]
        if targ_last_layer == model.layers - 1:
            shortest_paths.append(_shortest_path_with_cost(graph, node_id_src, node_id_targ))
    return shortest_paths
