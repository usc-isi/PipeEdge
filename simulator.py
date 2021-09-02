import numpy as np
import torch
import sys
from graph import Graph
import argparse
from profile import ProfileTransformer


#########################################################
#              Define Device/node Class                 #
#########################################################
class Device():
    def __init__(self, ID, memory, computation, inbound_bandwidth, outbound_bandwidth, model_name):
        self.ID = ID
        self.memory = memory
        self.model_name = model_name
        self.computation = computation
        self.inbound_bandwidth = inbound_bandwidth
        self.outbound_bandwidth = outbound_bandwidth
        self.comm_time = 0
        self.comp_time = 0
        self.microbatch_size = 0
        self.successors = None
        self.send_data_list = None
        self.stage = -1 # start index from 0, for double-check 

    def check_pass_memory_limit(self, runtime_memory, ratio=0.7):
        pass_check = False
        if runtime_memory < self.memory*ratio:
            pass_check = True 
        return pass_check
    
    def return_profiling_time(self, start_ops, end_ops):
        profiling_file_name = self.model_name.split('/')[-1] + "_" + str(self.computation) + "_" + str(self.microbatch_size)+".npz"
        # TODO put in profiling result file 
        print(f"profiling file name is {profiling_file_name}")
        profiling_file = np.load(profiling_file_name)
        # profiling_file = np.load("vit-base-patch16-224_Core_i5_8.npz")
        time_p = profiling_file['time']
        comp_time = 0 
        for i in range(start_ops-1, end_ops):
            # comp_time += profiling_file[i]
            ## for test
            comp_time += time_p[i]
        return comp_time

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

def initial_devices(total_nodes, ID_list, memory_list, computation_list, inbound_bandwidth_list, outband_bandwidth_list, model_name):
    for i in range(total_nodes):
        _name = f'n{i}'
        _ID = ID_list[i]
        _memory = memory_list[i]
        _computation = computation_list[i]
        _inbound_bandwidth = inbound_bandwidth_list[i]
        _outbound_bandwidth = outband_bandwidth_list[i]
        globals()[_name] = Device(_ID, _memory, _computation, _inbound_bandwidth, _outbound_bandwidth, model_name)


#########################################################
#           Communication Time Calculation              #
#########################################################

def return_actual_bandwidth(sender, receiver):
    return min(sender.outbound_bandwidth, receiver.inbound_bandwidth)

def return_communication_time(sender, receiver_list, data_size_list, process_time = 0.1):
    comm_time = 0
    num_receivers = len(receiver_list)
    if sender == receiver_list[0]:
        return 0
    for i in range(num_receivers):
        bandwidth = return_actual_bandwidth(sender, receiver_list[i])
        data_size = data_size_list[i]
        comm_time = max(comm_time, data_size/bandwidth)
    return num_receivers*process_time + comm_time

def detach_receiver_data(receiver_data_set, single_data_size = 4.7, to_class=True):
    receivers_list = []
    datasize_list = []
    for i in range(len(receiver_data_set)):
        if to_class:
            receivers_list.append(str_to_class(receiver_data_set[i][0]))
        else:
            receivers_list.append(receiver_data_set[i][0])
        datasize_list.append(receiver_data_set[i][1]*single_data_size)
    return receivers_list, datasize_list
         
def memory_runtime(memory_list, start_ops, end_ops):
    return memory_list[0] + memory_list[end_ops-1] - memory_list[start_ops-1]
#########################################################
#            Stage Exectuion Time estimation            #
#########################################################

def simulator(stages, partition, time_profile, comm_time, batch_size):
    max_stage_time = 0
    slowest_stage = -1
    stages_time_list = []
    comm_bound = False
    for i in range(len(stages)):
        start_ops = partition[2*i]
        end_ops = partition[2*i+1]
        comp_time_stage = 0
        for j in range(start_ops-1, end_ops):
            comp_time_stage += time_profile[j]
        comp_time_stage /= stages[i]
        comm_time_stage = comm_time[(end_ops-1)%4]
        stage_time = max(comp_time_stage, comm_time_stage)
        stages_time_list.append(stage_time)
        if stage_time> max_stage_time:
            slowest_stage = i
            max_stage_time = stage_time
            if comm_time_stage > comp_time_stage:
                comm_bound = True

    throughput = batch_size/max_stage_time
    return max_stage_time, stages_time_list, throughput, slowest_stage,comm_bound

def heterogenous_simulator(partition, graph):
    visited = {}
        # Mark all the vertices as not visited
    for _, key in enumerate(graph):
        visited[key] = False

    start_node = "Input"
    queue = []
    if start_node in graph:
        queue.append("Input")
    else:
        print("Input must be the first node")
    visited[start_node] = True
    stage = 0
    period_time = 0 
 
    while queue:
        count = len(queue)
        stage_period_time = 0
        while count > 0:
            s = queue.pop(0)
            for i in graph[s]:
                node = i[0]
                if visited[node] == False:
                    queue.append(node)
                    visited[node] = True
                    ## calculate node computation time
                    print(f"{node} starts ops from {partition[2*stage]} to {partition[2*stage+1]}")
                    ## check node memory limitation
                    memory_requirment = memory_runtime(memory_kernel,partition[2*stage], partition[2*stage+1] )
                    pass_memory_check = str_to_class(node).check_pass_memory_limit(memory_requirment)
                    if pass_memory_check == False:
                        print(f"Cannot pass the memory limitation check on node {str_to_class(node).ID} with {str_to_class(node).memory} < {100000}")
                        return -1
                    comp_time = str_to_class(node).return_profiling_time(partition[2*stage], partition[2*stage+1])
                    ##c calculate node communication time
                    receivers_tuple_list = graph[node]
                    receivers_list, datasize_list = detach_receiver_data(receivers_tuple_list)
                    comm_time = return_communication_time(str_to_class(node), receivers_list, datasize_list)
                    ## the execution time of this node
                    node_period_time = max(comp_time, comm_time)
                    stage_period_time = max(stage_period_time, node_period_time)
                    print(f"{node} comp time is {comp_time}, comm time is {comm_time}, period time is {node_period_time}")

            count -= 1
        stage += 1
        period_time = max(stage_period_time, period_time)
        print(f"Stage {stage}: stage period time is {stage_period_time}, global period time is {period_time}")
    return period_time


if __name__=="__main__":
    #########################################################
    #                 Configure the Input Data              #
    #########################################################
    parser = argparse.ArgumentParser(description="Simulator for Parallelism")
    parser.add_argument("-s","--stages", default="1,1", help="the replicas for each stage, eg. 1,1,2,1 means 2 replicas in stage 2 (index from 0)")
    parser.add_argument("-bd","--bandwidth",type=int, default=500,help="the bandwidth for communication")
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224", choices=["google/vit-base-patch16-224", 
    "google/vit-large-patch16-224", "google/vit-huge-patch14-224-in21k"], help="the neural network model for loading")
    parser.add_argument("-pt", "--partition", default="1,24,25,48", help="the partition method")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("-r", "--repeat-time", default=1, type=int, help="repeat time for profiling")

    args = parser.parse_args()
    stages = [int(i) for i in args.stages.split(',')]
    bandwidth = args.bandwidth #mbps
    model_name = args.model_name
    partition = [int(i) for i in args.partition.split(',')]
    batch_size = args.batch_size

    partition = [1,12, 13,24, 25,48]
    l = {"Input": [("n0", 5), ("n1", 5)], "n0": [("n2", 2), ("n3",3)], "n1":[("n3",3),("n4",2)],"n2": [("n5", 2)], "n3": [("n6", 6)], "n4":[("n5",2)], "n5":[("n5",0)],"n6":[("n6",0)]}
    g = Graph(l)
    g.print_graph()
    total_nodes = 7
    ID_list = [0,1,2,3,4,5,6]
    memory_list = [2000, 2000, 2000, 8000, 8000,4000,5000]
    computation_list = ["Core_i5" , "Core_i5", "Core_i5","Core_i5","Core_i5", "Core_i5", "Core_i5"]
    inbound_bandwidth_list = [100, 500, 1000, 300, 300,1000,500]
    outband_bandwidth_list = [100, 500, 1000, 300, 300, 1000,500] 
    model_name = "google/vit-base-patch16-224"

    #########################################################
    #                    Profiling Resutls                  #
    #########################################################
    profiling_file = np.load("vit-base-patch16-224_Core_i5_8.npz")
    memory_kernel = profiling_file['memory']
    initial_devices(total_nodes, ID_list, memory_list, computation_list, inbound_bandwidth_list, outband_bandwidth_list, model_name)
    node_microbatch_size = g.statistic_microbatch_size("Input")
    for i,key in enumerate(node_microbatch_size):
        str_to_class(key).microbatch_size = node_microbatch_size[key]
        print(f"{key}:{str_to_class(key).microbatch_size}")
    
    period_time = heterogenous_simulator(partition, l)
    print(f"Estimate period time is {period_time}")


# max_stage_time, stages_time_list, throughput, slowest_stage,comm_bound = simulator(stages, partition, time_p, comm_time, batch_size)
# print(f"max stage time is {max_stage_time}, slowest_stage is {slowest_stage}, time list is {stages_time_list}, throughput is {throughput} images/second, communication bound is {comm_bound}")

