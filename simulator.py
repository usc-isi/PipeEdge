import numpy as np
import torch
import argparse
from profile import ProfileTransformer

#########################################################
#                 Given Parameters Setting              #
#########################################################
parser = argparse.ArgumentParser(description="Simulator for Parallelism")
parser.add_argument("-s","--stages", default="1,1", help="the replicas for each stage, eg. 1,1,2,1 means 2 replicas in stage 2 (index from 0)")
parser.add_argument("-bd","--bandwidth",type=int, default=500,help="the bandwidth for communication")
parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224", choices=["google/vit-base-patch16-224", 
"google/vit-large-patch16-224", "google/vit-huge-patch14-224-in21k"], help="the neural network model for loading")
parser.add_argument("-pt", "--partition", default="1,24,25,48", help="the partition method")
parser.add_argument("-b", "--batch-size", default=8, type=int, help="batch size")
parser.add_argument("-r", "--repeat-time", default=10, type=int, help="repeat time for profiling")

args = parser.parse_args()
stages = [int(i) for i in args.stages.split(',')]
bandwidth = args.bandwidth #mbps
model_name = args.model_name
partition = [int(i) for i in args.partition.split(',')]
batch_size = args.batch_size

#########################################################
#                    Profiling Models                   #
#########################################################
   
model = ProfileTransformer(model_name, args.repeat_time)
inputs = torch.randn(batch_size,3,224,224)
time_p, total_time, data_shape = model(inputs)
data_Mbits = [i*32/10**6 for i in data_shape]
print(f"Time list is {time_p}, \ntotal time is {total_time} seconds, \ndata shape is {data_shape} Float32, {data_Mbits} Mega bits")


#########################################################
#           Communication Time Calculation              #
#########################################################

## current only support homo-bandwidth
comm_time = [i/bandwidth for i in data_Mbits]
print(f"Communication time is {comm_time} seconds")

#########################################################
#           Stage Exectuion Time Calculation            #
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



max_stage_time, stages_time_list, throughput, slowest_stage,comm_bound = simulator(stages, partition, time_p, comm_time, batch_size)
print(f"max stage time is {max_stage_time}, slowest_stage is {slowest_stage}, time list is {stages_time_list}, throughput is {throughput} images/second, communication bound is {comm_bound}")

