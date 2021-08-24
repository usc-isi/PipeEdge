import numpy as np
import torch
from profile import ProfileTransformer

#########################################################
#                 Given Parameters Setting              #
#########################################################
total_stages = 4
stages = [1,1,2,1]
bandwidth = 500 #mbps
model_name = "google/vit-base-patch16-224"
partition = [1,12, 13,24, 25,36, 37,48]
microbatch_size = 8
   
model = ProfileTransformer(model_name)
inputs = torch.randn(microbatch_size,3,224,224)
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



max_stage_time, stages_time_list, throughput, slowest_stage,comm_bound = simulator(stages, partition, time_p, comm_time, microbatch_size)
print(f"max stage time is {max_stage_time}, slowest_stage is {slowest_stage}, time list is {stages_time_list}, throughput is {throughput} images/second, communication bound is {comm_bound}")

