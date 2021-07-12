import numpy as np

#########################################################
#              Nodes and Bandwidth Settings            #
#########################################################


cpu_100 =[0.10, 0.12, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.25, 0.25, 0.40, 0.40, 0.35, 0.35, 0.3, 0.3,  0.35, 0.35, 0.35, 0.35, 1.0, 1.0, 0.45, 0.45]
cpu_70 = [0.44, 1.01, 0.43, 0.09, 0.96, 0.26, 0.98, 0.66, 0.87, 0.55, 0.50, 0.14, 0.69, 0.11, 0.76,0.64, 0.17,  0.5,  0.5,  0.5, 1.5,  1.5,  0.6,  0.6]
cpu_50 = [0.75, 1, 0.6,  0.6,  0.6,  0.6,  0.65, 0.65, 0.6,  0.6,  0.65,0.65,0.8,  0.8,  0.8, 0.8,  0.8,  0.8, 0.85, 0.85, 2, 2,  0.75, 0.75]
cpu_30 = [1.1,  1.1,  1,    1,    1,    1,    1.1,  1.1,  1,    1,    0.8,  0.8,1.4,  1.4,  1.25,1.25, 1,    1,    1,    1,   3,     3,  1.25, 1.25]

num_layers = 12
node_frequency = [100, 100, 100,100, 70, 70, 70,70, 50, 50, 50,50]
list_latency = [0.533, 0.533, 1.5, 1.5, 2.1666, 2.1666, 3.8333, 3.8333]
num_nodes = 12 
bandwidth = np.array([[-1, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000],
					  [10000000, -1, 10000000, 10000000,10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000],
					  [10000000, 10000000, -1, 10000000,10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000],
					  [10000000, 10000000, 10000000, -1,10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000],
					  [10000000, 10000000, 10000000, 10000000,-1, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000],
					  [10000000, 10000000, 10000000, 10000000,10000000, -1, 10000000, 10000000,10000000, 10000000, 10000000, 10000000],
					  [10000000, 10000000, 10000000, 10000000,10000000, 10000000, -1, 10000000,10000000, 10000000, 10000000, 10000000],
					  [10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, -1,10000000, 10000000, 10000000, 10000000],
					  [10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000,-1, 10000000, 10000000, 10000000],
					  [10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000,10000000, -1, 10000000, 10000000],
					  [10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000,10000000, 10000000, -1, 10000000],
					  [10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, 10000000,10000000, 10000000, 10000000, -1]])

#########################################################
#                      Time function                    #
#########################################################

def layer_device_latency(frequency, layer_i):
	if frequency == 100:
		return cpu_100[layer_i]
	elif frequency == 70:
		return cpu_70[layer_i]
	elif frequency == 50:
		return cpu_50[layer_i]
	elif frequency == 30:
		return cpu_30[layer_i]
	else:
		print("Not Found in dataset")

def device_to_latency(frequency, num_layers,start_layer, end_layer, use_latency=False, enable_print=False):
	latency = 0
	if use_latency == True:
		for layer_i in range(start_layer, end_layer):
			latency += layer_device_latency(int(frequency), layer_i)
			if enable_print== True:
				print(f"from {start_layer} to {end_layer}, current {layer_i}  frequency is{frequency} ,latency is {latency}")
	else:	
		base_freq = 1.5
		base_layer = 1
		base_latency_1_layer = 1.5
		ratio = frequency / base_freq * 1.34
		latency = num_layers*base_latency_1_layer/ratio
	return latency

def latency_direct(latency_1_layer, num_layers):
	latency = num_layers*latency_1_layer
	return latency

def devices_to_latencies(list_frequency, list_num_layers):
	list_latency = []
	for i in range(len(list_frequency)):
		latency = device_to_latency(list_frequency[i], list_num_layers[i])
		list_latency.append(latency)
	return list_latency

def communication_time_between_device(bandwidth = 10000000, data_size = 442368):
	return data_size*32/bandwidth


# print(devices_to_latencies([1.5, 3.4], [1, 2])) 
# print(communication_time_between_device(10000000, 442368))


def communication_time_between_id(src, dst, data_size=442368):
	bandwidth_src_dst = bandwidth[int(src)][int(dst)]
	return communication_time_between_device(bandwidth_src_dst, data_size)


def baseline_evenly_partition(total_layer, list_frequency, data_size):
	list_frequency.sort(reverse=True)
	partition_layer = total_layer // len(list_frequency)
	last_layer = total_layer - len(list_frequency)*partition_layer
	max_latency = 0
	for i in range(len(list_frequency)):
		if i != 0:
			max_latency = max(max_latency, communication_time_between_id(i-1, i, data_size))
		if i != len(list_frequency)-1:
			max_latency = max(max_latency, device_to_latency(list_frequency[i], partition_layer))
		else:
			max_latency = max(max_latency, device_to_latency(list_frequency[i], last_layer))
	return max_latency


#########################################################
#                      DP Solution                     #
#########################################################
mask = 2 ** len(node_frequency)

h = [[[-1 for k in range(num_nodes)] for j in range(mask)] for i in range(num_layers + 1)]

print(bin(mask))
h[0][0][0] = 0
parent = {}
data_size = 1000
for i in range(num_layers):
	for S in range(mask):
		for last_used in range(num_nodes):
			current_cost = h[i][S][last_used]
			if current_cost < 0:
				continue
			for j in range(i, num_layers):
				for node in range(len(node_frequency)):
					if (S >> node & 1): ## 1 if used
						continue
					# try use "node" to handle the layers [i, j]
					# frequency, num_layers,start_layer, end_layer, use_latency=False
					next_cost = max(current_cost, device_to_latency(node_frequency[node],0, i,j+1,use_latency=True), (communication_time_between_id(last_used, node, data_size) if i > 0 else 0))
					
					# print(f"next_cost is {next_cost}")
					# next_cost = max(current_cost, device_to_latency(node_frequency[node], j - i + 1), (communication_time_between_id(last_used, node, data_size) if i > 0 else 0))
					# next_cost = max(current_cost, latency_direct(list_latency[node], j - i + 1), (communication_time_between_id(last_used, node, data_size) if i > 0 else 0))
					
					remained = (S ^ (1 << node)) # flip the node since it was use 
					if h[j + 1][remained][node] < 0 or next_cost < h[j + 1][remained][node]:
						h[j + 1][remained][node] = next_cost
						# print(f"now node is {node}, last_used is {last_used} \n")
						parent[(j + 1, remained, node)] = (i, node, last_used)

#########################################################
#                      Print answer                     #
#########################################################

answer = (-1, -1, -1)
for i in range(mask):
	for j in range(num_nodes):
		if h[num_layers][i][j] >= 0:
			pair = (h[num_layers][i][j], i, j)
			answer = pair if answer[0] < 0 else min(answer, pair)
print("\n\nThe minimum latency is %f \n\n" % answer[0])

layer, s, last_used = (num_layers, answer[1], answer[2])
while layer > 0:
	p = parent[(layer, s, last_used)]
	print("layers in [%d , %d], calculated by node %d with frequency %f,\nbandwidth %d between %d and %d -> latency %f" 
		% (p[0], layer - 1, p[1], node_frequency[p[1]], bandwidth[p[1]][p[2]], p[1], p[2], 
		communication_time_between_id(p[1], p[2], data_size)))
	# print("latency for node %d is %f\n"%(p[1], device_to_latency(node_frequency[p[1]] ,layer - p[0] )) )
	print("latency for node %d is %f\n"%(p[1], device_to_latency(node_frequency[p[1]], 0, p[0], layer, use_latency=True,enable_print=True)) )
	layer = p[0]
	s -= (1 << p[1])
	last_used = p[2]



