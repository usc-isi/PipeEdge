#include"cnpy.h"
#include<complex>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include <tuple>
// #include <bits/stdc++.h>
using int64 = long long;
const int N = 200005;

int num_layers = 0;

// node 0: C2558 
// node 1: E3845
// node 2: C2558 75% 8GB
// node 3: C2558 50% 4GB
// node 4: C2558 25% 4GB 
const int device_type_num = 5;
const int device_count[] = {4,4,0, 0,8};
double* time_p_0;
double* time_p_1;
double* time_p_2;
double* time_p_3;
double* time_p_4;
double* memory;
int64* data_shape;

int decide_layers(std::string model_name) {
  if(model_name == "vit-large-patch16-224") num_layers=96;
  else if(model_name == "vit-huge-patch14-224-in21k") num_layers=128;
  else num_layers=48;
  return num_layers;
}

double* GetProfilingTime(int node) {
  if(node == 0) return time_p_0;
  else if (node == 1) return time_p_1;
  else if (node == 2) return time_p_2;
  else if (node == 3) return time_p_3;
  else if (node == 4) return time_p_4;
  return time_p_4;
}

double GetComputationTime(int layer_l, int layer_r, int node) {
    double comp_time = 0;
    for(int i = layer_l - 1; i < layer_r; i++){
        double* time_p =  GetProfilingTime(node); 
        double tmp;
        if(node == 4){
          tmp = (double)time_p[i];
        }else tmp = (double)time_p[i];
        // double tmp = (double)time_p[i];
        comp_time += tmp;
    }
  return comp_time;

}

double GetBandwidth(int node_u, int node_v) {
    // Mbps
    // printf("node_u is %d, node_v is %d\n", node_u, node_v);
    // if(node_u<1 && node_v<1){
    //   return 1;
    // }
    return 1000;
}

double GetCommunicationTime(int layer_r, int node_u, int node_v) {
    double bandwidth = GetBandwidth(node_u, node_v);
    // transmission data size + residul connection -> Mb
    // *8 use batch size 8
    double data_size = (double)data_shape[layer_r-1]*2*32/1000000; 
    double comm_time = (data_size/bandwidth)*1.1;
    // printf("layer_r is %d, datashape is %lld, node_u is %d, v is %d, data_size is %lf, bandwidth is %f, comm_time is %f\n", layer_r, data_shape[layer_r-1], node_u, node_v, data_size, bandwidth, comm_time);
    return comm_time;
    
}

double GetNodeMemory(int node) {
  if(node == 0) return 8000;
  else if(node == 1) return 2000;
  else if(node == 2) return 4000;
  return 4000;
}
bool isCapacityAllowed(int layer_l, int layer_r, int node) {
  // communication memory overheads
  double needed_memory = (memory[0] + memory[layer_r-1] - memory[layer_l-1])*1.6;
  // printf("node is %d, layer_l is %d, layer_r is %d, Memory is %f, memory[0] is %f, r is %f, l is %f\n", node, layer_l, layer_r, needed_memory, memory[0], memory[layer_r-1], memory[layer_l-1]);
  double node_memory =GetNodeMemory(node); //MB
  return node_memory*0.7 > needed_memory;
}

void work() {
  int mask = 1;
  for (int i = 0; i < device_type_num; ++i) {
    mask *= device_count[i] + 1;
  }

  std::cout << mask << std::endl;
  std::vector<std::vector<double>> h[num_layers + 1];
  std::vector<std::vector<std::pair<int, int>>> parent[num_layers + 1];
  for (int i = 0; i <= num_layers; ++i) {
    printf("Initial progress %d \\ %d\n", i, num_layers);
    for (int j = 0; j < mask; ++j) {
      h[i].emplace_back(std::vector<double>(device_type_num, 1e60));
      parent[i].emplace_back(std::vector<std::pair<int, int>>(
          device_type_num, std::make_pair(-1, -1)));
    }
  }
  std::vector<int> prefix_product(device_type_num + 1);
  for (int i = 0; i <= device_type_num; ++i) {
    prefix_product[i] =
        i == 0 ? 1 : prefix_product[i - 1] * (device_count[i - 1] + 1);
  }
  // num of device_i in S  == S / prefix_product[i] % (device_count[i] + 1);

  for (int u = 0; u < device_type_num; ++u) {
      h[0][0][u] = 0;
  }
  double res = 1e60;
  std::tuple<int, int, int> res_index = {-1, -1, -1};
  for (int i = 0; i < num_layers; ++i) {
    printf("Finish progress %d \\ %d\n", i, num_layers);;
    for (int S = 0; S < mask; ++S) {
      for (int u = 0; u < device_type_num; ++u) {
        if (h[i][S][u] > 1e59) {
          continue;
        }
        if (S / prefix_product[u] % (device_count[u] + 1) == device_count[u]) {
          // u in fully used in S
          continue;
        }
        for (int j = i + 1; j <= num_layers; ++j) {
          if (!isCapacityAllowed(i + 1, j, u)) {
            continue;
          }
          // assign layers [i + 1, j] to node u
          double computation_time = GetComputationTime(i + 1, j, u);
        //   printf("Current Node is %d, comp time is %lf\n", u, computation_time);
          if (j == num_layers) {
            double cost = std::max(h[i][S][u], computation_time);
            if (cost < res) {
              res = cost;
              res_index = {i, S, u};
            }
          }
          for (int v = 0; v < device_type_num; ++v) {
            if (S / prefix_product[v] % (device_count[v] + 1) ==
                device_count[v]) {
              // v in fully used in S
              continue;
            }
            double communication_time = GetCommunicationTime(j, u, v);
            // printf("Communication time is %lf\n", communication_time);
            double cost = std::max(
                h[i][S][u], std::max(computation_time, communication_time));
            // S + prefix_product[u] => S U {u}
            if (cost < h[j][S + prefix_product[u]][v]) {
              h[j][S + prefix_product[u]][v] = cost;
              parent[j][S + prefix_product[u]][v] = std::make_pair(i, u);
            }
          }
        }
      }
    }
  }

  printf("Minimum time : %f\n", res);
  int layer, S, u;
  std::tie(layer, S, u) = res_index;

  // calculate the selected nodes
  int selected_nodes = 0;
  while(layer > 0) {
      int last_l, last_u;
      selected_nodes++;
      std::tie(last_l, last_u) = parent[layer][S][u];
      layer = last_l;
      S -= prefix_product[last_u];
      u = last_u;
  }

  std::vector<int> partition;
  std::tie(layer, S, u) = res_index;
  printf("layer [%d, %d] used by node %d, rank %d\n", layer + 1, num_layers, u, selected_nodes);
  partition.push_back(layer + 1);
  partition.push_back(num_layers);
  while (layer > 0) {
    int last_l, last_u;
    selected_nodes--;
    std::tie(last_l, last_u) = parent[layer][S][u];
    partition.push_back(last_l + 1);
    partition.push_back(layer);
    printf("layer [%d, %d] used by node %d, rank %d\n", last_l + 1, layer, last_u,selected_nodes);
    layer = last_l;
    S -= prefix_product[last_u];
    u = last_u;
  }
  std::sort(partition.begin(), partition.end());
  for(auto& it : partition) {
    std::cout<<it<<",";
  }

}

int main(int argc, char **argv) {
  // argv[1]: model name (vit-base-patch16-224) vit-large-patch16-224 vit-huge-patch14-224-in21k
  // argb[2]: device name (Core_i5)
  std::string model_name = argv[1];
  num_layers = decide_layers(model_name);
  printf("num_layers is %d\n", num_layers);
  std::string time_file_name_0 = "./profiling/" + model_name + "_" + "C2558" + "_8.npz";
  std::string time_file_name_1 = "./profiling/" + model_name + "_" + "E3845" + "_8.npz";
  std::string time_file_name_2 = "./profiling/" + model_name + "_" + "C2558_75" + "_8.npz";
  std::string time_file_name_3 = "./profiling/" + model_name + "_" + "C2558_50" + "_8.npz";
  std::string time_file_name_4 = "./profiling/" + model_name + "_" + "C2558_10" + "_8.npz";
  std::string shape_file_name = "./profiling/" + model_name + "_shape.npz";
  std::string memory_file_name = "./profiling/" + model_name + "_memory.npz";
  std::cout<<"Loading profiling file:" <<time_file_name_0<<" "<<shape_file_name;
  cnpy::NpyArray arr_node0 = cnpy::npz_load(time_file_name_0,"time");
  time_p_0 = arr_node0.data<double>();

  cnpy::NpyArray arr_node1 = cnpy::npz_load(time_file_name_1,"time");
  time_p_1 = arr_node1.data<double>();

  cnpy::NpyArray arr_node2 = cnpy::npz_load(time_file_name_2,"time");
  time_p_2 = arr_node2.data<double>();

  cnpy::NpyArray arr_node3 = cnpy::npz_load(time_file_name_3,"time");
  time_p_3 = arr_node3.data<double>();

  cnpy::NpyArray arr_node4 = cnpy::npz_load(time_file_name_4,"time");
  time_p_4 = arr_node4.data<double>();

  cnpy::NpyArray arr2 = cnpy::npz_load(shape_file_name,"shape");
  data_shape = arr2.data<int64>(); 
  cnpy::NpyArray arr3 = cnpy::npz_load(memory_file_name,"memory");
  memory = arr3.data<double>(); 
//   printf("shape is %d %d, %d", arr.shape.size(), arr.shape[0], arr.shape[1]);
  for(int i = 0; i < arr_node0.shape[0]; i++) {
      printf("time_p_0[%d] = %lf, time_p_1[%d]=%lf, data_shape[%d] = %lld  ", i, time_p_0[i], i, time_p_1[i], i, data_shape[i]);
  }
  work();
  return 0;
}

//  g++  ./partition/partition.cpp -o a -O2 -L/usr/local -lcnpy -lz --std=c++17
// ./a vit-base-patch16-224 
// ./a vit-large-patch16-224 
// ./a vit-huge-patch14-224-in21k

// 2 4 3
// 3 * 5 * 4 = 60
// 2 ^ {9} = 512 => 00 0000 000

// {0, 0 , 0} => 0
// {1, 0 , 0} => 1
// {2, 0 , 0} => 2
// {0, 1 , 0} => 3
// .
// .
// {2, 4 , 2} => 58
// {2, 4 , 3} => 59

// {i, j , k} => i + j * 3 + k * 15 = x
// i = x / 1 % 3
// j = x / 3 % 5
// k = x / 15 % 4