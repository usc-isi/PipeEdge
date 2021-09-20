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

const int num_layers = 128;

const int device_type_num = 1;
const int device_count[] = {6};
double* time_p;
int* data_shape;

double GetComputationTime(int layer_l, int layer_r, int node) {
    double comp_time = 0;
    for(int i = layer_l - 1; i < layer_r; i++){
        double tmp = (double)time_p[i];
        comp_time += tmp;
    }
  return comp_time;

}

double GetBandwidth(int node_u, int node_v) {
    // Mbps
    return 300.0;
}

double GetCommunicationTime(int layer_r, int node_u, int node_v) {
    double bandwidth = GetBandwidth(node_u, node_v);
    // transmission data size + residul connection -> Mb
    double data_size = data_shape[layer_r-1]*2/1000000; 
    double comm_time = data_size/bandwidth;
    return comm_time;
    
}
bool isCapacityAllowed(int layer_l, int layer_r, int node) {
  return true;
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
  std::string device_name = argv[2];
  std::string time_file_name = "./profiling/" + model_name + "_" + device_name + "_8.npz";
  std::string shape_file_name = "./profiling/vit-base-patch16-224_8.npz";
  std::cout<<"Loading profiling file:" <<time_file_name<<" "<<shape_file_name;
  cnpy::NpyArray arr = cnpy::npz_load(time_file_name,"time");
  time_p = arr.data<double>();
  cnpy::NpyArray arr2 = cnpy::npz_load(shape_file_name,"shape");
  data_shape = arr2.data<int>();
//   printf("shape is %d %d, %d", arr.shape.size(), arr.shape[0], arr.shape[1]);
  for(int i = 0; i < arr.shape[0]; i++) {
      printf("time_p[%d] = %lf, data_shape[%d] = %d  ", i, time_p[i], i, data_shape[i]);
  }
  work();
  return 0;
}

//  g++  ./partition/partition.cpp -o a -O2 -L/usr/local -lcnpy -lz --std=c++17
// ./a vit-base-patch16-224 Core_i5

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