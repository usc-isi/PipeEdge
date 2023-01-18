#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "schedule.h"

#ifndef SCHEDULE_DEBUG
#define SCHEDULE_DEBUG 0
#endif

using namespace std;

static double compute_time(const vector<double> &t_comp, size_t layer_l, size_t layer_r) {
  double time = 0;
  for (size_t i = layer_l - 1; i < layer_r; i++){
    time += t_comp[i];
  }
  return time;
}

static size_t model_bytes_in(size_t parameters_in, size_t dtype_size, size_t batch_size) {
  return dtype_size * parameters_in * batch_size;
}

static size_t layer_bytes_out(const vector<model_layer> &layers, size_t layer,
                              size_t dtype_size, size_t batch_size) {
  size_t num_elements = layers[layer - 1].parameters_out;
  size_t data_bytes = dtype_size * num_elements * batch_size;
  return data_bytes;
}

static double comm_time(const vector<model_layer> &layers, size_t layer_r,
                        size_t dtype_size, size_t batch_size,
                        const vector<reference_wrapper<const device_type>> &devices, size_t node_u, size_t node_v) {
  size_t dat_bytes = layer_bytes_out(layers, layer_r, dtype_size, batch_size);
  double mbits_sec = min(devices[node_u].get().bw_Mbps, devices[node_v].get().bw_Mbps);
  double bytes_sec = mbits_sec * 1024 * 1024 / 8;
  double comm_time = static_cast<double>(dat_bytes) / bytes_sec;
  return comm_time;
}

static bool is_layers_fit(const vector<reference_wrapper<const device_type>> &devices, size_t node,
                          const vector<model_layer> &layers, size_t layer_l, size_t layer_r,
                          size_t parameters_in,
                          size_t dtype_size, size_t batch_size,
                          size_t data_buffers_in, size_t data_buffers_out) {
  /* memory used by the model weights */
  double mem_bytes_model = 0;
  assert(layer_l > 0);
  assert(layer_r >= layer_l);
  for (size_t i = layer_l - 1; i < layer_r; i++) {
    mem_bytes_model += layers[i].mem_MB * 1024 * 1024;
  }
  /* memory for data buffers */
  size_t dat_bytes_in;
  if (layer_l == 1) {
    dat_bytes_in = model_bytes_in(parameters_in, dtype_size, batch_size);
  } else {
    dat_bytes_in = layer_bytes_out(layers, layer_l - 1, dtype_size, batch_size);
  }
  size_t dat_bytes_out = layer_bytes_out(layers, layer_r, dtype_size, batch_size);
  // Communication and processing memory buffer overheads: send/recv/queue/processing buffers
  // Temporary processing buffers not accounted for - that's a function of the model impl
  size_t mem_bytes_buffers = 0;
  // receive buffer (and maybe queue)
  if (layer_l > 1) {
    mem_bytes_buffers += dat_bytes_in * data_buffers_in;
  }
  // send buffer (and maybe queue)
  mem_bytes_buffers += dat_bytes_out * data_buffers_out;
  // processing buffers (data not in queues or send/recv threads)
  mem_bytes_buffers += dat_bytes_in + dat_bytes_out;
  /* sum to get total memory required */
  double mem_bytes_req = mem_bytes_model + static_cast<double>(mem_bytes_buffers);
  /* actual available memory */
  double mem_bytes_avail = devices[node].get().mem_MB * 1024 * 1024;
  return mem_bytes_avail > mem_bytes_req;
}

static bool sort_dev_type_sched_stage(const dev_type_sched_stage &a, const dev_type_sched_stage &b) {
  // assumes no layer overlap
  return a.layer_r < b.layer_l;
}

void schedule_device_types(vector<dev_type_sched_stage> &dev_type_sched, const schedule_ctx &ctx) {
  const size_t num_layers = ctx.layers.size(); // L

  // Map device info to structures that algorithm needs
  // dev_types_count MUST keep the same ordering as ctx.dev_types to keep index tracking consistent
  vector<size_t> dev_types_count; // n = len(dev_types_count)
  vector<reference_wrapper<const device_type>> devices; // u (or v); |D| = len(devices)
  for (const auto& dev_type : ctx.dev_types) {
    // no viable overloaded operator[] b/c ctx is const, so use at()...
    size_t dev_type_count = ctx.dev_type_hosts.count(dev_type.name) > 0 ? ctx.dev_type_hosts.at(dev_type.name).size() : 0;
#if SCHEDULE_DEBUG
    cerr << "Devices: " << dev_type.name << ": " << dev_type_count << endl;
#endif
    dev_types_count.push_back(dev_type_count);
    for (size_t i = 0; i < dev_type_count; i++) {
      devices.push_back(ref(dev_type));
    }
  }

  size_t mask = 1;
  for (size_t i = 0; i < dev_types_count.size(); ++i) {
    mask *= dev_types_count[i] + 1;
  }

  vector<vector<vector<double>>> h(num_layers + 1);
  // TODO: the only signed type still used for indexing - can we eliminate (then rm assertions below)?
  vector<vector<vector<pair<ssize_t, ssize_t>>>> parent(num_layers + 1);
  for (size_t i = 0; i <= num_layers; ++i) {
#if SCHEDULE_DEBUG
    cerr << "Initial progress " << i << " \\ " << num_layers << endl;
#endif
    for (size_t j = 0; j < mask; ++j) {
      h[i].emplace_back(vector<double>(dev_types_count.size(), 1e60));
      parent[i].emplace_back(vector<pair<ssize_t, ssize_t>>(dev_types_count.size(), make_pair(-1, -1)));
    }
  }
  vector<size_t> prefix_product(dev_types_count.size() + 1);
  for (size_t i = 0; i <= dev_types_count.size(); ++i) {
    prefix_product[i] = i == 0 ? 1 : prefix_product[i - 1] * (dev_types_count[i - 1] + 1);
  }
  // num of device_i in S  == S / prefix_product[i] % (dev_types_count[i] + 1);

  for (size_t u = 0; u < dev_types_count.size(); ++u) {
      h[0][0][u] = 0;
  }
  double res = 1e60;
  optional<tuple<size_t, size_t, size_t>> res_index_opt = nullopt;
  for (size_t i = 0; i < num_layers; ++i) {
#if SCHEDULE_DEBUG
    cerr << "Finish progress " << i << " \\ " << num_layers << endl;
#endif
    for (size_t S = 0; S < mask; ++S) {
      for (size_t u = 0; u < dev_types_count.size(); ++u) {
        if (h[i][S][u] > 1e59) {
          continue;
        }
        if (S / prefix_product[u] % (dev_types_count[u] + 1) == dev_types_count[u]) {
          // u in fully used in S
          continue;
        }
        for (size_t j = i + 1; j <= num_layers; ++j) {
          if (!is_layers_fit(devices, u, ctx.layers, i + 1, j,
                             ctx.parameters_in,
                             ctx.dtype_size, ctx.batch_size,
                             ctx.data_buffers_in, ctx.data_buffers_out)) {
            continue;
          }
          // assign layers [i + 1, j] to node u
          double computation_time = compute_time(devices[u].get().t_comp, i + 1, j);
          // cerr << "Current Node is " << u << ", comp time is " << computation_time << endl;
          if (j == num_layers) {
            double cost = max(h[i][S][u], computation_time);
            if (cost < res) {
              res = cost;
              res_index_opt = {i, S, u};
            }
          }
          for (size_t v = 0; v < dev_types_count.size(); ++v) {
            if (S / prefix_product[v] % (dev_types_count[v] + 1) == dev_types_count[v]) {
              // v in fully used in S
              continue;
            }
            double communication_time = comm_time(ctx.layers, j, ctx.dtype_size, ctx.batch_size, devices, u, v);
            // cerr << "Communication time is " << communication_time << endl;
            double cost = max(h[i][S][u], max(computation_time, communication_time));
            // S + prefix_product[u] => S U {u}
            if (cost < h[j][S + prefix_product[u]][v]) {
              h[j][S + prefix_product[u]][v] = cost;
              parent[j][S + prefix_product[u]][v] = make_pair(i, u);
            }
          }
        }
      }
    }
  }

  if (!res_index_opt.has_value()) {
    return;
  }
  tuple<size_t, size_t, size_t> res_index = res_index_opt.value();

#if SCHEDULE_DEBUG
  cerr << "Minimum time : " << res << endl;
#endif
  size_t layer, S, u;
  tie(layer, S, u) = res_index;

  // calculate the selected nodes
  size_t stage_num = 0;
  while (layer > 0) {
    size_t last_l, last_u;
    stage_num++;
    assert(parent[layer][S][u].first >= 0);
    assert(parent[layer][S][u].second >= 0);
    tie(last_l, last_u) = parent[layer][S][u];
    layer = last_l;
    S -= prefix_product[last_u];
    u = last_u;
  }

  tie(layer, S, u) = res_index;
#if SCHEDULE_DEBUG
  cerr << "layer [" << layer + 1 << ", " << num_layers << "] used by device type " << u << ", stage " << stage_num << endl;
#endif
  dev_type_sched.push_back({
    .dev_type_idx = u,
    .layer_l = layer + 1,
    .layer_r = num_layers,
  });
  while (layer > 0) {
    size_t last_l, last_u;
    stage_num--;
    assert(parent[layer][S][u].first >= 0);
    assert(parent[layer][S][u].second >= 0);
    tie(last_l, last_u) = parent[layer][S][u];
    dev_type_sched.push_back({
      .dev_type_idx = last_u,
      .layer_l = last_l + 1,
      .layer_r = layer,
    });
#if SCHEDULE_DEBUG
    cerr << "layer [" << last_l + 1 << ", " << layer << "] used by device type " << last_u << ", stage " << stage_num << endl;
#endif
    layer = last_l;
    S -= prefix_product[last_u];
    u = last_u;
  }
  sort(dev_type_sched.begin(), dev_type_sched.end(), sort_dev_type_sched_stage);
}

void schedule_types_to_hosts(vector<host_sched_stage> &host_sched,
                             const vector<dev_type_sched_stage> &dev_type_sched,
                             const vector<device_type> &dev_types,
                             const map<string, vector<string>> &dev_type_hosts) {
  map<string, size_t> dev_type_hosts_idx;
  for (auto& s : dev_type_sched) {
    string dev_type_name = dev_types[s.dev_type_idx].name;
    size_t host_idx = 0;
    if (dev_type_hosts_idx.count(dev_type_name) > 0) {
      host_idx = dev_type_hosts_idx[dev_type_name];
    }
    string host = dev_type_hosts.at(dev_type_name)[host_idx];
    dev_type_hosts_idx[dev_type_name] = host_idx + 1;
    host_sched.push_back({
      .host = host,
      .layer_l = s.layer_l,
      .layer_r = s.layer_r,
    });
  }
}
