#ifndef SCHEDULE
#define SCHEDULE

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct model_layer {
  size_t parameters_out; // P_j
  double mem_MB; // M_j
};

struct device_type {
  std::string name;
  double mem_MB;
  // Bandwidths could be considered pairwise between device instances instead, but it's more complex
  double bw_Mbps;
  std::vector<double> t_comp; // T_comp(l, u)
};

struct schedule_ctx {
  size_t parameters_in;
  std::vector<model_layer> layers;

  std::vector<device_type> dev_types;
  // key: device type name, value: list of hosts
  std::map<std::string, std::vector<std::string>> dev_type_hosts;

  size_t dtype_size;
  size_t batch_size;
  size_t data_buffers_in;
  size_t data_buffers_out;
};

struct dev_type_sched_stage {
  size_t dev_type_idx;
  size_t layer_l;
  size_t layer_r;
};

struct host_sched_stage {
  std::string host;
  size_t layer_l;
  size_t layer_r;
};

void schedule_device_types(std::vector<dev_type_sched_stage> &dev_type_sched, const schedule_ctx &ctx);

void schedule_types_to_hosts(std::vector<host_sched_stage> &host_sched,
                             const std::vector<dev_type_sched_stage> &dev_type_sched,
                             const std::vector<device_type> &dev_types,
                             const std::map<std::string, std::vector<std::string>> &dev_type_hosts);

#endif // SCHEDULE
