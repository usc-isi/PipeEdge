#ifndef SCHEDULER_YAML_TYPES
#define SCHEDULER_YAML_TYPES

#include <cstdint>
#include <map>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "schedule.h"

struct yaml_model {
  size_t layers;
  size_t parameters_in;
  std::vector<size_t> parameters_out;
  std::vector<double> mem_MB;
};

// There's currently only a "time_s" field, which makes this struct seem superfluous.
// However, this makes it easier to extend profile specifications, e.g., with power/energy metrics.
struct yaml_model_profile {
  // config field(s)
  std::string dtype;
  size_t batch_size;
  // result field(s)
  std::vector<double> time_s;
};

struct yaml_device_type {
  double mem_MB;
  double bw_Mbps;
  // key: model name
  std::map<std::string, std::vector<yaml_model_profile>> model_profiles;
};

typedef host_sched_stage yaml_schedule_stage;

namespace YAML {
  template<>
  struct convert<yaml_model> {
    static bool decode(const Node& node, yaml_model& rhs) {
      rhs.layers = node["layers"].as<size_t>();
      rhs.parameters_in = node["parameters_in"].as<size_t>();
      rhs.parameters_out = node["parameters_out"].as<std::vector<size_t>>();
      rhs.mem_MB = node["mem_MB"].as<std::vector<double>>();
      return true;
    }
  };

  template<>
  struct convert<yaml_model_profile> {
    static bool decode(const Node& node, yaml_model_profile& rhs) {
      rhs.dtype = node["dtype"].as<std::string>();
      rhs.batch_size = node["batch_size"].as<size_t>();
      rhs.time_s = node["time_s"].as<std::vector<double>>();
      return true;
    }
  };

  template<>
  struct convert<yaml_device_type> {
    static bool decode(const Node& node, yaml_device_type& rhs) {
      rhs.mem_MB = node["mem_MB"].as<double>();
      rhs.bw_Mbps = node["bw_Mbps"].as<double>();
      if (node["model_profiles"]) {
        auto model_profiles = node["model_profiles"];
        for (auto it = model_profiles.begin(); it != model_profiles.end(); it++) {
          auto key = it->first.as<std::string>();
          rhs.model_profiles[key] = it->second.as<std::vector<yaml_model_profile>>();
        }
      }
      return true;
    }
  };

  static Emitter& operator<<(Emitter& out, const yaml_schedule_stage& rhs) {
    out << BeginMap;
    out << Key << rhs.host;
    out << Value << Flow << BeginSeq << rhs.layer_l << rhs.layer_r << EndSeq;
    out << EndMap;
    return out;
  }
}

#endif // SCHEDULER_YAML_TYPES
