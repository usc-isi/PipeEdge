#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <getopt.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "schedule.h"
#include "yaml_types.h"

using namespace std;

static void load_model(size_t *parameters_in, vector<model_layer> &layers,
                       const YAML::Node &models_yml, string model_name) {
  if (!models_yml[model_name]) {
    cerr << "Model not found: " << model_name << endl;
    exit(1);
  }
  yaml_model model = models_yml[model_name].as<yaml_model>();
  // verify data consistency
  if (model.parameters_out.size() < model.layers) {
    cerr << "Warning: Model parameters_out length " << model.parameters_out.size() << " < " << model.layers
         << ": block will be repeated" << endl;
  } else if (model.parameters_out.size() > model.layers) {
    cerr << "Model parameters_out length " << model.parameters_out.size() << " > " << model.layers << endl;
    exit(1);
  }
  if (model.mem_MB.size() != model.layers) {
    cerr << "Model mem_MB length " << model.mem_MB.size() << " != " << model.layers << endl;
    exit(1);
  }
  for (size_t i = 0; i < model.layers; i++) {
    assert(model.mem_MB[i] >= 0);
  }
  // get primitive field(s)
  *parameters_in = model.parameters_in;
  // struct conversion
  for (size_t i = 0; i < model.layers; i++) {
    layers.push_back({
      .parameters_out = model.parameters_out[i % model.parameters_out.size()],
      .mem_MB = model.mem_MB[i],
    });
  }
}

static void load_device_types(vector<device_type> &device_types,
                              const YAML::Node &device_types_yml, string model_name, string dtype, size_t batch_size) {
  if (!device_types_yml.IsMap()) {
    cerr << "No device types found" << endl;
    exit(1);
  }
  map<string, yaml_device_type> yaml_dev_types = device_types_yml.as<map<string, yaml_device_type>>();
  for (auto& [dev_name, ydev] : yaml_dev_types) {
    if (ydev.model_profiles.count(model_name) == 0) {
      // no profile data for the model we're interested in
      cerr << "Warning: Device type " << dev_name << " doesn't support model: type will be skipped" << endl;
      continue;
    }
    // look for matching profile for the model
    vector<double>* time_s_opt = nullptr;
    for (auto& ymp : ydev.model_profiles[model_name]) {
      if (ymp.dtype == dtype && ymp.batch_size == batch_size) {
        time_s_opt = &ymp.time_s;
      }
    }
    if (!time_s_opt) {
      // no batch size profile data for the model we're interested in
      cerr << "Warning: Device type " << dev_name << " doesn't have matching profile: type will be skipped" << endl;
      continue;
    }
    // verify data consistency
    assert(ydev.mem_MB >= 0);
    assert(ydev.bw_Mbps >= 0);
#ifndef NDEBUG
    for (double val : *time_s_opt) {
      assert(val >= 0);
    }
#endif // NDEBUG
    // struct conversion
    device_types.push_back({
      .name = dev_name,
      .mem_MB = ydev.mem_MB,
      .bw_Mbps = ydev.bw_Mbps,
      .t_comp = *time_s_opt,
    });
  }
}

static void load_devices(map<string, vector<string>> &dev_type_hosts, const YAML::Node &devices_yml) {
  if (!devices_yml.IsMap()) {
    cerr << "No devices found" << endl;
    exit(1);
  }
  map<string, vector<string>> ydevs = devices_yml.as<map<string, vector<string>>>();
  dev_type_hosts.insert(ydevs.begin(), ydevs.end());
}

static void load_files(schedule_ctx &ctx, string models_file, string device_types_file, string devices_file,
                       string model_name, string dtype) {
  // Get model info
  YAML::Node models_yml = YAML::LoadFile(models_file);
  load_model(&ctx.parameters_in, ctx.layers, models_yml, model_name);

  // Get device type info
  YAML::Node device_types_yml = YAML::LoadFile(device_types_file);
  load_device_types(ctx.dev_types, device_types_yml, model_name, dtype, ctx.batch_size);

  // Verify model info and device type info are compatible
  for (const auto& dev : ctx.dev_types) {
    if (dev.t_comp.size() != ctx.layers.size()) {
      cerr << "Device: " << dev.name << ": model layer size (" << dev.t_comp.size() << ") != device time size ("
           << ctx.layers.size() << ")" << endl;
      exit(1);
    }
  }

  // Get device info
  YAML::Node devices_yml = YAML::LoadFile(devices_file);
  load_devices(ctx.dev_type_hosts, devices_yml);
}

static void print_schedule(const vector<host_sched_stage> &host_sched) {
  // YAML: list of maps, where each map has key: host, value: size-2 list of start/end layers
  YAML::Emitter yml;
  yml << host_sched;
  cout << yml.c_str() << endl;
}

static const char short_options[] = "hd:b:i:o:m:M:T:D:";
static const struct option long_options[] = {
  {"help",              no_argument,       nullptr, 'h'},
  {"dtype",             required_argument, nullptr, 'd'},
  {"batch-size",        required_argument, nullptr, 'b'},
  {"buffers-in",        required_argument, nullptr, 'i'},
  {"buffers-out",       required_argument, nullptr, 'o'},
  {"model-name",        required_argument, nullptr, 'm'},
  {"models-file",       required_argument, nullptr, 'M'},
  {"dev-types-file",    required_argument, nullptr, 'T'},
  {"dev-file",          required_argument, nullptr, 'D'},
  {nullptr, 0, nullptr, 0}
};

// Number of data buffers: e.g., 1 for in-flight data exchanges, 1 for queues (p2p comm only)
#define DATA_BUFFERS_IN_DEFAULT 2
#define DATA_BUFFERS_OUT_DEFAULT 2

#define DTYPE_DEFAULT "torch.float32"
#define BATCH_SIZE_DEFAULT 8
#define MODEL_NAME_DEFAULT "google/vit-base-patch16-224"
#define MODELS_FILE_DEFAULT "models.yml"
#define DEV_TYPES_FILE_DEFAULT "device_types.yml"
#define DEV_FILE_DEFAULT "devices.yml"

[[noreturn]] static void print_usage(int exit_code) {
  auto& os = exit_code ? cerr : cout;
  os << "Usage: sched-pipeline [OPTION]..." << endl << endl
     << "Run the pipeline partition scheduling algorithm." << endl << endl
     << "Options:" << endl
     << "  -h, --help                 Print this message and exit" << endl
     << "  -d, --dtype=NAME           Data type (default=" << DTYPE_DEFAULT << ")" << endl
     << "  -b, --batch-size=N         Batch size (default=" << BATCH_SIZE_DEFAULT << ")" << endl
     << "  -i, --buffers-in=N         Inbound data buffers (default=" << DATA_BUFFERS_IN_DEFAULT << ")" << endl
     << "  -o, --buffers-out=N        Outbound data buffers (default=" << DATA_BUFFERS_OUT_DEFAULT << ")" << endl
     << "  -m, --model-name=NAME      Model name (default=" << MODEL_NAME_DEFAULT << ")" << endl
     << "  -M, --models-file=PATH     Models YAML file (default=" << MODELS_FILE_DEFAULT << ")" << endl
     << "  -T, --dev-types-file=PATH  Device types YAML file (default=" << DEV_TYPES_FILE_DEFAULT << ")" << endl
     << "  -D, --dev-file=PATH        Devices YAML file (default=" << DEV_FILE_DEFAULT << ")" << endl;
  exit(exit_code);
}

int main(int argc, char **argv) {
  schedule_ctx ctx;
  ctx.batch_size = BATCH_SIZE_DEFAULT;
  ctx.data_buffers_in = DATA_BUFFERS_IN_DEFAULT;
  ctx.data_buffers_out = DATA_BUFFERS_OUT_DEFAULT;
  string dtype = DTYPE_DEFAULT;
  string model_name = MODEL_NAME_DEFAULT;
  string models_file = MODELS_FILE_DEFAULT;
  string device_types_file = DEV_TYPES_FILE_DEFAULT;
  string devices_file = DEV_FILE_DEFAULT;
  int c;
  while ((c = getopt_long(argc, argv, short_options, long_options, nullptr)) != -1) {
    switch (c) {
      case 'h':
        print_usage(0);
      case 'd':
        dtype = optarg;
        break;
      case 'b':
        if (sscanf(optarg, "%zu", &ctx.batch_size) != 1) {
          cerr << "Failed to parse -b/--batch-size" << endl;
          print_usage(1);
        }
        break;
      case 'i':
        if (sscanf(optarg, "%zu", &ctx.data_buffers_in) != 1) {
          cerr << "Failed to parse -i/--buffers-in" << endl;
          print_usage(1);
        }
        break;
      case 'o':
        if (sscanf(optarg, "%zu", &ctx.data_buffers_out) != 1) {
          cerr << "Failed to parse -o/--buffers-out" << endl;
          print_usage(1);
        }
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'M':
        models_file = optarg;
        break;
      case 'T':
        device_types_file = optarg;
        break;
      case 'D':
        devices_file = optarg;
        break;
      case '?':
      default:
        print_usage(1);
    }
  }
  if (dtype == "torch.float32") {
    ctx.dtype_size = sizeof(float);
  } else {
    cerr << "Unsupported dtype: " << dtype << endl;
    print_usage(1);
  }

  // Process inputs into scheduling context
  load_files(ctx, models_file, device_types_file, devices_file, model_name, dtype);

  // Schedule layers to device types
  vector<dev_type_sched_stage> dev_type_sched;
  schedule_device_types(dev_type_sched, ctx);

  // Map device types to hosts
  vector<host_sched_stage> host_sched;
  schedule_types_to_hosts(host_sched, dev_type_sched, ctx.dev_types, ctx.dev_type_hosts);

  // Report schedule
  print_schedule(host_sched);

  return 0;
}
