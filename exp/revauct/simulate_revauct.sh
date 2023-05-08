#!/bin/bash
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${THIS_DIR}/../../"

# Evaluate multiple generated networks
NETWORK_PERMUTATIONS=100
# We realy only need 1 iteration since it's sufficient to simply generate/test more networks...
ITERATIONS_PER_NETWORK=1

# It's simpler to just generate large networks and then only consider a subset of devices below...
WORLD_SIZE=128

# Should be <= WORLD_SIZE
DEVS_CONSIDERED=${WORLD_SIZE}

MODELS_FILE="${THIS_DIR}/models.yml"
DEV_TYPES_FILE="${THIS_DIR}/device_types.yml"

MODEL='google/vit-base-patch16-224'
# MODEL='google/vit-large-patch16-224'
# MODEL='google/vit-huge-patch14-224-in21k'

SCHEDULER='latency_ordered'
# SCHEDULER='throughput_ordered'
# SCHEDULER='greedy_host_count'

OPTIONAL_ARGS=(
  # --filter-bids-chunk 4
  # --filter-bids-largest
  # --no-strict-order
  # --strict-first
  # --strict-last
)

for ((j=0; j<NETWORK_PERMUTATIONS; j++)); do
  DEVS_FILE="${THIS_DIR}/networks/${WORLD_SIZE}-FC/devices-${WORLD_SIZE}-${j}.yml"
  SCHED_DEV_NEIGHBORS_FILE="${THIS_DIR}/networks/${WORLD_SIZE}-FC/device_neighbors_world-${WORLD_SIZE}-${j}.yml"
  for ((i=0; i<ITERATIONS_PER_NETWORK; i++)); do
    CMD=(
      python "${THIS_DIR}/simulate_revauct.py"
      -sm "${MODELS_FILE}"
      -sdt "${DEV_TYPES_FILE}"
      -sd "${DEVS_FILE}"
      -sdnw "${SCHED_DEV_NEIGHBORS_FILE}"
      -m "${MODEL}"
      -s "${SCHEDULER}"
      -d "${DEVS_CONSIDERED}"
      "${OPTIONAL_ARGS[@]}"
    )
    log="simulate_revauct-${j}-${i}.log"
    echo "${CMD[*]}" | tee "${log}"
    time "${CMD[@]}" 2>&1 | tee -a "${log}" || exit $?
  done
done
