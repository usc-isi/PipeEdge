#!/bin/bash
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

NETWORK_SIZE=128
PERMUTATIONS=100
CONNECTIVITY="1.0"

BANDWIDTH_MIN=5
BANDWIDTH_MAX=1000
BANDWIDTH_MEAN=500
BANDWIDTH_STDEV=200

DEV_TYPES_FILE="${THIS_DIR}/device_types.yml"

CMD=(
  python "${THIS_DIR}/generate_networks.py"
  -s "${NETWORK_SIZE}"
  -p "${PERMUTATIONS}"
  -c "${CONNECTIVITY}"
  -b "${BANDWIDTH_MIN}"
  -B "${BANDWIDTH_MAX}"
  -bm "${BANDWIDTH_MEAN}"
  -bs "${BANDWIDTH_STDEV}"
  -dt "${DEV_TYPES_FILE}"
)
echo "${CMD[*]}"
"${CMD[@]}"
