#!/bin/bash
# Launch verl.trainer.main_ppo from the project root.
# Usage: training/verl_training.sh <experiment_name> <config_name> <data_path> [hydra overrides...]
# <data_path> is relative to the project root and must contain train.parquet and test.parquet.
set -euo pipefail
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
# deep_gemm's prebuilt kernels are ABI-incompatible with torch 2.9.0 and are only used for FP8 models.
# Worker actors also receive this setting through verl/trainer/constants_ppo.py.
export VLLM_USE_DEEP_GEMM=0
export PYTHONUNBUFFERED=1
ulimit -c 0

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <experiment_name> <config_name> <data_path> [hydra overrides...]"
    exit 1
fi
export EXPERIMENT=$1
CONFIG_NAME=$2
export TASK=$3
shift 3

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
echo "Arguments: $*"

python -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" "$@"
