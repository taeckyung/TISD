#!/bin/bash
# Shared setup for the experiment launchers. Source this file; do not run it directly.
#
# Environment variables understood by every launcher:
#   MODEL_PATH       Hugging Face model id or local path (default: Qwen/Qwen3-8B)
#   N_GPUS_PER_NODE  GPUs used on this node (default: 4)
#   WANDB_PROJECT    W&B project (default: TISD); set WANDB_MODE=offline to log locally
#   TIME_LIMIT       Optional wall-clock cap per run (GNU timeout duration, e.g. 10h)
#   RESUME_FROM_PATH Optional checkpoint directory (.../global_step_N) to resume from
#   LOG_DIR, CHECKPOINT_DIR  Output locations (default: logs/ and checkpoints/ in the project)
# Extra command-line arguments are forwarded to Hydra as config overrides.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export WANDB_PROJECT=${WANDB_PROJECT:-"TISD"}
TIME_LIMIT=${TIME_LIMIT:-}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# Keep torch.compile / Triton caches inside the project (some systems mount /tmp noexec).
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"$PROJECT_ROOT/.cache/torchinductor"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$PROJECT_ROOT/.cache/triton"}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

MODEL_NAME=$(basename "$MODEL_PATH")

# Arguments shared by all runs: model, GPUs, and optional resume.
common_args() {
    local args="actor_rollout_ref.model.path=$MODEL_PATH trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
    if [[ -n "$RESUME_FROM_PATH" ]]; then
        args="$args trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME_FROM_PATH"
    fi
    echo "$args"
}

# run_training <experiment_name> <config_name> <data_path> [hydra overrides...]
# Runs one training job, optionally under TIME_LIMIT, then stops the local Ray cluster.
run_training() {
    local exp_name=$1
    local config_name=$2
    local data_path=$3
    shift 3
    echo "----------------------------------------------------------------"
    echo "Experiment: $exp_name"
    echo "Config:     $config_name"
    echo "Data:       $data_path"
    echo "Model:      $MODEL_PATH"
    [[ -n "$TIME_LIMIT" ]] && echo "Time limit: $TIME_LIMIT"
    echo "----------------------------------------------------------------"
    local status=0
    if [[ -n "$TIME_LIMIT" ]]; then
        # SIGINT lets the trainer shut down cleanly; reaching the limit is the expected outcome.
        timeout --signal=INT --kill-after=120 "$TIME_LIMIT" \
            bash "$PROJECT_ROOT/training/verl_training.sh" "$exp_name" "$config_name" "$data_path" "$@" || status=$?
        if [[ $status -eq 124 || $status -eq 137 ]]; then
            echo "Reached the time limit ($TIME_LIMIT)."
            status=0
        fi
        # Ray daemons outlive the timed process group; stop them before the next run.
        ray stop --force >/dev/null 2>&1 || true
    else
        bash "$PROJECT_ROOT/training/verl_training.sh" "$exp_name" "$config_name" "$data_path" "$@" || status=$?
    fi
    return $status
}
