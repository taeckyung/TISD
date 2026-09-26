#!/bin/bash
# GRPO baseline on LiveCodeBench v6 (paper Table 1, Figure 6; Table 11).
#
# Usage: bash experiments/lcb/grpo.sh [hydra overrides...]
#   MODEL_PATH=Qwen/Qwen3-14B bash experiments/lcb/grpo.sh
#   DATA_PATH=datasets/lcb_v1_v6 bash experiments/lcb/grpo.sh     # LCB-Large (Figure 6)
#   GRPO_MODE=on_policy bash experiments/lcb/grpo.sh              # mini-batch 32, lr 1e-5

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DATA_PATH=${DATA_PATH:-"datasets/lcb_v6"}
GRPO_MODE=${GRPO_MODE:-off_policy}
case "$GRPO_MODE" in
    off_policy) MINI_BATCH_SIZE=8; LR=1e-6 ;;
    on_policy) MINI_BATCH_SIZE=32; LR=1e-5 ;;
    *) echo "GRPO_MODE must be off_policy or on_policy, got: $GRPO_MODE" >&2; exit 1 ;;
esac
EXP_NAME="GRPO-${GRPO_MODE}-$(basename "$DATA_PATH")-${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"

run_training "$EXP_NAME" baseline_grpo "$DATA_PATH" \
    $(common_args) \
    trainer.group_name=GRPO-lcb \
    trainer.total_training_steps=80 \
    trainer.val_before_train=True \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    algorithm.rollout_correction.rollout_is=token \
    "$@"
