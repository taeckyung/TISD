#!/bin/bash
# GRPO baselines on SciKnowEval L3 (paper Table 2; Table 12).
#   GRPO_MODE=off_policy (default): mini-batch 8, lr 1e-6   ("+ GRPO")
#   GRPO_MODE=on_policy:            mini-batch 32, lr 1e-5  ("+ GRPO (on-policy)")
#
# Each domain runs for TIME_LIMIT (default 10h). Table 2 reports the best validation Avg@128 within
# the first 200 steps ("Steps") and within 10 hours ("Hours") of the same run.
#
# Usage: GRPO_MODE=on_policy DOMAINS=biology bash experiments/science/grpo.sh [hydra overrides...]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DOMAINS=${DOMAINS:-"biology chemistry material physics"}
TIME_LIMIT=${TIME_LIMIT:-10h}
GRPO_MODE=${GRPO_MODE:-off_policy}
case "$GRPO_MODE" in
    off_policy) MINI_BATCH_SIZE=8; LR=1e-6 ;;
    on_policy) MINI_BATCH_SIZE=32; LR=1e-5 ;;
    *) echo "GRPO_MODE must be off_policy or on_policy, got: $GRPO_MODE" >&2; exit 1 ;;
esac
RUN_ID=$(date +%Y%m%d-%H%M%S)

for DOMAIN in $DOMAINS; do
    DATA_PATH="datasets/sciknoweval/${DOMAIN}"
    run_training "GRPO-${GRPO_MODE}-sciknoweval-${DOMAIN}-${MODEL_NAME}-${RUN_ID}" baseline_grpo "$DATA_PATH" \
        $(common_args) \
        trainer.group_name=GRPO-science \
        trainer.val_before_train=True \
        data.train_batch_size=32 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n=128 \
        actor_rollout_ref.actor.optim.lr=$LR \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.01 \
        actor_rollout_ref.actor.optim.clip_grad=1.0 \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=2.0 \
        "$@"
done
