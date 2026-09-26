#!/bin/bash
# TISD on LiveCodeBench v6 with rich environment feedback (paper Table 1, Figures 6-8; Table 11).
#
# Usage: bash experiments/lcb/tisd.sh [hydra overrides...]
#   MODEL_PATH=Qwen/Qwen3-14B bash experiments/lcb/tisd.sh
#   DATA_PATH=datasets/lcb_v1_v6 bash experiments/lcb/tisd.sh     # LCB-Large (Figure 6)
# Ablations:
#   BRANCH_SELECTION=random bash experiments/lcb/tisd.sh           # Figure 7, random position
#   BRANCH_SELECTION=first_divergence bash experiments/lcb/tisd.sh # Figure 7, first divergence
#   BRANCH_TOKEN_SOURCE=student bash experiments/lcb/tisd.sh       # Figure 7, student own token
#   REGENERATION_ROUNDS=2 bash experiments/lcb/tisd.sh             # Figure 8 (2 or 3 rounds)

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DATA_PATH=${DATA_PATH:-"datasets/lcb_v6"}
BRANCH_SELECTION=${BRANCH_SELECTION:-argmax}
BRANCH_TOKEN_SOURCE=${BRANCH_TOKEN_SOURCE:-teacher}
REGENERATION_ROUNDS=${REGENERATION_ROUNDS:-1}

EXP_NAME="TISD-$(basename "$DATA_PATH")-${MODEL_NAME}-${BRANCH_SELECTION}-${BRANCH_TOKEN_SOURCE}_token-rounds${REGENERATION_ROUNDS}-$(date +%Y%m%d-%H%M%S)"

run_training "$EXP_NAME" tisd "$DATA_PATH" \
    $(common_args) \
    trainer.group_name=TISD-lcb \
    trainer.total_training_steps=80 \
    trainer.val_before_train=True \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    algorithm.rollout_correction.rollout_is=token \
    actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
    actor_rollout_ref.actor.self_distillation.alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.branch_alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.branch_selection="$BRANCH_SELECTION" \
    actor_rollout_ref.actor.self_distillation.branch_token_source="$BRANCH_TOKEN_SOURCE" \
    actor_rollout_ref.actor.self_distillation.regeneration_rounds="$REGENERATION_ROUNDS" \
    "$@"
