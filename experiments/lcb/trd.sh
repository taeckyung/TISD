#!/bin/bash
# TRD baseline on LiveCodeBench v6: the EMA teacher rewrites each rollout and the student distills the
# refined trajectory with full-vocabulary reverse KL (paper Table 1, Figure 6; Table 11).
#
# Usage: bash experiments/lcb/trd.sh [hydra overrides...]
#   MODEL_PATH=Qwen/Qwen3-14B bash experiments/lcb/trd.sh
#   DATA_PATH=datasets/lcb_v1_v6 bash experiments/lcb/trd.sh      # LCB-Large (Figure 6)

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DATA_PATH=${DATA_PATH:-"datasets/lcb_v6"}
EXP_NAME="TRD-$(basename "$DATA_PATH")-${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"

run_training "$EXP_NAME" trd "$DATA_PATH" \
    $(common_args) \
    trainer.group_name=TRD-lcb \
    trainer.total_training_steps=80 \
    trainer.val_before_train=True \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    algorithm.rollout_correction.rollout_is=token \
    actor_rollout_ref.actor.self_distillation.distillation_topk=null \
    actor_rollout_ref.actor.self_distillation.alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    "$@"
