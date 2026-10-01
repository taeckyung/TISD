#!/bin/bash
# TISD on SciKnowEval L3 without rich environment feedback (paper Table 2; Table 12).
#
# Each domain runs for TIME_LIMIT (default 10h). Table 2 reports the best validation Avg@128 within
# the first 200 steps ("Steps") and within 10 hours ("Hours") of the same run.
#
# Usage: bash experiments/science/tisd.sh [hydra overrides...]
#   DOMAINS="biology physics" bash experiments/science/tisd.sh

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DOMAINS=${DOMAINS:-"biology chemistry material physics"}
TIME_LIMIT=${TIME_LIMIT:-10h}
RUN_ID=$(date +%Y%m%d-%H%M%S)

for DOMAIN in $DOMAINS; do
    DATA_PATH="datasets/sciknoweval/${DOMAIN}"
    run_training "TISD-sciknoweval-${DOMAIN}-${MODEL_NAME}-${RUN_ID}" tisd "$DATA_PATH" \
        $(common_args) \
        trainer.group_name=TISD-science \
        trainer.val_before_train=True \
        data.train_batch_size=32 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n=128 \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        algorithm.rollout_correction.rollout_is=token \
        actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
        actor_rollout_ref.actor.self_distillation.alpha=0.0 \
        actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
        actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
        actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
        actor_rollout_ref.actor.self_distillation.branch_alpha=1.0 \
        "$@"
done
