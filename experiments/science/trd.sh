#!/bin/bash
# TRD on SciKnowEval L3 (paper Figure 4: training collapse of full teacher regeneration; Table 12).
#
# Usage: DOMAINS=biology bash experiments/science/trd.sh [hydra overrides...]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DOMAINS=${DOMAINS:-"biology"}
TIME_LIMIT=${TIME_LIMIT:-10h}
RUN_ID=$(date +%Y%m%d-%H%M%S)

for DOMAIN in $DOMAINS; do
    DATA_PATH="datasets/sciknoweval/${DOMAIN}"
    run_training "TRD-sciknoweval-${DOMAIN}-${MODEL_NAME}-${RUN_ID}" trd "$DATA_PATH" \
        $(common_args) \
        trainer.group_name=TRD-science \
        trainer.val_before_train=True \
        data.train_batch_size=32 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n=128 \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        algorithm.rollout_correction.rollout_is=token \
        actor_rollout_ref.actor.self_distillation.distillation_topk=null \
        actor_rollout_ref.actor.self_distillation.alpha=0.0 \
        actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
        actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
        actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
        "$@"
done
