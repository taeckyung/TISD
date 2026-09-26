# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Agent loop that continues generation from a caller-supplied token prefix.

Used by TISD to regenerate the suffix from ``plain prompt + response prefix + forced token`` and by
TRD to generate the refined trajectory from the refinement prompt. Unlike SingleTurnAgentLoop (which
chat-templates ``raw_prompt`` into a prompt bounded by ``rollout.prompt_length``), this loop generates
directly from ``branch_prompt_ids``, which can be longer than ``prompt_length``. The standard
``_agent_loop_postprocess`` still pads the *bookkeeping* prompt to ``prompt_length``, so we return the
original (short) student prompt as ``prompt_ids`` and only the new tokens as ``response_ids``. The
driver reconstructs the full trajectory from the prefix it already holds plus these tokens.

Per-sample ``max_tokens`` is set so that ``len(branch_prompt_ids) + continuation <= max_model_len``
and ``continuation <= rollout.response_length``.
"""
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("branch_continuation_agent")
class BranchContinuationAgentLoop(AgentLoopBase):
    """Continue generation from a raw token prefix (TISD suffix regeneration, TRD refinement)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        self.prompt_length = rollout_cfg.prompt_length
        self.response_length = rollout_cfg.response_length
        # max_model_len may live on the top-level config or the rollout config; fall back to a generous cap.
        self.max_model_len = (
            self.config.get("max_model_len", None)
            or rollout_cfg.get("max_model_len", None)
            or (self.prompt_length + self.response_length)
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        branch_prompt_ids = list(kwargs["branch_prompt_ids"])

        # Continuation budget: keep prompt + prefix + continuation within max_model_len, and never
        # exceed response_length (the postprocess response pad width).
        budget = min(self.response_length, max(1, int(self.max_model_len) - len(branch_prompt_ids)))
        sp = dict(sampling_params)
        sp["max_tokens"] = budget

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=branch_prompt_ids,
                sampling_params=sp,
                image_data=None,
                video_data=None,
            )

        response_ids = output.token_ids[:budget]
        response_mask = [1] * len(response_ids)

        # Bookkeeping prompt: the original (short) student prompt, left-truncated to prompt_length so
        # _agent_loop_postprocess can pad it to a uniform width. It is NOT used for generation.
        messages = list(kwargs["raw_prompt"])
        bookkeeping_ids = await self.apply_chat_template(messages, tools=[])
        if len(bookkeeping_ids) > self.prompt_length:
            bookkeeping_ids = bookkeeping_ids[-self.prompt_length :]

        return AgentLoopOutput(
            prompt_ids=bookkeeping_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=output.log_probs[:budget] if output.log_probs else None,
            num_turns=2,
            metrics=metrics,
        )
