# TISD: On-Policy Self-Distillation with Trajectory Intervention

Taeckyung Lee, Rinat Amankos, Jeonghye Kim, Hyungjun Yoon, Woogyeol Jin, Sung-Ju Lee

KAIST · Contact: {taeckyung, profsj}@kaist.ac.kr

[Project page](https://miil.kaist.ac.kr/projects/tisd/) | [Paper](https://miil.kaist.ac.kr/projects/tisd/assets/TISD.pdf)

This repository contains the code for **Trajectory-Intervention Self-Distillation (TISD)**.

On-policy self-distillation (OPSD) distills a privileged-context teacher (the same model conditioned on a successful sibling rollout or on environment feedback) into the student, but it only evaluates teacher targets along student-sampled rollouts. TISD uses teacher–student disagreement to propose a trajectory branch instead of only a local correction. For every failed rollout, TISD

1. **branches**: finds the position of peak teacher–student divergence and forces the teacher's preferred token there,
2. **regenerates**: returns suffix generation to the student, and
3. **distills**: trains on the full retained trajectory toward the privileged-context teacher with the SDPO objective.

The implementation builds on [SDPO](https://github.com/lasgroup/SDPO) and [verl](https://github.com/volcengine/verl). It also includes the baselines used in the paper: GRPO, SDPO, and TRD (teacher-refined trajectories).

## Installation

The recorded environment is Linux x86_64 with Python 3.12, CUDA 12, PyTorch 2.9.0, vLLM 0.12.0, and FlashAttention 2.8.3. Install [uv](https://docs.astral.sh/uv/) and run:

```bash
bash setup_uv_env.sh              # creates .venv-uv/ (override with UV_VENV=/path/to/env)
source .venv-uv/bin/activate
```

`requirements.txt` is the locked dependency set; `requirements.in` lists the top-level requirements it was compiled from. The launchers default to 4 GPUs per run; the paper's LiveCodeBench runs used four NVIDIA B200 GPUs.

## Data

All commands run from the repository root.

**LiveCodeBench v6** (coding with rich execution feedback). Validation uses every problem with all hidden tests, and training exposes 50% of the tests of each problem:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python data/load_dataset.py --dataset_name livecodebench/code_generation_lite-v6 --output_path datasets/lcb_v6.json
python data/split_tests.py --json_path datasets/lcb_v6.json --output_dir datasets/lcb_v6
python data/preprocess.py --data_source datasets/lcb_v6
```

**LCB-Large** (LiveCodeBench v1–v6 before 2025-05-01, Figure 6), with 131 randomly selected validation problems:

```bash
python data/load_dataset.py --dataset_name livecodebench/code_generation_lite-v1-v6 --output_path datasets/lcb_v1_v6.json
python data/split_tests.py --json_path datasets/lcb_v1_v6.json --output_dir datasets/lcb_v1_v6 --validation_num_samples 131 --seed 0
python data/preprocess.py --data_source datasets/lcb_v1_v6
```

**SciKnowEval L3** (science without rich feedback). The train/test splits used in the paper (the SDPO splits) are included in `datasets/sciknoweval/<domain>/`. Convert them to parquet:

```bash
for d in biology chemistry material physics; do python data/preprocess.py --data_source datasets/sciknoweval/$d; done
```

See [data/README.md](data/README.md) for details.

## Training

Coding (LiveCodeBench v6, with execution feedback):

```bash
bash experiments/lcb/tisd.sh
```

Science (SciKnowEval L3; runs biology, chemistry, material, and physics in sequence):

```bash
bash experiments/science/tisd.sh
DOMAINS=biology bash experiments/science/tisd.sh   # a single domain
```

The baselines use the same interface: replace `tisd.sh` with `sdpo.sh`, `trd.sh`, or `grpo.sh`. Set `MODEL_PATH` to change the model (default `Qwen/Qwen3-8B`) and `N_GPUS_PER_NODE` to change the number of GPUs (default 4). Extra arguments are passed to Hydra as config overrides.

## Code map

| File | Content |
|---|---|
| `verl/trainer/ppo/ray_trainer.py` | TISD pipeline (`_build_tisd_batch`): privileged teacher context, branch selection, suffix regeneration, and batch assembly; TRD (`_build_trd_batch`); SDPO (`_maybe_build_self_distillation_batch`) |
| `verl/workers/actor/dp_actor.py` | `compute_branch_kl` (per-token divergence and top-1 tokens) and the distillation update |
| `verl/trainer/ppo/core_algos.py` | `compute_self_distillation_loss` and the top-k divergences |
| `verl/experimental/agent_loop/branch_continuation_agent_loop.py` | Generation from a token prefix (TISD suffix, TRD refinement) |
| `verl/trainer/ppo/trd_utils.py` | TRD refinement prompts |
| `verl/trainer/config/{tisd,sdpo,trd,baseline_grpo}.yaml` | Method configurations |
| `verl/utils/reward_score/feedback/` | Verifiers with execution feedback (from SDPO) |

## Citation

```bibtex
@misc{lee2026tisd,
  title  = {{TISD}: On-Policy Self-Distillation with Trajectory Intervention},
  author = {Lee, Taeckyung and Amankos, Rinat and Kim, Jeonghye and Yoon, Hyungjun and Jin, Woogyeol and Lee, Sung-Ju},
  year   = {2026},
  url    = {https://miil.kaist.ac.kr/projects/tisd/assets/TISD.pdf}
}
```

## Acknowledgements and license

This code is derived from [SDPO](https://github.com/lasgroup/SDPO) and [verl](https://github.com/volcengine/verl) and is released under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The SciKnowEval splits in `datasets/sciknoweval/` are redistributed from SDPO; SciKnowEval and LiveCodeBench remain subject to their original licenses.
