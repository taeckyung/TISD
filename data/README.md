# Data preparation

Run all commands from the repository root with the repository on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Each dataset directory ends up with `train.json`/`test.json` and the `train.parquet`/`test.parquet` files read by the trainer (`data.train_files` and `data.val_files` in `verl/trainer/config/user.yaml`).

## LiveCodeBench

`load_dataset.py` downloads `livecodebench/code_generation_lite` (revision `refs/pr/6`). `code_generation_lite-v6` keeps problems from contests between 2025-02-01 and 2025-04-30 (LCB-Small in the paper); `code_generation_lite-v1-v6` keeps all problems from contests before 2025-05-01 (LCB-Large).

`split_tests.py` writes
- `test.json`: validation problems with all hidden tests (all problems by default, or `--validation_num_samples` random problems), and
- `train.json`: every problem with a random 50% of its hidden tests, which the verifier uses during training. Without `--validation_num_samples`, the tests are sampled as in SDPO (one NumPy stream seeded with `--seed`, default 0); with it, each problem uses its own generator derived from the seed and the problem index. The commands below reproduce the splits used in the paper.

```bash
# LiveCodeBench v6
python data/load_dataset.py --dataset_name livecodebench/code_generation_lite-v6 --output_path datasets/lcb_v6.json
python data/split_tests.py --json_path datasets/lcb_v6.json --output_dir datasets/lcb_v6
python data/preprocess.py --data_source datasets/lcb_v6

# LiveCodeBench v1-v6 (LCB-Large)
python data/load_dataset.py --dataset_name livecodebench/code_generation_lite-v1-v6 --output_path datasets/lcb_v1_v6.json
python data/split_tests.py --json_path datasets/lcb_v1_v6.json --output_dir datasets/lcb_v1_v6 --validation_num_samples 131 --seed 0
python data/preprocess.py --data_source datasets/lcb_v1_v6
```

## SciKnowEval L3

The paper uses the SDPO train/test splits of the four L3 multiple-choice domains, included in `datasets/sciknoweval/<domain>/{train,test}.json`. Convert them to parquet:

```bash
for d in biology chemistry material physics; do
    python data/preprocess.py --data_source datasets/sciknoweval/$d
done
```

`python data/load_dataset.py --dataset_name <Biology|Chemistry|Material|Physics> --output_path <file>.json` re-downloads the full L3 multiple-choice set of a domain from `hicai-zju/SciKnowEval` (the `datasets/sciknoweval/<domain>.json` files).
