import os
import json
import numpy as np
import datasets
import copy
import argparse

PERCENTAGE_TO_KEEP = 0.5


def sample_tests(example, idx=None, seed=None):
    """Keep a random PERCENTAGE_TO_KEEP of the hidden tests of one problem.

    With seed=None the tests are drawn from the global NumPy stream (seeded once in main), as in
    SDPO. Otherwise each problem uses its own generator derived from (seed, idx).
    """
    tests = json.loads(example["tests"])
    inputs = tests["inputs"]
    outputs = tests["outputs"]

    num_tests = len(inputs)
    keep_count = max(1, int(num_tests * PERCENTAGE_TO_KEEP))
    if seed is None:
        keep_indices = np.sort(
            np.random.choice(num_tests, size=keep_count, replace=False)
        )
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, idx]))
        keep_indices = np.sort(
            rng.choice(num_tests, size=keep_count, replace=False)
        )

    reduced_inputs = [inputs[i] for i in keep_indices]
    reduced_outputs = [outputs[i] for i in keep_indices]

    reduced_tests = copy.deepcopy(tests)
    reduced_tests["inputs"] = reduced_inputs
    reduced_tests["outputs"] = reduced_outputs

    example["tests"] = json.dumps(reduced_tests)
    return example


def main(data_path, output_dir, validation_num_samples=None, seed=0):
    """Write test.json (validation problems, all tests) and train.json (all problems, 50% of tests).

    Without validation_num_samples (LiveCodeBench v6 in the paper), every problem is used for
    validation and the training tests are sampled from one global NumPy stream seeded with `seed`,
    exactly as in SDPO. With validation_num_samples (LCB-Large in the paper), the validation
    problems are sampled with `seed`, and the training tests of each problem are sampled with a
    per-problem generator derived from (seed, problem index).
    """
    if validation_num_samples is None:
        np.random.seed(seed)
    ds_train = datasets.load_dataset("json", data_files=data_path, split="train")

    ds_validation = ds_train
    if validation_num_samples is not None:
        if not 1 <= validation_num_samples <= len(ds_train):
            raise ValueError(
                "validation_num_samples must be between 1 and "
                f"{len(ds_train)}, got {validation_num_samples}"
            )
        rng = np.random.default_rng(seed)
        validation_indices = np.sort(
            rng.choice(len(ds_train), size=validation_num_samples, replace=False)
        )
        ds_validation = ds_train.select(validation_indices)

    # Save the validation problems with all hidden tests.
    test_file = os.path.join(output_dir, "test.json")
    ds_validation.to_json(test_file)

    # Keep every training problem, but expose only 50% of its hidden tests.
    if validation_num_samples is None:
        ds_train_reduced = ds_train.map(sample_tests)
    else:
        ds_train_reduced = ds_train.map(
            sample_tests,
            with_indices=True,
            fn_kwargs={"seed": seed},
        )

    # Save reduced dataset as train.json
    train_file = os.path.join(output_dir, "train.json")
    ds_train_reduced.to_json(train_file)

    # check counts
    orig_counts = []
    reduced_counts = []
    for orig_item, new_item in zip(ds_train, ds_train_reduced):
        orig_tests = json.loads(orig_item["tests"]) if isinstance(orig_item["tests"], str) else orig_item["tests"]
        new_tests = json.loads(new_item["tests"]) if isinstance(new_item["tests"], str) else new_item["tests"]

        orig_counts.append(len(orig_tests["inputs"]))
        reduced_counts.append(len(new_tests["inputs"]))

    print(
        f"Saved validation dataset with all tests to: {test_file} "
        f"({len(ds_validation)} of {len(ds_train)} problems, seed={seed})"
    )
    print(f"Saved reduced dataset (train set) to: {train_file}")
    print(f"Share of tests kept in train set: {PERCENTAGE_TO_KEEP:.2f}")
    print(f"Mean tests per item: {np.mean(orig_counts):.2f} → {np.mean(reduced_counts):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the training set JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=None,
        help="Randomly select this many validation problems; default keeps all problems"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for validation undersampling and training-test sampling"
    )
    args = parser.parse_args()
    main(
        args.json_path,
        args.output_dir,
        validation_num_samples=args.validation_num_samples,
        seed=args.seed,
    )
