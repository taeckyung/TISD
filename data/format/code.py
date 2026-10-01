from datasets import Dataset
from datetime import datetime

from data.format.prompts import CODE_PROMPT
from data.format.utils import cast_large_strings
from data.utils.livecodebench import load_livecodebench


def load_code(dataset_name: str) -> Dataset:
    assert dataset_name in ["livecodebench/code_generation_lite-v6", "livecodebench/code_generation_lite-v1-v6"]

    if dataset_name == "livecodebench/code_generation_lite-v6":
        ds = load_livecodebench(dataset_split="test", until=datetime(2025, 5, 1))
        ds = cast_large_strings(ds, columns=list(ds.features.keys()))
        ds = ds.map(lambda ex: {"prompt": CODE_PROMPT.format(problem=ex["problem"])})
    elif dataset_name == "livecodebench/code_generation_lite-v1-v6":
        ds = load_livecodebench(dataset_split="all", until=datetime(2025, 5, 1))
        ds = cast_large_strings(ds, columns=list(ds.features.keys()))
        ds = ds.map(lambda ex: {"prompt": CODE_PROMPT.format(problem=ex["problem"])})

    def check_description(ex):
        assert ex["description"].strip() != ""
    ds.map(check_description)

    return ds
