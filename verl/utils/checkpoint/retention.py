"""Keep periodic actor weights while retaining only the latest resume state."""
from pathlib import Path
import re
import shutil


def prune_previous_training_state(root, latest_step: int):
    """Call only after a complete synchronous save and latest-pointer update.

    Preserve actor shards, tokenizer/config and optional HF/LoRA weights. Delete
    only known training-state paths in older step directories of this run.
    """
    root = Path(root)
    if not (root / f"global_step_{latest_step}" / "data.pt").is_file():
        raise ValueError("Latest checkpoint must include dataloader state before pruning")
    for directory in root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", directory.name)
        if not match or int(match[1]) >= latest_step or directory.is_symlink() or not directory.is_dir():
            continue
        (directory / "data.pt").unlink(missing_ok=True)
        actor = directory / "actor"
        if actor.is_dir() and not actor.is_symlink():
            for pattern in ("optim_world_size_*_rank_*.pt", "extra_state_world_size_*_rank_*.pt"):
                for path in actor.glob(pattern):
                    path.unlink()
            teacher = actor / "ema_teacher"
            if teacher.is_symlink():
                teacher.unlink()
            elif teacher.is_dir():
                shutil.rmtree(teacher)
        critic = directory / "critic"
        if critic.is_symlink():
            critic.unlink()
        elif critic.is_dir():
            shutil.rmtree(critic)
