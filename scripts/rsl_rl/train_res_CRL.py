# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continual-RL driver for the residual policy method.

Runs ``train_res.py`` once per task in a fixed sequence. After each task the driver
takes the ``ComposedActor`` that ``train_res.py`` saves (frozen base + trained
residual, stored under the ``"actor"`` key of ``complete_model_with_metadata.pth``)
and feeds it back as the ``--original_policy_path`` (frozen base) for the *next*
task. This way the base policy actually adapts across the sequence instead of every
task learning a residual on top of the same fixed flat-walking policy.

The first task uses the default base configured in ``ResCfg.original_policy_path``
(unless ``--init_base`` is given here).

Usage (extra CLI args after the known ones are forwarded to every train_res.py call):

    python scripts/rsl_rl/train_res_CRL.py --headless --num_envs 4096

    # start from a specific base checkpoint instead of the ResCfg default:
    python scripts/rsl_rl/train_res_CRL.py --init_base /path/to/base.pth --headless
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# Task sequence (mirrors the ordering used by train_moe_CRL.py). The first task is
# trained from the default/initial base; each subsequent task is trained on top of
# the previous task's composed (base + residual) policy.
TRAIN_TASKS = [
    # "Less-AnymalC-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Direct-v1",
    "Less-AnymalC-Rough-Walking-Direct-v1",
    "Less-Leg-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Rough-Direct-v1",
    "Less-Leg-Rough-Walking-Direct-v1",
]

# train_res.py writes its saved policy here (experiment_name depends on the task):
#   logs/rsl_rl/<experiment_name>/residual/complete_model_with_metadata.pth
_MODEL_GLOB = "logs/rsl_rl/*/residual/complete_model_with_metadata.pth"


def _latest_model_since(t_start: float) -> str | None:
    """Return the complete_model_with_metadata.pth written after ``t_start`` (newest)."""
    candidates = [p for p in glob.glob(_MODEL_GLOB) if os.path.getmtime(p) >= t_start]
    if not candidates:
        return None
    return os.path.abspath(max(candidates, key=os.path.getmtime))


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual residual training driver.")
    parser.add_argument(
        "--init_base",
        type=str,
        default=None,
        help="Optional initial base-policy checkpoint for the first task. "
        "If omitted, ResCfg.original_policy_path is used.",
    )
    args, forward_args = parser.parse_known_args()

    train_script = Path(__file__).with_name("train_res.py")
    if not train_script.is_file():
        raise FileNotFoundError(str(train_script))

    base_path = args.init_base  # None => train_res.py uses the ResCfg default
    completed = []

    for task_name in TRAIN_TASKS:
        command = [sys.executable, str(train_script), "--task", task_name]
        if base_path is not None:
            command += ["--original_policy_path", base_path]
        command += forward_args

        print(f"\n[RES-CRL] ==============================================")
        print(f"[RES-CRL] Training task {len(completed) + 1}/{len(TRAIN_TASKS)}: {task_name}")
        print(f"[RES-CRL] Base policy: {base_path if base_path else '(ResCfg default)'}")
        print(f"[RES-CRL] ==============================================\n", flush=True)

        t_start = time.time()
        subprocess.run(command, check=True)

        produced = _latest_model_since(t_start)
        if produced is None:
            raise RuntimeError(
                f"No complete_model_with_metadata.pth was produced for task '{task_name}'. "
                f"Expected one under {_MODEL_GLOB}."
            )

        # The saved 'actor' is a ComposedActor(base, residual) -> next task's frozen base.
        base_path = produced
        completed.append(task_name)
        print(f"[RES-CRL] Completed {task_name}; next base = {base_path}", flush=True)

    print(f"\n[RES-CRL] Finished all {len(completed)} tasks: {completed}")


if __name__ == "__main__":
    main()
