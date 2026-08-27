# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Continual-learning driver for FAME.

Mirrors ``scripts/rsl_rl/train_moe_CRL.py``: it launches ``train_FAME.py`` once
per task, each in a fresh Isaac Sim process, and chains the end-of-task FAME
checkpoint into the next task's ``--checkpoint``. Because every task runs in its
own process, the IsaacLab ``SimulationContext`` is created and destroyed cleanly
each time — there is no in-process environment recreation.

Each task's TensorBoard logs still land under ``logs/task1/<experiment_name>/FAME``
(handled by ``train_FAME.py``); the chained checkpoints live under
``--ckpt_dir`` (default ``logs/skrl/FAME_CRL/checkpoints``).

Examples
--------
# Run the whole default sequence (args after are passed through to train_FAME.py)
python scripts/skrl/train_FAME_CRL.py --switch_steps 100000 --headless

# Override the task sequence
python scripts/skrl/train_FAME_CRL.py \\
    --tasks Less-AnymalC-Flat-Walking-Direct-v1,Less-Leg-Flat-Walking-Direct-v1 \\
    --switch_steps 100000 --headless

# Resume the sequence from task index 2, seeding it with a checkpoint
python scripts/skrl/train_FAME_CRL.py \\
    --start_idx 2 --checkpoint logs/skrl/FAME_CRL/checkpoints/task01_end \\
    --switch_steps 100000 --headless
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Authoritative task sequence (same order as scripts/rsl_rl/train_moe_CRL.py:
# its ORIGINAL_TASK first, then TRAIN_TASKS). Keep in sync with the active
# entries in train_FAME.py's DEFAULT_TASKS.
TASKS = [
    "Less-AnymalC-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Direct-v1",
    "Less-Leg-Flat-Walking-Direct-v1",
    "Less-AnymalC-Rough-Walking-Direct-v1",
    "Less-AnymalC-Jump-Rough-Direct-v1",
    "Less-Leg-Rough-Walking-Direct-v1",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the FAME continual-learning sequence, one task per subprocess.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task sequence override. Defaults to the hardcoded TASKS list.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Task index to start the sequence from (for resuming). Default 0.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="logs/skrl/FAME_CRL/checkpoints",
        help="Directory for the chained per-task checkpoints handed between subprocesses.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Initial FAME checkpoint to seed the --start_idx task (for resuming a sequence).",
    )
    # Everything else (e.g. --switch_steps, --num_envs, --headless, --seed) is
    # forwarded verbatim to each train_FAME.py invocation.
    driver_args, passthrough = parser.parse_known_args()

    tasks = (
        [t.strip() for t in driver_args.tasks.split(",") if t.strip()]
        if driver_args.tasks
        else list(TASKS)
    )
    if not tasks:
        raise SystemExit("[FAME-CRL] Empty task list.")
    if not (0 <= driver_args.start_idx < len(tasks)):
        raise SystemExit(f"[FAME-CRL] --start_idx {driver_args.start_idx} out of range for {len(tasks)} tasks.")

    train_script = Path(__file__).with_name("train_FAME.py")
    if not train_script.is_file():
        raise FileNotFoundError(str(train_script))

    ckpt_dir = Path(driver_args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tasks_arg = ",".join(tasks)

    print(f"[FAME-CRL] Task sequence ({len(tasks)} tasks): {tasks}")
    print(f"[FAME-CRL] Starting from index {driver_args.start_idx}")
    print(f"[FAME-CRL] Checkpoint dir: {ckpt_dir.resolve()}")

    for i in range(driver_args.start_idx, len(tasks)):
        task = tasks[i]
        end_ckpt = ckpt_dir / f"task{i:02d}_end"

        cmd = [
            sys.executable,
            str(train_script),
            "--tasks", tasks_arg,
            "--task_idx", str(i),
            "--end_ckpt", str(end_ckpt),
        ]

        # Chain the previous task's checkpoint (or the seed checkpoint for the
        # very first task in this run).
        if i == driver_args.start_idx:
            if driver_args.checkpoint:
                cmd += ["--checkpoint", driver_args.checkpoint]
        else:
            prev_ckpt = ckpt_dir / f"task{i-1:02d}_end"
            if not prev_ckpt.is_file():
                raise FileNotFoundError(
                    f"[FAME-CRL] Expected previous checkpoint not found: {prev_ckpt}"
                )
            cmd += ["--checkpoint", str(prev_ckpt)]

        cmd += passthrough

        print("\n[FAME-CRL] ══════════════════════════════════════════")
        print(f"[FAME-CRL] Task {i+1}/{len(tasks)}: {task}")
        print(f"[FAME-CRL] Command: {' '.join(cmd)}")
        print("[FAME-CRL] ══════════════════════════════════════════\n", flush=True)

        subprocess.run(cmd, check=True)

        if not end_ckpt.is_file():
            raise RuntimeError(
                f"[FAME-CRL] Task {i} finished but end checkpoint is missing: {end_ckpt}"
            )
        print(f"[FAME-CRL] Completed task {i+1}/{len(tasks)}: {task}")

    print("\n[FAME-CRL] All tasks completed.")


if __name__ == "__main__":
    main()
