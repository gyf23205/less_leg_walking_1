# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continual-RL driver for the CompoNet method.

Runs ``train_componet.py`` once per task in a fixed sequence. Unlike the residual
method (which only needs the single previous composed policy), CompoNet composes
over ALL previous stages, so this driver accumulates the growing list of saved
actor checkpoints and passes the full ordered list via ``--prevs_dir`` to each new
task:

    task 0 : (no --prevs_dir)                     -> FirstModuleWrapper
    task 1 : --prevs_dir t0                        -> CompoNet([t0])
    task 2 : --prevs_dir t0 t1                      -> CompoNet([t0, t1])
    task k : --prevs_dir t0 t1 ... t_{k-1}          -> CompoNet([t0..t_{k-1}])

Each ``train_componet.py`` run saves ``model.actor`` under the ``"actor"`` key of
``complete_model_with_metadata.pth``; that file becomes the next stage's prev.

Usage (extra CLI args after the known ones are forwarded to every run):

    python scripts/rsl_rl/train_componet_CRL.py --headless --num_envs 4096
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# Task sequence (mirrors train_moe_CRL.py / train_res_CRL.py ordering).
TRAIN_TASKS = [
    # "Less-AnymalC-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Direct-v1",
    "Less-AnymalC-Rough-Walking-Direct-v1",
    "Less-Leg-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Rough-Direct-v1",
    "Less-Leg-Rough-Walking-Direct-v1",
]

# train_componet.py writes its saved policy here (experiment_name depends on the task):
#   logs/rsl_rl/<experiment_name>/componet/complete_model_with_metadata.pth
_MODEL_GLOB = "logs/rsl_rl/*/componet/complete_model_with_metadata.pth"


def _latest_model_since(t_start: float) -> str | None:
    """Return the complete_model_with_metadata.pth written after ``t_start`` (newest)."""
    candidates = [p for p in glob.glob(_MODEL_GLOB) if os.path.getmtime(p) >= t_start]
    if not candidates:
        return None
    return os.path.abspath(max(candidates, key=os.path.getmtime))


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual CompoNet training driver.")
    parser.add_argument(
        "--init_prevs",
        type=str,
        nargs="+",
        default=None,
        help="Optional initial list of prev-stage actor checkpoints to seed the "
        "composition chain (in order). If omitted, the first task starts fresh.",
    )
    args, forward_args = parser.parse_known_args()

    train_script = Path(__file__).with_name("train_componet.py")
    if not train_script.is_file():
        raise FileNotFoundError(str(train_script))

    prevs: list[str] = list(args.init_prevs) if args.init_prevs else []
    completed = []

    for task_name in TRAIN_TASKS:
        command = [sys.executable, str(train_script), "--task", task_name]
        if prevs:
            command += ["--prevs_dir", *prevs]
        command += forward_args

        print(f"\n[COMPO-CRL] ==============================================")
        print(f"[COMPO-CRL] Training task {len(completed) + 1}/{len(TRAIN_TASKS)}: {task_name}")
        print(f"[COMPO-CRL] Previous stages ({len(prevs)}): {prevs if prevs else '(none — fresh first module)'}")
        print(f"[COMPO-CRL] ==============================================\n", flush=True)

        t_start = time.time()
        subprocess.run(command, check=True)

        produced = _latest_model_since(t_start)
        if produced is None:
            raise RuntimeError(
                f"No complete_model_with_metadata.pth was produced for task '{task_name}'. "
                f"Expected one under {_MODEL_GLOB}."
            )

        # Append this stage's saved actor so it composes into every later task.
        prevs.append(produced)
        completed.append(task_name)
        print(f"[COMPO-CRL] Completed {task_name}; chain length now {len(prevs)}", flush=True)

    print(f"\n[COMPO-CRL] Finished all {len(completed)} tasks: {completed}")


if __name__ == "__main__":
    main()
