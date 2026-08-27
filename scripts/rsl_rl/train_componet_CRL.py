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

Because unknown args are forwarded verbatim, a Hydra override applies to EVERY task in the
chain -- which is how the architecture sweep gives every CompoNet module the same structure:

    python scripts/rsl_rl/train_componet_CRL.py --headless \
        agent.policy.internal_policy_hidden_dims=[256,128,64]

With ``--wandb`` the driver owns a single wandb run covering the whole chain; see
scripts/common/crl_wandb.py and scripts/wandb_sweep/componet_internal_width.yaml.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# Task sequence, matching the authoritative order in scripts/skrl/train_FAME_CRL.py (TASKS)
# and the active entries of train_res_CRL.py, so the two methods are directly comparable.
# Less-AnymalC-Flat-Walking stays commented out to mirror train_res_CRL.py (for the residual
# method it IS the initial base; CompoNet instead starts from a fresh FirstModuleWrapper).
TRAIN_TASKS = [
    # "Less-AnymalC-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Direct-v1",
    "Less-Leg-Flat-Walking-Direct-v1",
    "Less-AnymalC-Rough-Walking-Direct-v1",
    "Less-AnymalC-Jump-Rough-Direct-v1",
    "Less-Leg-Rough-Walking-Direct-v1",
]

# train_componet.py writes its saved policy here (experiment_name depends on the task):
#   logs/task1/<experiment_name>/componet[_<run_name>]/complete_model_with_metadata.pth
# The "componet*" wildcard matters: under a wandb sweep every trial passes
# `agent.run_name=${oc.env:WANDB_RUN_ID}`, so the directory is suffixed with the run id.
_MODEL_GLOB = "logs/task1/*/componet*/complete_model_with_metadata.pth"


def _latest_model_since(t_start: float) -> str | None:
    """Return the complete_model_with_metadata.pth written after ``t_start`` (newest)."""
    candidates = [p for p in glob.glob(_MODEL_GLOB) if os.path.getmtime(p) >= t_start]
    if not candidates:
        return None
    return os.path.abspath(max(candidates, key=os.path.getmtime))


def _make_wandb_reporter(project: str | None):
    """Build the shared CRL wandb reporter (scripts/common lives one level up)."""
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from common.crl_wandb import CrlWandbReporter

    return CrlWandbReporter(project, prefix="COMPO-CRL")


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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Report the whole chain to a single wandb run owned by this driver "
        "(per-task children are forced to --logger=tensorboard and their curves are "
        "replayed into that run). Used by the componet_internal_width sweep.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="wandb project for --wandb. Under a sweep this must match the sweep's project.",
    )
    args, forward_args = parser.parse_known_args()

    reporter = _make_wandb_reporter(args.wandb_project) if args.wandb else None

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
        if reporter is not None:
            # Appended last so it wins over any --logger in forward_args: the driver owns
            # the wandb run, the children only write TensorBoard events.
            command += ["--logger", "tensorboard"]

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

        if reporter is not None:
            reporter.replay_task(os.path.dirname(produced), len(completed), task_name)

    print(f"\n[COMPO-CRL] Finished all {len(completed)} tasks: {completed}")

    if reporter is not None:
        reporter.finish()


if __name__ == "__main__":
    main()
