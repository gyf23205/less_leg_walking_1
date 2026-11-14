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

The first task's base is the shared base model (base_models/rough_walking.pth),
overridable with ``--init_base``.

Every run logs to ``logs/task2/<task name>/residual/<seed>`` (task name = the
experiment name used under logs/task1), with the seed picked on
the command line (``--seed``, default 42) and applied to every task in the chain.

Usage (extra CLI args after the known ones are forwarded to every train_res.py call):

    python scripts/rsl_rl/train_res_CRL.py --seed 1 --headless --num_envs 4096

    # start from a specific base checkpoint instead of the shared base model:
    python scripts/rsl_rl/train_res_CRL.py --init_base /path/to/base.pth --headless

Because unknown args are forwarded verbatim, a Hydra override applies to EVERY task in the
chain -- which is how the architecture sweep gives every residual in the chain the same
structure:

    python scripts/rsl_rl/train_res_CRL.py --headless agent.policy.actor_hidden_dims=[256,128,64]

With ``--wandb`` the driver owns a single wandb run covering the whole chain; see
scripts/common/crl_wandb.py and scripts/wandb_sweep/residual_actor_width.yaml.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# Task sequence, matching the authoritative order in scripts/skrl/train_FAME_CRL.py
# (TASKS). The first task is trained from the default/initial base; each subsequent
# task is trained on top of the previous task's composed (base + residual) policy.
# Less-AnymalC-Flat-Walking is skipped here because for the residual method it IS the
# initial base (see BASE_MODEL below).
# NOTE: train_moe_CRL.py still lists Rough-Walking before Less-Leg-Flat-Walking; it has
# not been realigned (train_componet_CRL.py has).
TRAIN_TASKS = [
    # "Less-AnymalC-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Direct-v1",
    "Less-Leg-Flat-Walking-Direct-v1",
    "Less-AnymalC-Rough-Walking-Direct-v1",
    "Less-AnymalC-Jump-Rough-Direct-v1",
    "Less-Leg-Rough-Walking-Direct-v1",
]

# The frozen base the first task's residual is trained on top of; shared with CompoNet.
BASE_MODEL = str(
    Path(__file__).resolve().parents[2]
    / "source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1"
    / "base_models/rough_walking.pth"
)

# train_res.py writes its saved policy here:
#   logs/task2/<experiment_name>/residual/<seed>[_<run_name>]/complete_model_with_metadata.pth
# The experiment name is a Hydra-config short name (anymal_c_jump_flat, ...) rather than the
# gym task id the driver passes on the command line, so it stays a wildcard here.
# The "<seed>*" wildcard matters too: under a wandb sweep every trial passes
# `agent.run_name=${oc.env:WANDB_RUN_ID}`, so the directory is suffixed with the run id.
def _model_glob(seed: int) -> str:
    return f"logs/task2/*/residual/{seed}*/complete_model_with_metadata.pth"


def _latest_model_since(seed: int, t_start: float) -> str | None:
    """Return the complete_model_with_metadata.pth written after ``t_start`` (newest)."""
    candidates = [p for p in glob.glob(_model_glob(seed)) if os.path.getmtime(p) >= t_start]
    if not candidates:
        return None
    return os.path.abspath(max(candidates, key=os.path.getmtime))


def _resolve_start_index(start_from: str | None) -> int:
    """Map --start-from (1-based index or task name) to a 0-based index into TRAIN_TASKS."""
    if start_from is None:
        return 0
    if start_from.isdigit():
        index = int(start_from) - 1
    elif start_from in TRAIN_TASKS:
        index = TRAIN_TASKS.index(start_from)
    else:
        raise SystemExit(
            f"Invalid --start-from value: {start_from}\n"
            f"Expected 1..{len(TRAIN_TASKS)} or one of {TRAIN_TASKS}"
        )
    if not 0 <= index < len(TRAIN_TASKS):
        raise SystemExit(f"--start-from out of range: {start_from}")
    return index


def _make_wandb_reporter(project: str | None):
    """Build the shared CRL wandb reporter (scripts/common lives one level up)."""
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from common.crl_wandb import CrlWandbReporter

    return CrlWandbReporter(project, prefix="RES-CRL")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual residual training driver.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for every task in the chain. Also the leaf of the log path: "
        "logs/task2/<task name>/residual/<seed>.",
    )
    parser.add_argument(
        "--start-from",
        dest="start_from",
        default=None,
        help="Resume the chain at the N-th task (1-based) or at a named task, instead of "
        "task 1. Requires --init_base pointing at the previous task's composed policy, "
        "since that is the frozen base this run must build on.",
    )
    parser.add_argument(
        "--init_base",
        type=str,
        default=None,
        help=f"Initial base-policy checkpoint for the first task. Defaults to the "
        f"shared base model: {BASE_MODEL}",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Report the whole chain to a single wandb run owned by this driver "
        "(per-task children are forced to --logger=tensorboard and their curves are "
        "replayed into that run). Used by the residual_actor_width sweep.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="wandb project for --wandb. Under a sweep this must match the sweep's project.",
    )
    args, forward_args = parser.parse_known_args()

    reporter = _make_wandb_reporter(args.wandb_project) if args.wandb else None

    train_script = Path(__file__).with_name("train_res.py")
    if not train_script.is_file():
        raise FileNotFoundError(str(train_script))

    base_path = args.init_base or BASE_MODEL
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Initial base-policy checkpoint does not exist: {base_path}")

    start_index = _resolve_start_index(args.start_from)
    if start_index > 0 and not args.init_base:
        raise SystemExit(
            f"--start-from {args.start_from} skips {start_index} task(s), so the frozen base "
            f"for this run is the previous task's composed policy, not the shared base model. "
            f"Pass --init_base with that task's complete_model_with_metadata.pth."
        )
    tasks_to_run = TRAIN_TASKS[start_index:]
    completed = list(TRAIN_TASKS[:start_index])
    if start_index:
        print(f"[RES-CRL] Resuming at task {start_index + 1}/{len(TRAIN_TASKS)}; "
              f"treating {completed} as already done.")

    for task_name in tasks_to_run:
        command = [sys.executable, str(train_script), "--task", task_name, "--seed", str(args.seed)]
        if base_path is not None:
            command += ["--original_policy_path", base_path]
        command += forward_args
        if reporter is not None:
            # Appended last so it wins over any --logger in forward_args: the driver owns
            # the wandb run, the children only write TensorBoard events.
            command += ["--logger", "tensorboard"]

        print(f"\n[RES-CRL] ==============================================")
        print(f"[RES-CRL] Training task {len(completed) + 1}/{len(TRAIN_TASKS)}: {task_name}")
        print(f"[RES-CRL] Seed: {args.seed}")
        print(f"[RES-CRL] Base policy: {base_path}")
        print(f"[RES-CRL] ==============================================\n", flush=True)

        t_start = time.time()
        subprocess.run(command, check=True)

        produced = _latest_model_since(args.seed, t_start)
        if produced is None:
            raise RuntimeError(
                f"No complete_model_with_metadata.pth was produced for task '{task_name}'. "
                f"Expected one under {_model_glob(args.seed)}."
            )

        # The saved 'actor' is a ComposedActor(base, residual) -> next task's frozen base.
        base_path = produced
        completed.append(task_name)
        print(f"[RES-CRL] Completed {task_name}; next base = {base_path}", flush=True)

        if reporter is not None:
            reporter.replay_task(os.path.dirname(produced), len(completed), task_name)

    print(f"\n[RES-CRL] Finished all {len(completed)} tasks: {completed}")

    if reporter is not None:
        reporter.finish()


if __name__ == "__main__":
    main()
