# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continual-RL driver for the CompoNet method.

Runs ``train_componet.py`` once per task in a fixed sequence. Unlike the residual
method (which only needs the single previous composed policy), CompoNet composes
over ALL previous stages, so this driver accumulates the growing list of saved
actor checkpoints and passes the full ordered list via ``--prevs_dir`` to each new
task. The chain is seeded with the shared base model (base_models/rough_walking.pth),
so even the first task composes over a pretrained policy instead of starting from a
bare FirstModuleWrapper:

    task 0 : --prevs_dir base                       -> CompoNet([base])
    task 1 : --prevs_dir base t0                    -> CompoNet([base, t0])
    task 2 : --prevs_dir base t0 t1                 -> CompoNet([base, t0, t1])
    task k : --prevs_dir base t0 ... t_{k-1}        -> CompoNet([base, t0..t_{k-1}])

Each ``train_componet.py`` run saves ``model.actor`` under the ``"actor"`` key of
``complete_model_with_metadata.pth``; that file becomes the next stage's prev.

Every run logs to ``logs/task2/<task name>/componet/<seed>`` (task name = the
experiment name used under logs/task1), with the seed picked
on the command line (``--seed``, default 42) and applied to every task in the chain.

Usage (extra CLI args after the known ones are forwarded to every run):

    python scripts/rsl_rl/train_componet_CRL.py --seed 1 --headless --num_envs 4096

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
# Less-AnymalC-Flat-Walking stays commented out to mirror train_res_CRL.py; both methods now
# start from the same shared BASE_MODEL instead of training that task as the first stage.
TRAIN_TASKS = [
    # "Less-AnymalC-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Direct-v1",
    "Less-Leg-Flat-Walking-Direct-v1",
    "Less-AnymalC-Rough-Walking-Direct-v1",
    "Less-AnymalC-Jump-Rough-Direct-v1",
    "Less-Leg-Rough-Walking-Direct-v1",
]

# The base policy every chain starts from; also the base for the residual method.
BASE_MODEL = str(
    Path(__file__).resolve().parents[2]
    / "source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1"
    / "base_models/rough_walking.pth"
)

# train_componet.py writes its saved policy here:
#   logs/task2/<experiment_name>/componet/<seed>[_<run_name>]/complete_model_with_metadata.pth
# The experiment name is a Hydra-config short name (anymal_c_jump_flat, ...) rather than the
# gym task id the driver passes on the command line, so it stays a wildcard here.
# The "<seed>*" wildcard matters too: under a wandb sweep every trial passes
# `agent.run_name=${oc.env:WANDB_RUN_ID}`, so the directory is suffixed with the run id.
def _model_glob(seed: int) -> str:
    return f"logs/task2/*/componet/{seed}*/complete_model_with_metadata.pth"


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

    return CrlWandbReporter(project, prefix="COMPO-CRL")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual CompoNet training driver.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for every task in the chain. Also the leaf of the log path: "
        "logs/task2/<task name>/componet/<seed>.",
    )
    parser.add_argument(
        "--start-from",
        dest="start_from",
        default=None,
        help="Resume the chain at the N-th task (1-based) or at a named task, instead of "
        "task 1. Requires --init_prevs, since the composition chain for the skipped "
        "stages can no longer be built by this run.",
    )
    parser.add_argument(
        "--init_prevs",
        type=str,
        nargs="+",
        default=None,
        help="Initial list of prev-stage actor checkpoints to seed the composition "
        f"chain (in order). Defaults to the shared base model: {BASE_MODEL}",
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

    prevs: list[str] = list(args.init_prevs) if args.init_prevs else [BASE_MODEL]
    for prev in prevs:
        if not os.path.isfile(prev):
            raise FileNotFoundError(f"Initial prev-stage checkpoint does not exist: {prev}")

    start_index = _resolve_start_index(args.start_from)
    if start_index > 0 and not args.init_prevs:
        raise SystemExit(
            f"--start-from {args.start_from} skips {start_index} task(s), so this run cannot "
            f"rebuild their composition chain. Pass --init_prevs with the base model followed "
            f"by each skipped task's complete_model_with_metadata.pth, in order."
        )
    tasks_to_run = TRAIN_TASKS[start_index:]
    completed = list(TRAIN_TASKS[:start_index])
    if start_index:
        print(f"[COMPO-CRL] Resuming at task {start_index + 1}/{len(TRAIN_TASKS)}; "
              f"treating {completed} as already done.")

    for task_name in tasks_to_run:
        command = [sys.executable, str(train_script), "--task", task_name, "--seed", str(args.seed)]
        if prevs:
            command += ["--prevs_dir", *prevs]
        command += forward_args
        if reporter is not None:
            # Appended last so it wins over any --logger in forward_args: the driver owns
            # the wandb run, the children only write TensorBoard events.
            command += ["--logger", "tensorboard"]

        print(f"\n[COMPO-CRL] ==============================================")
        print(f"[COMPO-CRL] Training task {len(completed) + 1}/{len(TRAIN_TASKS)}: {task_name}")
        print(f"[COMPO-CRL] Seed: {args.seed}")
        print(f"[COMPO-CRL] Previous stages ({len(prevs)}): {prevs if prevs else '(none — fresh first module)'}")
        print(f"[COMPO-CRL] ==============================================\n", flush=True)

        t_start = time.time()
        subprocess.run(command, check=True)

        produced = _latest_model_since(args.seed, t_start)
        if produced is None:
            raise RuntimeError(
                f"No complete_model_with_metadata.pth was produced for task '{task_name}'. "
                f"Expected one under {_model_glob(args.seed)}."
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
