# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the residual and scratch training methods by reading their tensorboard logs
and printing the converged Train/mean_reward for each task.

No Isaac Sim / simulation startup required - this only reads existing tensorboard logs.
"""

import argparse
import os
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "rsl_rl")

TASK_TO_EXPERIMENT = {
    "Less-AnymalC-Flat-Walking-Direct-v1": "anymal_c_flat_leg_walking",
    "Less-AnymalC-Jump-Direct-v1": "anymal_c_jump_flat",
    "Less-AnymalC-Rough-Walking-Direct-v1": "anymal_c_rough_leg_walking",
    "Less-Leg-Flat-Walking-Direct-v1": "less_leg_walking_flat",
    "Less-AnymalC-Jump-Rough-Direct-v1": "anymal_c_jump_rough",
    "Less-Leg-Rough-Walking-Direct-v1": "less_leg_walking_rough",
}

METRIC = "Train/mean_reward"
MIN_POINTS = 50

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def find_scratch_dir(experiment_dir: str) -> str | None:
    if not os.path.isdir(experiment_dir):
        return None
    literal_scratch = os.path.join(experiment_dir, "scratch")
    if os.path.isdir(literal_scratch):
        return literal_scratch
    candidates = sorted(name for name in os.listdir(experiment_dir) if TIMESTAMP_RE.match(name))
    if not candidates:
        return None
    return os.path.join(experiment_dir, candidates[-1])


def converged_reward(run_dir: str | None, window: int) -> tuple[float | None, int]:
    if run_dir is None or not os.path.isdir(run_dir):
        return None, 0
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if METRIC not in ea.Tags().get("scalars", []):
        return None, 0
    points = ea.Scalars(METRIC)
    tail = points[-window:]
    return sum(p.value for p in tail) / len(tail), len(points)


def format_cell(value: float | None, num_points: int) -> str:
    if value is None:
        return "MISSING"
    if num_points < MIN_POINTS:
        return f"INSUFFICIENT DATA (n={num_points})"
    return f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Compare converged reward between residual and scratch methods.")
    parser.add_argument("--window", type=int, default=100, help="Number of trailing points to average over.")
    args = parser.parse_args()

    rows = []
    for task, experiment in TASK_TO_EXPERIMENT.items():
        experiment_dir = os.path.join(LOG_ROOT, experiment)
        scratch_dir = find_scratch_dir(experiment_dir)
        residual_dir = os.path.join(experiment_dir, "residual")

        scratch_value, scratch_n = converged_reward(scratch_dir, args.window)
        residual_value, residual_n = converged_reward(residual_dir, args.window)

        delta = None
        if (
            scratch_value is not None
            and residual_value is not None
            and scratch_n >= MIN_POINTS
            and residual_n >= MIN_POINTS
        ):
            delta = residual_value - scratch_value

        rows.append(
            (
                task,
                format_cell(scratch_value, scratch_n),
                format_cell(residual_value, residual_n),
                f"{delta:+.3f}" if delta is not None else "-",
            )
        )

    headers = ("Task", "Scratch (converged)", "Residual (converged)", "Delta (res - scratch)")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(row):
        return "  ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    main()
