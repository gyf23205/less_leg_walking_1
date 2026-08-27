# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the training methods by reading their tensorboard logs and printing the
converged Train/mean_reward for each task.

No Isaac Sim / simulation startup required - this only reads existing tensorboard logs.

Layout: logs/task1/<experiment_name>/<method>. FAME logs the same ``Train/mean_reward``
tag as the rsl_rl methods, so all five methods are directly comparable per task.
"""

import argparse
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "task1")

TASK_TO_EXPERIMENT = {
    "Less-AnymalC-Flat-Walking-Direct-v1": "anymal_c_flat_leg_walking",
    "Less-AnymalC-Jump-Direct-v1": "anymal_c_jump_flat",
    "Less-AnymalC-Rough-Walking-Direct-v1": "anymal_c_rough_leg_walking",
    "Less-Leg-Flat-Walking-Direct-v1": "less_leg_walking_flat",
    "Less-AnymalC-Jump-Rough-Direct-v1": "anymal_c_jump_rough",
    "Less-Leg-Rough-Walking-Direct-v1": "less_leg_walking_rough",
}

# Method -> log subfolder under logs/task1/<experiment_name>/.
METHODS = {
    "scratch": "scratch",
    "residual": "residual",
    "componet": "componet",
    "KAE_MoE": "KAE_MoE",
    "FAME": "FAME",
}

METRIC = "Train/mean_reward"
MIN_POINTS = 50


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
    parser = argparse.ArgumentParser(description="Compare converged reward across training methods, per task.")
    parser.add_argument("--window", type=int, default=100, help="Number of trailing points to average over.")
    args = parser.parse_args()

    rows = []
    for task, experiment in TASK_TO_EXPERIMENT.items():
        experiment_dir = os.path.join(LOG_ROOT, experiment)
        cells = [task]
        for folder in METHODS.values():
            value, num_points = converged_reward(os.path.join(experiment_dir, folder), args.window)
            cells.append(format_cell(value, num_points))
        rows.append(tuple(cells))

    headers = ("Task", *METHODS.keys())
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(row):
        return "  ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    main()
