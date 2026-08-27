# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared wandb reporting for the CRL drivers (``scripts/rsl_rl/train_*_CRL.py``).

A CRL driver runs one training subprocess per task. Under a wandb sweep the whole chain is a
single trial, so the DRIVER owns the single wandb run and the per-task children are forced to
``--logger=tensorboard``. Letting each child call ``wandb.init()`` itself does not work: the
children run sequentially under the same ``WANDB_RUN_ID``, each restarting the step axis at
zero (wandb silently drops out-of-order steps, so every task after the first would vanish),
and each calling ``wandb.config.update`` with per-task values that legitimately differ.

Instead, after each task finishes, its TensorBoard scalars are replayed into the driver's run
with a monotonically increasing step offset, so the sweep sees ONE continuous learning curve
spanning the whole chain, plus per-task final rewards as summary values.

Steps are in cumulative-gradient-step units, because ``cli_args.patch_tensorboard_gradient_steps``
rescales the rsl_rl x-axis (see ``scripts/common/tb_xaxis.py``).
"""

from __future__ import annotations

import os

TB_REWARD_TAG = "Train/mean_reward"


def read_tb_series(log_dir: str, tag: str = TB_REWARD_TAG) -> list[tuple[int, float]]:
    """Return ``[(step, value), ...]`` for ``tag`` from the TensorBoard events in ``log_dir``."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [(int(e.step), float(e.value)) for e in acc.Scalars(tag)]


class CrlWandbReporter:
    """Owns one wandb run for a whole CRL chain; replays each task's curve into it.

    Args:
        project: wandb project. Under a sweep this must match the sweep's project, otherwise
            the run lands elsewhere and the controller never sees the metric.
        prefix: log prefix for driver messages, e.g. ``"RES-CRL"``.
    """

    def __init__(self, project: str | None = None, prefix: str = "CRL"):
        import wandb

        self._wandb = wandb
        self._prefix = prefix
        self._step_offset = 0
        self._final_rewards: list[float] = []
        wandb.init(project=project, entity=os.environ.get("WANDB_USERNAME"))
        print(
            f"[{prefix}] wandb run: {wandb.run.name} ({wandb.run.id}) in project {wandb.run.project}",
            flush=True,
        )

    def replay_task(self, log_dir: str, task_index: int, task_name: str) -> float | None:
        """Replay one finished task's reward curve into the run. Returns its final reward."""
        series = read_tb_series(log_dir)
        if not series:
            print(f"[{self._prefix}] WARNING: no '{TB_REWARD_TAG}' scalars under {log_dir}; nothing logged.")
            return None

        for step, value in series:
            self._wandb.log({TB_REWARD_TAG: value, "crl/task_index": task_index}, step=self._step_offset + step)

        final_reward = series[-1][1]
        self._wandb.summary[f"crl/task{task_index}_{task_name}_final_reward"] = final_reward
        self._step_offset += series[-1][0]
        self._final_rewards.append(final_reward)
        print(
            f"[{self._prefix}] Logged {len(series)} points for {task_name}; "
            f"final {TB_REWARD_TAG} = {final_reward:.4f}",
            flush=True,
        )
        return final_reward

    def finish(self) -> None:
        """Write the chain-level summary metrics and close the run."""
        if self._final_rewards:
            # The sweep objective: how good the swept setting is across the WHOLE chain, not
            # just on the task that happened to be trained last.
            self._wandb.summary["crl/mean_final_reward"] = sum(self._final_rewards) / len(self._final_rewards)
            self._wandb.summary["crl/final_task_reward"] = self._final_rewards[-1]
            self._wandb.summary["crl/num_tasks"] = len(self._final_rewards)
        self._wandb.finish()
