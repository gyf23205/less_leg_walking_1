# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared TensorBoard x-axis rescaling.

All training methods in this repo (rsl_rl PPO variants and the skrl FAME baseline) should plot
against the *same* x quantity: the cumulative number of minibatch optimizer updates
(``num_learning_epochs * num_mini_batches`` per rollout). Each framework logs a different native x:

- rsl_rl logs the learning-iteration index ``it`` (= number of rollouts collected). Multiply by
  ``num_learning_epochs * num_mini_batches`` to get cumulative minibatch updates.
- FAME (skrl) logs ``task_step`` (= number of environment-step rows collected). Multiply by
  ``(num_learning_epochs * num_mini_batches) / rollout_steps``.

Both are a single linear rescale of ``global_step``, so we monkeypatch
``torch.utils.tensorboard.SummaryWriter.add_scalar`` once with the appropriate multiplier.
"""

from __future__ import annotations


def patch_writer_gradient_steps(multiplier: float) -> None:
    """Rescale every ``SummaryWriter.add_scalar`` global_step by ``multiplier``.

    This affects all ``torch.utils.tensorboard.SummaryWriter`` instances process-wide (both the
    rsl_rl runner's writer and, for FAME, the train-loop writer and the skrl agent writer), so runs
    with different ``num_learning_epochs`` / ``num_mini_batches`` / rollout lengths stay directly
    comparable at equal x-axis values.

    Wall-clock series (tags ending in ``/time``) and calls without a ``global_step`` are left
    untouched. Idempotent-safe: applying twice would compound the multiplier, so call exactly once
    per process.
    """
    from torch.utils.tensorboard import SummaryWriter

    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")

    original_add_scalar = SummaryWriter.add_scalar

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        if global_step is not None and not str(tag).endswith("/time"):
            global_step = int(global_step * multiplier)
        return original_add_scalar(self, tag, scalar_value, global_step, *args, **kwargs)

    SummaryWriter.add_scalar = add_scalar
