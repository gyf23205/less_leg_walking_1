"""Replay buffers for FAME in continuous-action IsaacLab environments.

All buffers operate purely in PyTorch (no numpy), so they work correctly with
the vectorised Isaac-Sim tensors that live on the GPU.

Classes
-------
ExpReplay
    Fast-learner transition buffer: stores (s, a, r, s', done).
    Used for training the Fast-Learner (critic + actor) and the Target-network.

ExpReplayMeta
    Distillation buffer: stores (s, a) pairs.
    - exp_replay_fast2meta  – filled from the *last* ``size_fast2meta`` steps of
      the current environment segment, then copied into exp_replay_meta.
    - exp_replay_meta       – accumulates (s, a) across all past environments;
      used to train the Meta-Learner via behaviour-cloning / NLL loss.
"""

from __future__ import annotations

import random
from collections import deque

import torch


class ExpReplay:
    """Circular transition replay buffer for continuous-action environments.

    Parameters
    ----------
    batch_size : int
        Number of transitions returned by :meth:`sample`.
    device : torch.device
        Device on which sampled batches are returned.
    max_size : int
        Maximum number of transitions to store.
    """

    def __init__(self, batch_size: int, device: torch.device, max_size: int = 100_000):
        self.memory: deque = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.device = device

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def store(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Store a *batch* of transitions (one per parallel env).

        All tensors are expected to have shape ``(num_envs, dim)`` or
        ``(num_envs,)`` and will be split into individual rows before
        appending to the deque.

        Parameters
        ----------
        obs, next_obs : (num_envs, obs_dim) float tensors
        action        : (num_envs, act_dim) float tensor
        reward        : (num_envs,) or (num_envs, 1) float tensor
        done          : (num_envs,) or (num_envs, 1) bool/float tensor
        """
        obs = obs.detach().cpu()
        action = action.detach().cpu()
        next_obs = next_obs.detach().cpu()
        reward = reward.detach().cpu().view(-1, 1).float()
        done = done.detach().cpu().view(-1, 1).float()

        for i in range(obs.shape[0]):
            self.memory.append(
                (obs[i], action[i], next_obs[i], reward[i], done[i])
            )

    def sample(self):
        """Return a batch of ``(states, actions, next_states, rewards, dones)``
        each shaped ``(batch_size, dim)`` on ``self.device``.
        """
        n = min(len(self.memory), self.batch_size)
        batch = random.sample(self.memory, n)
        states, actions, next_states, rewards, dones = map(torch.stack, zip(*batch))
        return (
            states.to(self.device),
            actions.to(self.device),
            next_states.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
        )

    def size(self) -> int:
        return len(self.memory)

    def delete(self) -> None:
        self.memory.clear()


class ExpReplayMeta:
    """(state, action) distillation buffer.

    Stores only state-action pairs; used for behaviour-cloning the
    Meta-Learner.

    Parameters
    ----------
    batch_size : int
    device     : torch.device
    max_size   : int
        When ``max_size`` is reached the oldest entries are evicted (FIFO
        via :class:`deque`).
    """

    def __init__(self, batch_size: int, device: torch.device, max_size: int = 100_000):
        self.memory: deque = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.device = device

    def store(self, obs: torch.Tensor, action: torch.Tensor) -> None:
        """Store a *batch* of (state, action) pairs.

        Parameters
        ----------
        obs    : (num_envs, obs_dim) float tensor
        action : (num_envs, act_dim) float tensor
        """
        obs = obs.detach().cpu()
        action = action.detach().cpu()
        for i in range(obs.shape[0]):
            self.memory.append((obs[i], action[i]))

    def sample(self):
        """Return ``(states, actions)`` each shaped ``(batch_size, dim)``."""
        n = min(len(self.memory), self.batch_size)
        batch = random.sample(self.memory, n)
        states, actions = map(torch.stack, zip(*batch))
        return states.to(self.device), actions.to(self.device)

    def size(self) -> int:
        return len(self.memory)

    def delete(self) -> None:
        self.memory.clear()

    def copy_to(self, target: "ExpReplayMeta") -> None:
        """Append all entries of this buffer into *target*."""
        for item in self.memory:
            target.memory.append(item)
