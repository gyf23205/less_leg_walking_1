"""Neural-network models for FAME in continuous-action IsaacLab environments.

Four classes are exposed:

PPOActor  (fast learner)
    MLP trunk → mean vector; state-independent log_std parameter.
    ``.forward(obs)`` → ``Normal`` distribution.
    ``.evaluate(obs, actions)`` → ``(log_probs, entropy)`` for PPO update.
    ``get_dist_params(obs)`` → ``(mu, std)`` for meta distillation.

PPOCritic  (fast learner value network)
    MLP trunk → scalar V(s).

DiagGaussianActor  (meta learner – SAC-style, tanh-squashed)
    Kept for meta-learner distillation.  Matches the Metaworld FAME actor.

DoubleQCritic  (meta learner – SAC-style)
    Kept for meta-learner Q-value distillation.

All MLPs default to ``hidden_dims=[512, 256, 128]`` to match the PPO
train-from-scratch baseline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd


# ---------------------------------------------------------------------------
# Shared MLP builder
# ---------------------------------------------------------------------------

def _make_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    activation: str = "elu",
) -> nn.Sequential:
    """Build a fully-connected network with the given layer sizes."""
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), act_cls()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _weight_init(m: nn.Module) -> None:
    """Orthogonal init for Linear layers (mirrors ``agent/utils.py``)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


# ---------------------------------------------------------------------------
# TanhTransform + SquashedNormal  (mirrors actor.py verbatim)
# ---------------------------------------------------------------------------

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size: int = 1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TanhTransform)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return x.tanh()

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.atanh(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """Normal distribution squashed through tanh.  ``mean`` property returns
    the tanh-transformed mean (deterministic action)."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        super().__init__(self.base_dist, [TanhTransform()])

    @property
    def mean(self) -> torch.Tensor:
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# ---------------------------------------------------------------------------
# Actor: DiagGaussianActor  (mirrors Metaworld actor.py)
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class DiagGaussianActor(nn.Module):
    """Diagonal Gaussian actor with tanh-squashed actions.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    hidden_dims : list[int]
        e.g. ``[256, 256]`` to match the Metaworld default of
        ``hidden_dim=256, hidden_depth=2``.
    log_std_bounds : tuple[float, float]
        ``(log_std_min, log_std_max)`` – clamps the log-std output.
    activation : str
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        # Matches train-from-scratch PPO baseline: [512, 256, 128]
        # NOTE: original FAME Metaworld default was: hidden_dims=(256, 256)
        hidden_dims: list[int] = (512, 256, 128),
        log_std_bounds: tuple[float, float] = (LOG_STD_MIN, LOG_STD_MAX),
        activation: str = "elu",
    ):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        # trunk outputs 2 * action_dim: first half = mu, second = log_std
        self.trunk = _make_mlp(obs_dim, list(hidden_dims), 2 * action_dim, activation)
        self.apply(_weight_init)

    def forward(self, obs: torch.Tensor) -> SquashedNormal:
        """Return a ``SquashedNormal`` distribution over actions."""
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max] via scaled tanh
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)

    def get_dist_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, std)`` before the tanh squash.  Used in meta distillation."""
        dist = self.forward(obs)
        return dist.loc, dist.scale


# ---------------------------------------------------------------------------
# Critic: DoubleQCritic  (mirrors Metaworld critic.py)
# ---------------------------------------------------------------------------

class DoubleQCritic(nn.Module):
    """Two independent Q-networks for double Q-learning (SAC-style).

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    hidden_dims : list[int]
    activation : str
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        # Matches train-from-scratch PPO baseline: [512, 256, 128]
        # NOTE: original FAME Metaworld default was: hidden_dims=(256, 256)
        hidden_dims: list[int] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.Q1 = _make_mlp(in_dim, list(hidden_dims), 1, activation)
        self.Q2 = _make_mlp(in_dim, list(hidden_dims), 1, activation)
        self.apply(_weight_init)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Q1(s,a), Q2(s,a))``, each shaped ``(B, 1)``."""
        x = torch.cat([obs, action], dim=-1)
        return self.Q1(x), self.Q2(x)


# ---------------------------------------------------------------------------
# PPO Fast-Learner networks
# ---------------------------------------------------------------------------

class PPOActor(nn.Module):
    """Gaussian actor for PPO (no tanh squash).

    Architecture: MLP trunk → mean; separate state-independent ``log_std``
    parameter (matches standard IsaacLab PPO baselines).

    ``forward(obs)`` → ``torch.distributions.Normal``
    ``evaluate(obs, actions)`` → ``(log_probs, entropy)``  (for PPO update)
    ``get_dist_params(obs)`` → ``(mu, std)``  (for meta WD distillation)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (512, 256, 128),
        activation: str = "elu",
        init_log_std: float = 0.0,
    ):
        super().__init__()
        self.trunk = _make_mlp(obs_dim, list(hidden_dims), action_dim, activation)
        # State-independent log_std (one per action dimension)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        self.apply(_weight_init)
        # Re-init log_std after _weight_init so it keeps the specified value
        nn.init.constant_(self.log_std, init_log_std)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Return an un-squashed Normal distribution."""
        mu  = self.trunk(obs)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-prob and entropy for PPO surrogate objective."""
        dist     = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)  # (B,1)
        entropy  = dist.entropy().sum(-1, keepdim=True)          # (B,1)
        return log_prob, entropy

    def get_dist_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, std)`` for meta WD distillation."""
        mu  = self.trunk(obs)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        return mu, std


class PPOCritic(nn.Module):
    """State-value network V(s) for PPO.

    ``forward(obs)`` → ``(B, 1)`` tensor of state values.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list[int] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        self.net = _make_mlp(obs_dim, list(hidden_dims), 1, activation)
        self.apply(_weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
