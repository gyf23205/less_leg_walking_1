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
    """Tanh-squashed diagonal-Gaussian actor for FAME's fast/meta learner.

    Faithful mirror of the original FAME ``DiagGaussianActor``
    (``FAME/Metaworld/agent/actor.py``): a single MLP trunk outputs
    ``2 * action_dim`` = ``(mu, log_std)`` with **state-dependent** ``log_std``
    smoothly bounded into ``[LOG_STD_MIN, LOG_STD_MAX]`` via a scaled ``tanh``
    (not a free ``nn.Parameter`` + hard ``clamp``), and the action is
    ``tanh``-squashed through ``SquashedNormal`` so raw actions are bounded to
    (-1, 1) and cannot explode.

    ``forward(obs)`` → ``SquashedNormal`` distribution over actions ∈ (-1, 1)
    ``evaluate(obs, actions)`` → ``(log_probs, entropy_proxy)`` for the PPO update.
        The squashed distribution has no closed-form entropy, so the entropy
        proxy is ``-log_prob`` (as in SAC); exploration is regulated by the
        agent's auto-tuned temperature, not a fixed bonus.
    ``get_dist_params(obs)`` → ``(mu, std)`` of the **pre-squash** Normal, used
        for the meta Wasserstein-distillation loss.

    The class name is kept as ``PPOActor`` so existing imports stay valid; note
    the architecture is now squashed (state-dependent log_std, 2*action_dim head).
    """

    # small epsilon to keep atanh(action) finite for stored boundary actions
    _ACT_EPS = 1e-6

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (512, 256, 128),
        activation: str = "elu",
        log_std_bounds: tuple[float, float] = (LOG_STD_MIN, LOG_STD_MAX),
    ):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        # trunk outputs 2 * action_dim: first half = mu, second = log_std
        self.trunk = _make_mlp(obs_dim, list(hidden_dims), 2 * action_dim, activation)
        self.apply(_weight_init)

    def _mu_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max] via scaled tanh
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return mu, std

    def forward(self, obs: torch.Tensor) -> SquashedNormal:
        """Return a tanh-squashed Normal distribution over actions ∈ (-1, 1)."""
        mu, std = self._mu_std(obs)
        return SquashedNormal(mu, std)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-prob (with tanh Jacobian) and entropy proxy for PPO.

        ``actions`` are the squashed actions previously sent to the env; they
        are clamped just inside (-1, 1) so the internal ``atanh`` stays finite.
        """
        dist     = self.forward(obs)
        eps      = self._ACT_EPS
        actions  = actions.clamp(-1.0 + eps, 1.0 - eps)
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)  # (B,1)
        # No closed-form entropy for the squashed dist; use -log_prob as proxy.
        entropy  = -log_prob
        return log_prob, entropy

    def get_dist_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, std)`` of the pre-squash Normal for meta WD distillation."""
        return self._mu_std(obs)


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
