import copy
import gymnasium

import torch

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from typing import Any, Mapping, Optional, Tuple, Union
# fmt: off
# [start-config-dict-torch]
TDCLS_DEFAULT_CONFIG = {
    "gamma": 0.99,             # discount factor

    "epsilon": 0.1,            # exploration noise std scale

    "batch_size": 512,         # training batch size

    "grad_norm_clip": 1.0,     # max gradient norm for actor and critic updates

    "rewards_shaper_scale": 0.1,  # multiply raw env rewards by this before storing

    "rebase_epoch": 200,       # steps between target_net <- T_net sync

    "P_net_update_epoch": 50000,  # steps between P_net update pass

    "decay": 0.75,             # multiplicative decay applied to T_net weights after P update

    "lr_T": 1e-4,              # learning rate for T_net (transient value)
    "lr_P": 1e-6,              # learning rate for P_net (permanent value)
    "lr_actor": 3e-4,          # learning rate for actor (policy)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class TDCLS(Agent):
    """Actor-Critic variant of TDCLS (Transient-and-Permanent Decomposed Continual Learning).

    Architecture
    ------------
    * **policy**    : GaussianMixin actor  → continuous actions  (batch, action_dim)
    * **T_net**     : DeterministicMixin   → scalar Q-value  Q_T(s,a)  (batch, 1)  [transient]
    * **target_net**: DeterministicMixin   → frozen copy of T_net for TD targets
    * **P_net**     : DeterministicMixin   → scalar Q-value  Q_P(s,a)  (batch, 1)  [permanent]

    Value networks take STATES_ACTIONS (concatenated) as input, i.e. Q(s, a).

    At each timestep:
      1. ``act``: actor samples actions; Q_T+Q_P evaluated at (s, a) gives Q(s,a).
      2. ``record_transition``: stores (s, a, Q(s,a), r, s') in replay.
      3. ``_update``: trains T_net with TD loss; trains actor with policy gradient
         using Q as advantage; periodically retrains P_net and decays T_net.
    """

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """
        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(TDCLS_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # --- models ---
        self.policy     = self.models.get("policy",     None)
        self.T_net      = self.models.get("T_net",      None)
        self.target_net = self.models.get("target_net", None)
        self.P_net      = self.models.get("P_net",      None)

        # initialise target_net as a copy of T_net
        self.target_net.load_state_dict(self.T_net.state_dict())
        self._exploration_timesteps = _cfg.get("exploration_timesteps", 1000)

        # --- optimisers ---
        self.actor_opt = torch.optim.Adam(self.policy.parameters(),     lr=_cfg.get("lr_actor", 3e-4))
        self.T_opt     = torch.optim.Adam(self.T_net.parameters(),      lr=_cfg.get("lr_T",     1e-4))
        self.P_opt     = torch.optim.Adam(self.P_net.parameters(),      lr=_cfg.get("lr_P",     1e-6))

        # --- hyper-parameters ---
        self._gamma              = _cfg["gamma"]
        self._epsilon            = _cfg["epsilon"]
        self._batch_size         = _cfg["batch_size"]
        self._grad_norm_clip     = _cfg.get("grad_norm_clip",     1.0)
        self._rewards_shaper_scale = _cfg.get("rewards_shaper_scale", 0.1)
        self._rebase_epoch       = _cfg.get("rebase_epoch",        200)
        self._P_net_update_epoch = _cfg.get("P_net_update_epoch", 50000)
        self._decay              = _cfg.get("decay", 0.75)
        self._count              = 0

        self.value_criterion = torch.nn.MSELoss()

        # buffer: P_net Q-value at collection time, stored in replay and used by _train_P_Net
        self._current_values = None

        # learning starts only after the replay buffer has at least one full sweep
        # (memory_size rows × num_envs envs), ensuring temporal diversity in early batches.
        # We retrieve memory_size from memory if available, else fall back to batch_size.
        _mem_size = getattr(memory, "memory_size", None) if memory is not None else None
        _num_envs  = getattr(memory, "num_envs",    1)    if memory is not None else 1
        self._learning_starts = (_mem_size * _num_envs) if _mem_size is not None else self._batch_size

        self._tensors_names = [
            "states", "actions", "values", "rewards", "next_states", "terminated", "truncated"
        ]

    def init(self, trainer_cfg: Optional[dict] = None) -> None:
        """Initialize the agent (create memory tensors, set eval mode).

        Must be called by the trainer before the first interaction.
        Mirrors the pattern used by skrl's SAC/TD3 agents.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        if self.memory is not None:
            self.memory.create_tensor(name="states",      size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions",     size=self.action_space,      dtype=torch.float32)
            self.memory.create_tensor(name="rewards",     size=1,                      dtype=torch.float32)
            self.memory.create_tensor(name="values",      size=1,                      dtype=torch.float32)
            self.memory.create_tensor(name="terminated",  size=1,                      dtype=torch.bool)
            self.memory.create_tensor(name="truncated",   size=1,                      dtype=torch.bool)

    # ------------------------------------------------------------------
    # Q-value helpers  — all networks take (s, a) as input
    # ------------------------------------------------------------------

    def _q_value(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Combined Q(s,a) = Q_T(s,a) + Q_P(s,a), shape (batch, 1)."""
        inp = {"states": states, "taken_actions": actions}
        T = self.T_net(inp, role="T_net")[0]
        P = self.P_net(inp, role="P_net")[0]
        return T + P

    def _target_q_value(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Bootstrap target: Q_target_T(s,a) + Q_P(s,a), shape (batch, 1)."""
        inp   = {"states": states, "taken_actions": actions}
        T_tgt = self.target_net(inp, role="target_net")[0]
        P     = self.P_net(inp,    role="P_net")[0]
        return T_tgt + P

    # ------------------------------------------------------------------
    # Training sub-routines
    # ------------------------------------------------------------------

    def _train_T_Net(self, states, actions, next_states, rewards, done):
        """One TD step on the transient Q-network."""
        with torch.no_grad():
            # Use deterministic mean actions for the bootstrap target to reduce variance
            next_actions, _, next_outputs = self.policy({"states": next_states}, role="policy")
            next_actions = next_outputs.get("mean_actions", next_actions)  # deterministic
            # DEBUG Is it correct to use the next actions here?
            next_q   = self._target_q_value(next_states, next_actions)  # (batch, 1)
            targets  = rewards + (1 - done) * self._gamma * next_q

        inp   = {"states": states, "taken_actions": actions}
        T_pred = self.T_net(inp, role="T_net")[0]
        P_pred = self.P_net(inp, role="P_net")[0].detach()
        loss   = self.value_criterion(T_pred + P_pred, targets)

        if not torch.isfinite(loss):
            return float("nan")

        self.T_opt.zero_grad()
        loss.backward()
        if self._grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.T_net.parameters(), self._grad_norm_clip)
        self.T_opt.step()
        return loss.item()

    def _train_P_Net(self):
        """Full pass over replay buffer to update the permanent Q-network."""
        loss_u  = 0.0
        # Mirror reference: min(memory_size // batch_size - 1, 100) batches
        # skrl memory.sample() uses 'mini_batches' (not 'batches')
        u_steps = min(max(1, (len(self.memory) // self._batch_size) - 1), 100)
        for batch in self.memory.sample(
            names=self._tensors_names, batch_size=self._batch_size, mini_batches=u_steps
        ):
            (
                sampled_states, sampled_actions, sampled_values,
                sampled_rewards, sampled_next_states,
                sampled_terminated, sampled_truncated,
            ) = batch
            inp = {"states": sampled_states, "taken_actions": sampled_actions}
            with torch.no_grad():
                T_pred = self.T_net(inp, role="T_net")[0]
            P_pred = self.P_net(inp, role="P_net")[0]
            # target for P: Q_T(s,a) + old_Q_P(s,a)  (old_Q_P stored as 'values' in replay)
            loss = self.value_criterion(P_pred, T_pred + sampled_values)
            self.P_opt.zero_grad()
            loss.backward()
            self.P_opt.step()
            loss_u += loss.item()
        return loss_u / u_steps

    def _train_actor(self, states):
        """Deterministic policy gradient step (DDPG-style): maximise E[Q_T(s, π(s))].

        The policy's deterministic mean is fed into T_net; gradients flow back
        through T_net into the policy.  T_net parameters are not updated here
        (only actor_opt is stepped), so this is the standard DDPG actor update.

        We zero T_net grads before and after the actor backward so that stale
        actor-induced gradients never accumulate on T_net between critic updates.
        """
        # Compute deterministic mean actions — differentiable w.r.t. policy params
        actions, _, outputs = self.policy({"states": states}, role="policy")
        mean_actions = outputs.get("mean_actions", actions)

        inp        = {"states": states, "taken_actions": mean_actions}
        q_t        = self.T_net(inp, role="T_net")[0]   # (batch, 1); grad flows to policy
        actor_loss = -q_t.mean()                        # maximise Q_T

        if not torch.isfinite(actor_loss):
            return float("nan")

        self.actor_opt.zero_grad()
        # Zero T_net grads BEFORE backward so no residual grad from _train_T_Net accumulates
        self.T_opt.zero_grad()
        actor_loss.backward()
        if self._grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
        self.actor_opt.step()
        # Zero T_net grads AFTER backward so actor-induced grads don't pollute next T update
        self.T_opt.zero_grad()
        return actor_loss.item()

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Trigger _update every step once memory has enough samples."""
        if self.memory is not None and len(self.memory) >= self._learning_starts:
            # print("starts learning")
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # base class handles TensorBoard writes and checkpoint saving
        super().post_interaction(timestep, timesteps)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        with torch.no_grad():
            # actor samples continuous actions
            actions, _, _ = self.policy({"states": states}, role="policy")

            # add exploration noise for the first _exploration_timesteps steps
            if timestep < self._exploration_timesteps:
                actions = actions + self._epsilon * torch.randn_like(actions)

            # evaluate Q_P on the actual action that will be sent to env
            inp = {"states": states, "taken_actions": actions}
            self._current_values = self.P_net(inp, role="P_net")[0]   # (batch, 1)

        return actions, None, {}

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        # V(s) buffered by act() this timestep  →  stored as 'values'
        values = self._current_values

        # Scale rewards to prevent Q-value explosion (same as PPO rewards_shaper_scale)
        rewards = rewards * self._rewards_shaper_scale

        if self.memory is not None:
            self.memory.add_samples(
                states=states, actions=actions, values=values,
                rewards=rewards, next_states=next_states,
                terminated=terminated, truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states, actions=actions, values=values,
                    rewards=rewards, next_states=next_states,
                    terminated=terminated, truncated=truncated,
                )

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample a batch from memory
        (
            sampled_states, sampled_actions, sampled_values,
            sampled_rewards, sampled_next_states,
            sampled_terminated, sampled_truncated,
        ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]
        sampled_done = (sampled_terminated | sampled_truncated).float()

        # 1. update transient Q-network (T)
        t_loss = self._train_T_Net(
            sampled_states, sampled_actions, sampled_next_states, sampled_rewards, sampled_done
        )

        # 2. update actor with DDPG-style deterministic policy gradient
        actor_loss = self._train_actor(sampled_states)

        self._count += 1

        # 3. periodically sync target network
        if self._count % self._rebase_epoch == 0:
            self.target_net.load_state_dict(self.T_net.state_dict())

        # 4. periodically update permanent value network (P) then decay T
        if self._count % self._P_net_update_epoch == 0:
            self._train_P_Net()
            with torch.no_grad():
                for param in self.T_net.parameters():
                    param.data *= self._decay