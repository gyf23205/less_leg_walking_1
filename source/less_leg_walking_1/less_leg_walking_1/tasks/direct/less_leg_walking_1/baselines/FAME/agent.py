"""FAME agent for continuous-action IsaacLab environments.

This module adapts the FAME algorithm
("Principled Fast and Meta Knowledge Learners for Continual RL",
 Anand et al. 2023) from the discrete MinAtar setting to the
*continuous-action, vectorised* IsaacLab setting.

Both the **Fast Learner** and the **Meta Learner** use **PPO** (Proximal
Policy Optimisation) with a clipped surrogate objective, GAE advantage
estimation, and a shared MLP actor + separate value network — identical in
architecture to the train-from-scratch PPO baseline.

The Meta Learner (``PPOActor`` + ``PPOCritic``) is distilled from the fast
learner via Wasserstein-like loss on ``(mu, std)`` of the policy Gaussian.
The meta critic is warm-started by matching ``V_meta(s) ≈ V_fast(s)``.

skrl integration:
  * Inherits ``skrl.agents.torch.Agent`` for TensorBoard, checkpointing,
    ``set_mode()``, ``track_data()``, and ``post_interaction()``.
  * Uses ``skrl.memories.torch.RandomMemory`` as a rollout buffer (PPO)
    and as persistent distillation buffers (fast2meta, meta).

Public API (called by ``scripts/skrl/train_FAME.py``)
------------------------------------------------------
    agent.act(states, timestep, timesteps)         → (actions, log_probs, {values})
    agent.record_transition(...)                   → fills rollout + fast2meta
    agent.post_interaction(timestep, timesteps)    → PPO update every rollout_steps
    agent.on_env_switch(new_env_name, eval_fn, …)  → detection + init choice
    agent.end_of_env_segment()                     → meta distillation + buffer swap
"""

from __future__ import annotations

import copy
import time
from typing import Any, List, Mapping, Optional, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory, RandomMemory
from skrl.models.torch import Model

from .model import PPOActor, PPOCritic


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

FAME_DEFAULT_CONFIG = {
    # --- PPO learning rates ---
    "lr_actor":  3e-4,
    "lr_critic": 3e-4,
    "lr_meta":   3e-4,    # meta-learner distillation lr

    # --- PPO RL ---
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_ratio":     0.2,
    "entropy_coef":   0.02,
    "value_coef":     1.0,
    # Tighter gradient clipping can help stabilise learning on large batches
    "grad_norm_clip": 0.5,

    # --- PPO update schedule ---
    # rollout_steps: number of env-step rows collected before each PPO update.
    # With 4096 parallel envs: rollout_steps=24 → 24*4096 ≈ 98k transitions
    # (matches the IsaacLab PPO baseline of 24 rollout steps × 4096 envs).
    "rollout_steps": 24,
    # Increase optimisation per-sample: more epochs + smaller minibatches
    # => better sample-efficiency (more SGD steps per collected row)
    "ppo_epochs":    8,     # optimisation epochs over each rollout
    "mini_batch_size": 2048,  # mini-batch size within each epoch
    # Number of minibatches to split the total collected samples into
    # if `mini_batch_size` would otherwise be larger than `total_samples`.
    "num_mini_batches": 4,

    # --- rewards ---
    "rewards_shaper_scale": 1.0,

    # --- FAME-specific ---
    "detection_step": 100,
    "epoch_meta":     200,
    "meta_lr_scheduler_gamma": 0.95,
    "reset":     True,
    "warmstep":  50_000,    # rollout rows with BC warm-start regularisation
    "lambda_reg": 1.0,

    # --- network architecture ---
    # Matches train-from-scratch PPO baseline: [512, 256, 128]
    # NOTE: original FAME Metaworld defaults were: [256, 256]
    "hidden_dims": [512, 256, 128],
    "activation":  "elu",

    # --- skrl experiment bookkeeping ---
    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FAMEAgent
# ---------------------------------------------------------------------------

class FAMEAgent(Agent):
    """FAME PPO continual-learning agent for IsaacLab.

    Both the **Fast Learner** and **Meta Learner** use PPO with
    ``PPOActor`` (MLP, hidden=[512,256,128]) and ``PPOCritic``
    (same MLP → scalar V).

    The Fast Learner is trained online via rollout buffer + PPO.
    The Meta Learner is distilled from the fast learner via WD loss on
    ``(mu, std)`` of the policy Gaussian, and its value network is trained
    to match ``V_fast(s)``.

    Three ``skrl.memories.torch.RandomMemory`` instances:

    ``memory`` (rollout buffer)
        Stores ``(states, actions, rewards, next_states, terminated,
        truncated, values, log_probs)`` for one PPO rollout.
        Size = ``rollout_steps``, ``num_envs`` = actual parallel envs.

    ``memory_fast2meta``
        ``(states, actions)`` collected during this segment for distillation.
        ``num_envs=1``, CPU.

    ``memory_meta``
        Permanent ``(states, actions)`` buffer across all past segments.
        ``num_envs=1``, CPU.
    """

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]],
        memory_fast2meta: RandomMemory,
        memory_meta: RandomMemory,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(FAME_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # ---- Distillation memories --------------------------------------
        self.memory_fast2meta = memory_fast2meta
        self.memory_meta      = memory_meta

        # ---- Dimensions -------------------------------------------------
        obs_dim = (
            observation_space.shape[-1]
            if hasattr(observation_space, "shape")
            else int(observation_space)
        )
        action_dim = (
            action_space.shape[-1]
            if hasattr(action_space, "shape")
            else int(action_space)
        )
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        hidden = _cfg["hidden_dims"]
        act    = _cfg["activation"]

        # ---- Fast Learner: PPO actor + value network --------------------
        self.fast_actor  = PPOActor(obs_dim, action_dim, hidden, activation=act).to(self.device)
        self.fast_critic = PPOCritic(obs_dim, hidden, activation=act).to(self.device)
        self.fast_actor_opt  = Adam(self.fast_actor.parameters(),  lr=_cfg["lr_actor"])
        self.fast_critic_opt = Adam(self.fast_critic.parameters(), lr=_cfg["lr_critic"])

        # ---- Meta Learner: same PPO architecture as fast learner -------
        self.meta_actor  = PPOActor(obs_dim, action_dim, hidden, activation=act).to(self.device)
        self.meta_critic = PPOCritic(obs_dim, hidden, activation=act).to(self.device)
        self.meta_actor_opt  = Adam(self.meta_actor.parameters(),  lr=_cfg["lr_meta"])
        self.meta_critic_opt = Adam(self.meta_critic.parameters(), lr=_cfg["lr_meta"])
        self.meta_scheduler        = ExponentialLR(self.meta_actor_opt,  gamma=_cfg["meta_lr_scheduler_gamma"])
        self.meta_critic_scheduler = ExponentialLR(self.meta_critic_opt, gamma=_cfg["meta_lr_scheduler_gamma"])

        # ---- Random Learner: same PPO architecture as fast learner. This is only used for initialization, not to be trained -------
        self.random_actor  = PPOActor(obs_dim, action_dim, hidden, activation=act).to(self.device)

        # ---- Register for skrl checkpointing ---------------------------
        self.checkpoint_modules["fast_actor"]  = self.fast_actor
        self.checkpoint_modules["fast_critic"] = self.fast_critic
        self.checkpoint_modules["meta_actor"]  = self.meta_actor
        self.checkpoint_modules["meta_critic"] = self.meta_critic

        # ---- Hyper-parameter aliases -----------------------------------
        self._gamma         = _cfg["gamma"]
        self._gae_lambda    = _cfg["gae_lambda"]
        self._clip_ratio    = _cfg["clip_ratio"]
        self._entropy_coef  = _cfg["entropy_coef"]
        self._value_coef    = _cfg["value_coef"]
        self._grad_clip     = _cfg["grad_norm_clip"]
        self._reward_scale  = _cfg["rewards_shaper_scale"]
        self._rollout_steps = _cfg["rollout_steps"]
        self._ppo_epochs    = _cfg["ppo_epochs"]
        self._mini_batch    = _cfg["mini_batch_size"]
        self._warmstep      = _cfg["warmstep"]
        self._lambda_reg    = _cfg["lambda_reg"]

        # ---- Tensor name lists -----------------------------------------
        self._rollout_names = ["states", "actions", "rewards", "next_states",
                               "terminated", "truncated", "values", "log_probs"]
        self._meta_names    = ["states", "actions"]

        # ---- Internal state --------------------------------------------
        self._update_count: int = 0
        self._task_step: int    = 0
        self._meta_warmup: bool = False
        self._rollout_ptr: int  = 0   # rows collected in current rollout

        # ---- FAME tracking ---------------------------------------------
        self.games: List[str]    = []
        self.flag_reg: List[str] = []

    # ------------------------------------------------------------------
    # skrl Agent interface
    # ------------------------------------------------------------------

    def init(self, trainer_cfg: Optional[dict] = None) -> None:
        """Create memory tensors and initialise TensorBoard writer."""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        if self.memory is not None:
            self.memory.create_tensor(name="states",      size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions",     size=self.action_space,      dtype=torch.float32)
            self.memory.create_tensor(name="rewards",     size=1,                      dtype=torch.float32)
            self.memory.create_tensor(name="terminated",  size=1,                      dtype=torch.bool)
            self.memory.create_tensor(name="truncated",   size=1,                      dtype=torch.bool)
            self.memory.create_tensor(name="values",      size=1,                      dtype=torch.float32)
            self.memory.create_tensor(name="log_probs",   size=1,                      dtype=torch.float32)

        for mem in (self.memory_fast2meta, self.memory_meta):
            mem.create_tensor(name="states",  size=self.observation_space, dtype=torch.float32)
            mem.create_tensor(name="actions", size=self.action_space,      dtype=torch.float32)

    def set_mode(self, mode: str) -> None:
        """Toggle train/eval on all plain nn.Module networks."""
        training = (mode == "train")
        for net in (self.fast_actor, self.fast_critic,
                    self.meta_actor, self.meta_critic):
            net.train(training)

    def act(
        self,
        states: torch.Tensor,
        timestep: int,
        timesteps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Sample action from the PPO fast actor.

        Caches ``log_probs`` and ``values`` in ``_last_log_probs`` /
        ``_last_values`` so ``record_transition`` can store them without
        a second forward pass.
        """
        with torch.no_grad():
            dist     = self.fast_actor(states)
            actions  = dist.sample()
            log_prob = dist.log_prob(actions).sum(-1, keepdim=True)  # (N,1)
            values   = self.fast_critic(states)                       # (N,1)
        self._last_log_probs = log_prob
        self._last_values    = values
        return actions, log_prob, {}

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
        """Store one env-step row into the rollout buffer and fast2meta.

        The rollout buffer uses the full ``num_envs`` dimension (on-policy).
        fast2meta uses ``num_envs=1`` (CPU) for distillation.
        """
        super().record_transition(
            states, actions, rewards, next_states,
            terminated, truncated, infos, timestep, timesteps,
        )

        scaled_rewards = rewards.float() * self._reward_scale

        if self.memory is not None:
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=scaled_rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                values=self._last_values,
                log_probs=self._last_log_probs,
            )
            self._rollout_ptr += 1

        # fast2meta (CPU, num_envs=1) — flatten num_envs
        n = states.shape[0]
        s_cpu = states.cpu()
        a_cpu = actions.cpu()
        for i in range(n):
            self.memory_fast2meta.add_samples(
                states=s_cpu[i:i+1],
                actions=a_cpu[i:i+1],
            )

        self._task_step += 1

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Trigger a PPO update once a full rollout has been collected."""
        if self._rollout_ptr >= self._rollout_steps:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")
            self.memory.reset()
            self._rollout_ptr = 0
        super().post_interaction(timestep, timesteps)

    # ------------------------------------------------------------------
    # Environment-switch logic
    # ------------------------------------------------------------------

    def on_env_switch(
        self,
        new_env_name: str,
        eval_fn,
        is_first_switch: bool = False,
    ) -> str:
        """Detection + initialisation at task switch (mirrors test_main.py).

        Parameters
        ----------
        eval_fn : callable
            ``eval_fn(actor, num_steps) -> float`` – runs the actor in the
            *new* environment and returns mean episode return.
        """
        self.games.append(new_env_name)
        self._meta_warmup = False
        self._task_step   = 0

        if is_first_switch:
            print(f"[FAME] First environment: {new_env_name} – no detection needed.")
            self.flag_reg.append("Initial")
            return "Initial"

        det_steps = self.cfg["detection_step"]
        print(f"\n[FAME] ══ Switching to {new_env_name} ══")
        print("[FAME] Step 1: Detection via Policy Evaluation …")

        avg_fast = eval_fn(self.fast_actor, det_steps)
        print(f"[FAME]   Fast-Learner avg return: {avg_fast:.3f}")


        avg_meta = eval_fn(self.meta_actor, det_steps)
        print(f"[FAME]   Meta-Learner avg return: {avg_meta:.3f}")
    
        avg_random = eval_fn(self.random_actor, det_steps)
        print(f"[FAME]   Random-Learner avg return: {avg_random:.3f}")

        avgs = [avg_fast, avg_meta, avg_random]

        if np.argmax(avgs) == 0:
            print("[FAME] Step 2: Fine-tune Fast-Learner (Fast ≥ Meta).")
            choice = "Fast"
        elif np.argmax(avgs) == 1:
            print("[FAME] Step 2: Meta init → warm-starting Fast-Learner.")
            self._copy_meta_to_fast()
            self._meta_warmup = True
            choice = "Meta"
        else:
            print("[FAME] Step 2: Random init (both policies poor).")
            self._reset_fast_learner()
            choice = "Random"

        self.flag_reg.append(choice)
        return choice

    def end_of_env_segment(self) -> None:
        """Distil Fast→Meta, copy fast2meta→meta, reset buffers for next task."""
        if len(self.memory_meta) > 0:
            print("[FAME] Step 3: Updating Meta-Learner …")
            t0 = time.time()
            self._train_meta()
            print(f"[FAME]   Meta update done in {time.time() - t0:.1f}s")
        else:
            print("[FAME] Step 3: First segment – no Meta update needed.")

        self._copy_fast2meta_to_meta()
        self.memory_fast2meta.reset()
        print(
            f"[FAME] Step 4: Meta buffer size: {len(self.memory_meta)}"
            f"  |  fast2meta cleared."
        )
        self.memory.reset()

    # ------------------------------------------------------------------
    # Internal PPO update
    # ------------------------------------------------------------------

    def _update(self, timestep: int, timesteps: int) -> None:
        """Full PPO update over the collected rollout.

        1. Compute GAE returns + advantages over the rollout buffer.
        2. Optimise for ``ppo_epochs`` epochs with ``mini_batch_size`` batches.
        """
        # ---- Bootstrap value for last observation ---------------------
        # next_states[-1] is the observation after the last rollout step.
        # We need V(s_T) to bootstrap if the episode did not terminate.
        (
            s_all, a_all, r_all, s2_all, term_all, trunc_all, v_all, lp_all,
        ) = self.memory.sample_all(names=self._rollout_names)[0]
        # All tensors: shape (rollout_steps, num_envs, dim) or (T*N, dim)
        # skrl's RandomMemory returns (memory_size * num_envs, dim) flattened
        # We need to unflatten: T = rollout_steps, N = num_envs
        T = self._rollout_steps
        N = s_all.shape[0] // T

        def _uf(x):
            return x.view(T, N, *x.shape[1:])

        s_all   = _uf(s_all).to(self.device)     # (T, N, obs)
        a_all   = _uf(a_all).to(self.device)     # (T, N, act)
        r_all   = _uf(r_all).to(self.device)     # (T, N, 1)
        s2_all  = _uf(s2_all).to(self.device)    # (T, N, obs)
        term_all= _uf(term_all).to(self.device)  # (T, N, 1)
        trunc_all=_uf(trunc_all).to(self.device) # (T, N, 1)
        v_all   = _uf(v_all).to(self.device)     # (T, N, 1)
        lp_all  = _uf(lp_all).to(self.device)    # (T, N, 1)

        # Bootstrap last value
        with torch.no_grad():
            last_obs = s2_all[-1]                      # (N, obs)
            last_val = self.fast_critic(last_obs)       # (N, 1)
            last_done = (term_all[-1] | trunc_all[-1]).float()  # (N, 1)

        # ---- GAE advantage estimation --------------------------------
        advantages = torch.zeros_like(r_all)  # (T, N, 1)
        gae = torch.zeros(N, 1, device=self.device)
        for t in reversed(range(T)):
            next_val   = last_val if t == T - 1 else v_all[t + 1]
            next_done  = last_done if t == T - 1 else (term_all[t + 1] | trunc_all[t + 1]).float()
            # not_terminated only (truncated → bootstrap; terminated → absorb)
            not_term   = (~term_all[t]).float()
            delta      = r_all[t] + self._gamma * next_val * not_term - v_all[t]
            gae        = delta + self._gamma * self._gae_lambda * (1.0 - (term_all[t] | trunc_all[t]).float()) * gae
            advantages[t] = gae

        returns = advantages + v_all   # (T, N, 1)

        # Flatten to (T*N, dim) for mini-batch sampling
        s_flat   = s_all.view(T * N, -1)
        a_flat   = a_all.view(T * N, -1)
        lp_flat  = lp_all.view(T * N, 1)
        v_flat   = v_all.view(T * N, 1)
        adv_flat = advantages.view(T * N, 1)
        ret_flat = returns.view(T * N, 1)

        # Normalise advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Reg actor (meta warm-start)
        reg_actor = self.meta_actor if self._meta_warmup else None

        total_samples = T * N
        total_pg_loss = total_vf_loss = total_ent = 0.0
        n_updates = 0

        for _ in range(self._ppo_epochs):
            perm = torch.randperm(total_samples, device=self.device)
            # Compute an effective minibatch size at runtime. This prevents
            # configurations where `mini_batch_size` is larger than the total
            # collected samples (T*N). We follow the rsl_rl convention of
            # splitting into `num_mini_batches` if needed.
            num_mini_batches_cfg = int(self.cfg.get("num_mini_batches", 4))
            effective_mb = min(self._mini_batch, max(1, total_samples // num_mini_batches_cfg))
            for start in range(0, total_samples, effective_mb):
                idx = perm[start: start + effective_mb]
                if len(idx) < 2:
                    continue

                obs_b   = s_flat[idx]
                act_b   = a_flat[idx]
                old_lp  = lp_flat[idx]
                ret_b   = ret_flat[idx]
                adv_b   = adv_flat[idx]

                # ---- Actor loss (clipped surrogate + entropy) --------
                new_lp, entropy = self.fast_actor.evaluate(obs_b, act_b)
                ratio    = (new_lp - old_lp).exp()
                surr1    = ratio * adv_b
                surr2    = ratio.clamp(1 - self._clip_ratio, 1 + self._clip_ratio) * adv_b
                pg_loss  = -torch.min(surr1, surr2).mean()
                ent_loss = -entropy.mean()

                # BC warm-start: WD loss on (mu, std) against meta actor
                bc_loss = torch.tensor(0.0, device=self.device)
                if reg_actor is not None and self._lambda_reg > 0 and self._task_step < self._warmstep:
                    with torch.no_grad():
                        meta_mu, meta_std = reg_actor.get_dist_params(obs_b)
                    fast_mu, fast_std = self.fast_actor.get_dist_params(obs_b)
                    bc_loss = torch.mean(
                        torch.square(fast_mu - meta_mu).sum(-1) +
                        torch.square(fast_std - meta_std).sum(-1)
                    )

                actor_loss = pg_loss + self._entropy_coef * ent_loss + self._lambda_reg * bc_loss

                self.fast_actor_opt.zero_grad()
                actor_loss.backward()
                if self._grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.fast_actor.parameters(), self._grad_clip)
                self.fast_actor_opt.step()

                # ---- Value loss (clipped) ----------------------------
                new_val = self.fast_critic(obs_b)
                old_val = v_flat[idx]
                vf_unclip = F.mse_loss(new_val, ret_b)
                val_clipped = old_val + (new_val - old_val).clamp(-self._clip_ratio, self._clip_ratio)
                vf_clip   = F.mse_loss(val_clipped, ret_b)
                vf_loss   = self._value_coef * torch.max(vf_unclip, vf_clip)

                self.fast_critic_opt.zero_grad()
                vf_loss.backward()
                if self._grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.fast_critic.parameters(), self._grad_clip)
                self.fast_critic_opt.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent     += (-ent_loss.item())
                n_updates     += 1

        self._update_count += 1
        if n_updates > 0:
            self.track_data("Fast/pg_loss",    total_pg_loss / n_updates)
            self.track_data("Fast/value_loss", total_vf_loss / n_updates)
            self.track_data("Fast/entropy",    total_ent     / n_updates)
            self.track_data("Fast/returns_mean", ret_flat.mean().item())
            self.track_data("Fast/advantages_mean", adv_flat.mean().item())

    # ------------------------------------------------------------------
    # Meta-learner distillation
    # ------------------------------------------------------------------

    def _train_meta(self) -> None:
        """Distil Fast PPO actor/critic into Meta PPO actor/critic.

        Actor distillation: WD loss on ``(mu, std)`` of the Gaussian policy.
        Critic distillation: MSE loss ``V_meta(s) ≈ V_fast(s)``.

        Both Fast and Meta use ``PPOActor`` / ``PPOCritic``, so
        ``get_dist_params`` returns comparable ``(mu, std)`` pairs.
        """
        batch_size = self.cfg.get("mini_batch_size", 1024)
        if len(self.memory_meta) < batch_size:
            print("[FAME]   Meta buffer too small to train; skipping.")
            return

        # Re-init optimisers to restore base lr
        self.meta_actor_opt  = Adam(self.meta_actor.parameters(),  lr=self.cfg["lr_meta"])
        self.meta_critic_opt = Adam(self.meta_critic.parameters(), lr=self.cfg["lr_meta"])
        self.meta_scheduler        = ExponentialLR(self.meta_actor_opt,  gamma=self.cfg["meta_lr_scheduler_gamma"])
        self.meta_critic_scheduler = ExponentialLR(self.meta_critic_opt, gamma=self.cfg["meta_lr_scheduler_gamma"])

        u_steps   = max(1, len(self.memory_meta) // batch_size - 1)
        f2m_ready = len(self.memory_fast2meta) >= batch_size

        for epoch in range(self.cfg["epoch_meta"]):
            for i in range(u_steps):
                # --- Old-task WD loss (meta buffer) ----------------------
                (states_meta, _) = self.memory_meta.sample(
                    names=self._meta_names, batch_size=batch_size
                )[0]
                states_meta = states_meta.to(self.device)
                with torch.no_grad():
                    fast_mu, fast_std = self.fast_actor.get_dist_params(states_meta)
                meta_mu, meta_std = self.meta_actor.get_dist_params(states_meta)
                meta_loss = torch.mean(
                    torch.square(meta_mu - fast_mu).sum(-1) +
                    torch.square(meta_std - fast_std).sum(-1)
                )

                # --- Current-task WD loss (fast2meta buffer) -------------
                if f2m_ready and (i % max(1, u_steps) == 0):
                    (states_fast, _) = self.memory_fast2meta.sample(
                        names=self._meta_names, batch_size=batch_size
                    )[0]
                    states_fast = states_fast.to(self.device)
                    with torch.no_grad():
                        curr_mu, curr_std = self.fast_actor.get_dist_params(states_fast)
                    m2_mu, m2_std = self.meta_actor.get_dist_params(states_fast)
                    current_loss = torch.mean(
                        torch.square(m2_mu - curr_mu).sum(-1) +
                        torch.square(m2_std - curr_std).sum(-1)
                    )
                    actor_loss = meta_loss + current_loss
                else:
                    actor_loss = meta_loss

                self.meta_actor_opt.zero_grad()
                actor_loss.backward()
                self.meta_actor_opt.step()

            # --- Critic distillation: V_meta(s) ≈ V_fast(s) -------------
            (s_m, _) = self.memory_meta.sample(
                names=self._meta_names, batch_size=batch_size
            )[0]
            s_m = s_m.to(self.device)
            with torch.no_grad():
                fast_v = self.fast_critic(s_m)   # (B, 1)
            meta_v = self.meta_critic(s_m)        # (B, 1)
            critic_loss = F.mse_loss(meta_v, fast_v)
            self.meta_critic_opt.zero_grad()
            critic_loss.backward()
            self.meta_critic_opt.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"[FAME]   Epoch {epoch+1}/{self.cfg['epoch_meta']}  "
                    f"actor_loss={actor_loss.item():.3e}  "
                    f"critic_loss={critic_loss.item():.3e}  "
                    f"lr={self.meta_actor_opt.param_groups[0]['lr']:.2e}  "
                    + time.strftime("%H:%M:%S")
                )
            if (epoch + 1) % 2 == 0:
                self.meta_scheduler.step()
                self.meta_critic_scheduler.step()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _copy_fast2meta_to_meta(self) -> None:
        """Append all (states, actions) from fast2meta → meta buffer."""
        if len(self.memory_fast2meta) == 0:
            return
        (all_states, all_actions) = self.memory_fast2meta.sample_all(
            names=self._meta_names
        )[0]
        n_envs = self.memory_meta.num_envs
        n      = all_states.shape[0]
        for start in range(0, n, n_envs):
            end     = min(start + n_envs, n)
            chunk_s = all_states[start:end]
            chunk_a = all_actions[start:end]
            if chunk_s.shape[0] < n_envs:
                pad     = n_envs - chunk_s.shape[0]
                chunk_s = torch.cat([chunk_s, chunk_s[:pad]], dim=0)
                chunk_a = torch.cat([chunk_a, chunk_a[:pad]], dim=0)
            self.memory_meta.add_samples(states=chunk_s, actions=chunk_a)

    def _copy_meta_to_fast(self) -> None:
        """Warm-start Fast PPO actor/critic from Meta PPO networks.

        Since both Fast and Meta share identical ``PPOActor`` / ``PPOCritic``
        architectures, we can directly copy all weights.
        """
        self.fast_actor.load_state_dict(
            copy.deepcopy(self.meta_actor.state_dict())
        )
        self.fast_critic.load_state_dict(
            copy.deepcopy(self.meta_critic.state_dict())
        )
        # Re-init optimisers so momentum buffers don't carry over
        self.fast_actor_opt  = Adam(self.fast_actor.parameters(),  lr=self.cfg["lr_actor"])
        self.fast_critic_opt = Adam(self.fast_critic.parameters(), lr=self.cfg["lr_critic"])

    def _reset_fast_learner(self) -> None:
        """Re-initialise Fast-Learner PPO networks with random weights."""
        hidden = self.cfg["hidden_dims"]
        act    = self.cfg["activation"]
        self.fast_actor  = PPOActor(self.obs_dim, self.action_dim, hidden, activation=act).to(self.device)
        self.fast_critic = PPOCritic(self.obs_dim, hidden, activation=act).to(self.device)
        self.fast_actor_opt  = Adam(self.fast_actor.parameters(),  lr=self.cfg["lr_actor"])
        self.fast_critic_opt = Adam(self.fast_critic.parameters(), lr=self.cfg["lr_critic"])
        self.checkpoint_modules["fast_actor"]  = self.fast_actor
        self.checkpoint_modules["fast_critic"] = self.fast_critic
