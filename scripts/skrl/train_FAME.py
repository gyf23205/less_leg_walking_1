# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Train the FAME continual-RL agent on a SINGLE task segment of a sequence.

The script mirrors ``train_TDCLS.py`` for the AppLauncher / Hydra setup, but
replaces the SequentialTrainer loop with a custom continual-learning segment
that:
  1. Builds the task selected by ``--task_idx`` and runs it for
     ``--switch_steps`` timesteps.
  2. Optionally loads the previous task's checkpoint (``--checkpoint``) and, on
     any task after the first, calls ``agent.on_env_switch()`` to detect the
     best Fast / Meta / Random initialisation for the new task.
  3. At the end of the segment, calls ``agent.end_of_env_segment()`` (distils
     Fast → Meta), saves a checkpoint, and exits.

Running the FULL sequence is orchestrated by ``train_FAME_CRL.py``, which
launches this script once per task (one fresh Isaac Sim process each), chaining
the end-of-task checkpoint into the next task's ``--checkpoint``. This mirrors
``scripts/rsl_rl/train_moe_CRL.py`` and avoids recreating environments in a
single process.

Command-line examples
---------------------
# Train just the first task of the default sequence
python scripts/skrl/train_FAME.py --task_idx 0 --switch_steps 100000 --headless

# Train the third task, resuming FAME state from the previous task's checkpoint
python scripts/skrl/train_FAME.py \\
    --task_idx 2 \\
    --checkpoint logs/skrl/FAME_CRL/checkpoints/task01_end \\
    --switch_steps 100000 \\
    --headless

# Usually you just run the whole sequence via the driver:
python scripts/skrl/train_FAME_CRL.py --switch_steps 100000 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
# Default task sequence, matching the order used by scripts/rsl_rl/train_moe_CRL.py
# (its ORIGINAL_TASK first, then TRAIN_TASKS). Used when --tasks is not passed.
DEFAULT_TASKS = [
    "Less-AnymalC-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Direct-v1",
    # "Less-AnymalC-Rough-Walking-Direct-v1",
    # "Less-Leg-Flat-Walking-Direct-v1",
    # "Less-AnymalC-Jump-Rough-Direct-v1",
    # "Less-Leg-Rough-Walking-Direct-v1",
]

parser = argparse.ArgumentParser(description="Train FAME continual-RL agent with IsaacLab.")
parser.add_argument(
    "--tasks",
    type=str,
    default=",".join(DEFAULT_TASKS),
    help=(
        "Comma-separated list of IsaacLab gym environment IDs to cycle through. "
        "Defaults to the hardcoded DEFAULT_TASKS sequence (same order as train_moe_CRL.py). "
        "Example override: 'Less-Leg-Flat-Walking-Direct-v1,Less-AnymalC-Flat-Walking-Direct-v1'."
    ),
)
parser.add_argument(
    "--switch_steps",
    type=int,
    default=100_000,
    help=(
        "Number of environment timesteps (rows) to run per task segment. Default 100000 = "
        "2500 rollouts x 40 rollout_steps, matching the rsl_rl methods' 2500 iterations so both "
        "reach the same unified x-axis budget of 2500 x (8 epochs x 8 minibatches) = 160k updates."
    ),
)
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel envs.")
parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to FAME checkpoint to resume from.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help="Alternative agent cfg entry-point key (default: skrl_fame_cfg_entry_point).",
)
parser.add_argument(
    "--task_idx",
    type=int,
    default=0,
    help="Index into --tasks to start the sequence from (e.g. when resuming from a checkpoint).",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Explicit log directory override. If set, ALL tasks log here; otherwise each task gets "
         "its own logs/task1/<experiment_name>/<method> directory.",
)
parser.add_argument(
    "--global_step",
    type=int,
    default=0,
    help="Global step offset to continue TensorBoard logging across task segments.",
)
parser.add_argument(
    "--end_ckpt",
    type=str,
    default=None,
    help="If set, also save the end-of-segment checkpoint to this explicit path (in addition to "
         "the per-task log dir). Used by train_FAME_CRL.py to chain checkpoints between tasks.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports (after Isaac Sim is up)."""

import os
import signal
import threading
import random
import time

import gymnasium as gym
import omni
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import less_leg_walking_1.tasks  # noqa: F401

from less_leg_walking_1.tasks.direct.less_leg_walking_1.baselines.FAME.agent import (
    FAMEAgent,
    FAME_DEFAULT_CONFIG,
)
from skrl.memories.torch import RandomMemory

# The cfg entry-point to load from the gym registry
_AGENT_CFG_KEY = args_cli.agent if args_cli.agent else "skrl_fame_cfg_entry_point"

# Load the STARTING task's env cfg via Hydra. Subsequent tasks in the sequence
# are built in-process with parse_env_cfg (see _build_task_env below).
_task_list_global = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
_CURRENT_TASK = _task_list_global[args_cli.task_idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(task_id: str, env_cfg, num_envs: int | None, device: str, video_cfg: dict | None):
    """Instantiate and wrap an IsaacLab environment."""
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if video_cfg else None)
    if video_cfg:
        env = gym.wrappers.RecordVideo(env, **video_cfg)
    env = SkrlVecEnvWrapper(env)
    return env


@hydra_task_config(_CURRENT_TASK, _AGENT_CFG_KEY)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Main continual-learning training loop (single process, all tasks)."""

    # ------------------------------------------------------------------
    # Seeds
    # ------------------------------------------------------------------
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    if seed == -1:
        seed = random.randint(0, 10_000)
    torch.manual_seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # Task list / common settings
    # ------------------------------------------------------------------
    task_list = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
    switch_steps = args_cli.switch_steps
    device = args_cli.device if args_cli.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)
    cpu_device = torch.device("cpu")
    start_idx = args_cli.task_idx

    print(f"[INFO] Task sequence ({len(task_list)} tasks): {task_list}")
    print(f"[INFO] Steps per task: {switch_steps}")
    print(f"[INFO] Starting from task index {start_idx}: {task_list[start_idx]}")

    # ------------------------------------------------------------------
    # Per-task log directory: mirror the other methods by keying on each task's
    # rsl_rl experiment_name -> logs/task1/<experiment_name>/<method>.
    # ------------------------------------------------------------------
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    _method_name = agent_cfg.get("agent", {}).get("experiment", {}).get("directory", "FAME")

    def _task_log_dir(task_id: str) -> str:
        if args_cli.log_dir:
            return args_cli.log_dir
        try:
            _rsl_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
            task_folder = getattr(_rsl_cfg, "experiment_name", None) or task_id
        except Exception:
            task_folder = task_id
        return os.path.abspath(os.path.join("logs", "task1", task_folder, _method_name))

    # ------------------------------------------------------------------
    # FAME agent cfg (shared across all task segments)
    # ------------------------------------------------------------------
    fame_cfg = FAME_DEFAULT_CONFIG.copy()
    if "agent" in agent_cfg:
        fame_cfg.update({k: v for k, v in agent_cfg["agent"].items() if k not in ("class", "experiment")})
    if "experiment" in agent_cfg.get("agent", {}):
        fame_cfg["experiment"].update(agent_cfg["agent"]["experiment"])

    # ------------------------------------------------------------------
    # Unified TensorBoard x-axis: rescale every add_scalar global_step from
    # env-step rows (task_step) to cumulative minibatch optimizer updates, so
    # FAME's x-axis matches the rsl_rl methods (which count it * epochs * minibatches).
    #   x = task_step * (ppo_epochs * num_mini_batches) / rollout_steps
    # Only the fast-actor PPO minibatch updates are counted; FAME's auxiliary
    # meta/distillation optimizer steps have no rsl_rl analog and are excluded.
    _ppo_epochs = int(fame_cfg.get("ppo_epochs", 8))
    _num_mini_batches = int(fame_cfg.get("num_mini_batches", 8))
    _rollout_steps_cfg = int(fame_cfg.get("rollout_steps", 24))
    _xaxis_multiplier = (_ppo_epochs * _num_mini_batches) / _rollout_steps_cfg
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from common.tb_xaxis import patch_writer_gradient_steps

    patch_writer_gradient_steps(_xaxis_multiplier)

    # Disable skrl's own writer (write_interval=0); we route the agent's
    # track_data into a per-task SummaryWriter created inside the loop.
    _agent_write_interval = fame_cfg["experiment"].get("write_interval", 1008)
    if _agent_write_interval in ("auto", None):
        _agent_write_interval = 1008
    _agent_write_interval = int(_agent_write_interval)
    fame_cfg["experiment"] = dict(fame_cfg["experiment"])  # avoid mutating the shared default dict
    fame_cfg["experiment"]["write_interval"] = 0

    write_interval = agent_cfg.get("agent", {}).get("experiment", {}).get("write_interval", 1000)
    ckpt_interval = agent_cfg.get("agent", {}).get("experiment", {}).get("checkpoint_interval", 20_000)
    rollout_steps = fame_cfg.get("rollout_steps", 24)

    # Filled lazily when the first environment is built.
    agent = None
    resolved_num_envs = args_cli.num_envs

    # ------------------------------------------------------------------
    # Checkpoint helper: save a human-friendly .pt (model + optimizer states)
    # alongside the skrl checkpoint at '<ckpt_path>.pt'.
    # ------------------------------------------------------------------
    def _save_pt_checkpoint(path_base: str) -> None:
        try:
            pt_path = f"{path_base}.pt"
            data = {"models": {}, "optimizers": {}, "meta": {}}
            try:
                for name, module in getattr(agent, "checkpoint_modules", {}).items():
                    try:
                        data["models"][name] = module.state_dict()
                    except Exception:
                        pass
            except Exception:
                pass

            for opt_name in [
                "fast_actor_opt",
                "fast_critic_opt",
                "meta_actor_opt",
            ]:
                if hasattr(agent, opt_name):
                    try:
                        data["optimizers"][opt_name] = getattr(agent, opt_name).state_dict()
                    except Exception:
                        pass

            # --- FAME continual-learning state that skrl checkpoints miss -----
            # The meta buffer + bookkeeping must survive the per-process CRL
            # boundary, otherwise meta distillation restarts empty each task.
            try:
                mm = getattr(agent, "memory_meta", None)
                if mm is not None and len(mm) > 0:
                    n = len(mm)
                    data["meta_buffer"] = {
                        "states":  mm.tensors_view["states"][:n].clone().cpu(),
                        "actions": mm.tensors_view["actions"][:n].clone().cpu(),
                    }
            except Exception as e:
                print(f"[FAME] Warning: could not serialize meta buffer: {e}", flush=True)

            data["fame"] = {
                "_meta_trained": bool(getattr(agent, "_meta_trained", False)),
                "games":    list(getattr(agent, "games", [])),
                "flag_reg": list(getattr(agent, "flag_reg", [])),
            }

            data["meta"]["saved_at"] = time.time()
            data["meta"]["cfg"] = getattr(agent, "cfg", {}) if hasattr(agent, "cfg") else {}

            torch.save(data, pt_path)
            print(f"[FAME] Wrote .pt checkpoint: {pt_path}", flush=True)
        except Exception as e:
            print(f"[FAME] Failed to write .pt checkpoint for {path_base}: {e}", flush=True)

    def _load_fame_state(ckpt_base: str) -> None:
        """Restore the meta buffer + FAME bookkeeping from '<ckpt_base>.pt'.

        Called right after ``agent.load()`` (which restores the networks) so
        meta distillation continues to accumulate across task processes.
        """
        pt_path = f"{ckpt_base}.pt"
        if not os.path.isfile(pt_path):
            print(f"[FAME] No .pt sidecar for meta state at {pt_path}; "
                  f"meta buffer starts empty.", flush=True)
            return
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[FAME] Failed to load meta state from {pt_path}: {e}", flush=True)
            return

        # Restore meta buffer contents. NOTE: skrl's batched add_samples path
        # (num_envs==1) increments memory_index once per named tensor, which
        # corrupts the length, so we write the underlying storage directly.
        mb = data.get("meta_buffer")
        mm = getattr(agent, "memory_meta", None)
        if mb is not None and mm is not None:
            try:
                mm.reset()
                dev = mm.device
                cap = mm.memory_size
                s = mb["states"].to(dev)
                a = mb["actions"].to(dev)
                n = min(s.shape[0], cap)
                if n < s.shape[0]:
                    print(f"[FAME] Meta buffer ({s.shape[0]}) exceeds capacity ({cap}); "
                          f"keeping the last {n} samples.", flush=True)
                    s, a = s[-n:], a[-n:]
                # tensors[name] shape: (memory_size, num_envs=1, feat)
                mm.tensors["states"][:n, 0].copy_(s)
                mm.tensors["actions"][:n, 0].copy_(a)
                if n >= cap:
                    mm.memory_index = 0
                    mm.filled = True
                else:
                    mm.memory_index = n
                    mm.filled = False
                mm.env_index = 0
                print(f"[FAME] Restored meta buffer: {len(mm)} samples.", flush=True)
            except Exception as e:
                print(f"[FAME] Failed to restore meta buffer: {e}", flush=True)

        # Restore bookkeeping. A prior distillation implies the meta actor holds
        # past-task knowledge, so the old-knowledge anchor should be active.
        fame = data.get("fame", {})
        agent._meta_trained = bool(fame.get("_meta_trained", True))
        if fame.get("games"):
            agent.games = list(fame["games"])
        if fame.get("flag_reg"):
            agent.flag_reg = list(fame["flag_reg"])

    # ------------------------------------------------------------------
    # Detection eval function (used by on_env_switch for task_idx > 0)
    # ------------------------------------------------------------------
    def _make_eval_fn(env_handle):
        """Return eval_fn(actor, num_steps) -> mean_episode_return.

        Runs ``actor`` deterministically in ``env_handle`` for ``num_steps``
        env-step rows (across all parallel envs) and returns the mean
        per-episode return of all episodes that completed.
        """
        def eval_fn(actor, num_steps: int) -> float:
            actor.eval()
            obs_e, _ = env_handle.reset()
            ep_rewards = torch.zeros(env_handle.num_envs, device=torch_device)
            completed = []
            # NOTE: use no_grad (not inference_mode) here. The env creates/clones persistent
            # tensors during _get_observations/_get_rewards/_reset_idx; under inference_mode
            # those become "inference tensors" that then break later in-place updates in the
            # normal training loop. no_grad gives the same gradient-free eval without that side
            # effect (and matches what agent.act() / the main loop already use).
            with torch.no_grad():
                for _ in range(num_steps):
                    dist    = actor(obs_e)
                    actions = dist.mean          # deterministic
                    obs_e, rewards, terminated, truncated, _ = env_handle.step(actions)
                    ep_rewards += rewards.squeeze(-1) if rewards.dim() > 1 else rewards
                    done = (terminated | truncated)
                    done = done.squeeze(-1) if done.dim() > 1 else done
                    if done.any():
                        completed.extend(ep_rewards[done].tolist())
                        ep_rewards[done] = 0.0
            actor.train()
            return float(sum(completed) / len(completed)) if completed else 0.0
        return eval_fn

    # ------------------------------------------------------------------
    # Env builder for a task segment. The starting task reuses the Hydra-injected
    # env_cfg (so CLI/Hydra overrides apply); later tasks are parsed fresh.
    # num_envs is pinned to the first task's value so the agent's fixed-size
    # rollout buffer stays valid across every segment.
    # ------------------------------------------------------------------
    def _build_task_env(task_id: str, log_dir: str, use_injected: bool):
        nonlocal resolved_num_envs
        cfg = env_cfg if use_injected else parse_env_cfg(task_id, device=device, num_envs=resolved_num_envs)
        cfg.sim.device = device
        cfg.seed = seed
        if resolved_num_envs is None:
            resolved_num_envs = cfg.scene.num_envs
        cfg.scene.num_envs = resolved_num_envs
        cfg.log_dir = log_dir

        video_cfg = None
        if args_cli.video:
            video_cfg = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
        return _make_env(task_id, cfg, resolved_num_envs, device, video_cfg)

    # ------------------------------------------------------------------
    # Run a SINGLE task segment (the one selected by --task_idx) and exit.
    # The full sequence is orchestrated by train_FAME_CRL.py, which launches
    # this script once per task in a fresh process.
    # ------------------------------------------------------------------
    for task_idx in (start_idx,):
        current_task = task_list[task_idx]
        is_first_in_run = (task_idx == start_idx)   # first task built in THIS process
        is_global_first = (task_idx == 0)           # very first task of the sequence (no detection)

        # -- Per-task logging --
        log_dir = _task_log_dir(current_task)
        os.makedirs(log_dir, exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[INFO] Logging to: {log_dir}")

        # -- Build this task's environment --
        env = _build_task_env(current_task, log_dir, use_injected=is_first_in_run)
        num_envs = env.num_envs

        # -- Build the FAME agent once (on the first segment of this run) --
        if agent is None:
            obs_dim = env.observation_space.shape[-1]
            action_dim = env.action_space.shape[-1]
            print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}  num_envs={num_envs}")

            memory_rollout = RandomMemory(memory_size=rollout_steps, num_envs=num_envs, device=torch_device)
            f2m_size = fame_cfg.get("size_fast2meta", 100_000)
            meta_size = fame_cfg.get("size_meta", 500_000)
            memory_fast2meta = RandomMemory(memory_size=f2m_size, num_envs=1, device=cpu_device)
            memory_meta = RandomMemory(memory_size=meta_size, num_envs=1, device=cpu_device)

            agent = FAMEAgent(
                models={},
                memory=memory_rollout,
                memory_fast2meta=memory_fast2meta,
                memory_meta=memory_meta,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=torch_device,
                cfg=fame_cfg,
            )
            agent.init()

            if args_cli.checkpoint:
                resume_path = retrieve_file_path(args_cli.checkpoint)
                print(f"[INFO] Loading checkpoint: {resume_path}")
                agent.load(resume_path)
                # Restore the meta buffer + FAME bookkeeping from the .pt sidecar
                # so meta distillation continues across the per-process boundary.
                _load_fame_state(args_cli.checkpoint)

        # Route the agent's track_data metrics into this task's writer.
        fame_cfg["experiment"]["directory"] = log_dir
        agent.writer = writer
        agent.write_interval = _agent_write_interval

        # -- Notify agent about the current task --
        # Global first task ever: no detection. Otherwise run detection to pick
        # Fast / Meta / Random initialisation for the fast policy on the new task.
        eval_fn = None if is_global_first else _make_eval_fn(env)
        agent.on_env_switch(current_task, eval_fn=eval_fn, is_first_switch=is_global_first)

        obs, infos = env.reset()

        print(f"\n[FAME] ══════════════════════════════════════════")
        print(f"[FAME] Starting task {task_idx+1}/{len(task_list)}: {current_task}")
        print(f"[FAME] ══════════════════════════════════════════\n")

        # -- Per-task counters (task_step resets to 0 so each task logs from x=0) --
        task_step = 0
        episode_rewards = torch.zeros(num_envs, device=torch_device)
        episode_lengths = torch.zeros(num_envs, device=torch_device)
        completed_episode_rewards = []
        completed_episode_lengths = []
        _train_start_time = time.time()
        _perf_last_time = time.time()
        _perf_last_step = 0

        # -- Segment training loop --
        while task_step < switch_steps:
            # ---- Act (skrl-style: returns (actions, None, {})) --------------
            actions, _, _ = agent.act(obs, timestep=task_step, timesteps=switch_steps)

            # ---- Action diagnostics ---------------------------------------
            try:
                # Coerce actions to a torch tensor for diagnostics (supports numpy, lists, torch)
                a_tensor = None
                try:
                    if isinstance(actions, torch.Tensor):
                        a_tensor = actions
                    else:
                        # try numpy array
                        import numpy as _np

                        if _np and hasattr(actions, "__array__"):
                            a_tensor = torch.as_tensor(_np.asarray(actions), device=torch_device)
                        else:
                            # fallback: try converting via torch.tensor
                            a_tensor = torch.tensor(actions, device=torch_device)
                except Exception:
                    # Last-resort: skip diagnostics if conversion fails
                    a_tensor = None

                if a_tensor is not None:
                    a_mean = float(a_tensor.mean().item())
                    a_std = float(a_tensor.std().item())
                    a_min = float(a_tensor.min().item())
                    a_max = float(a_tensor.max().item())
                    # NaN/Inf checks
                    has_nan = bool(torch.isnan(a_tensor).any())
                    has_inf = bool(torch.isinf(a_tensor).any())
                    # Log to TB if writer available
                    try:
                        writer.add_scalar("Action/mean", a_mean, task_step)
                        writer.add_scalar("Action/std", a_std, task_step)
                        writer.add_scalar("Action/min", a_min, task_step)
                        writer.add_scalar("Action/max", a_max, task_step)
                    except Exception:
                        pass
                    # Print a brief summary every 1000 steps so the user sees action stats
                    if task_step % 1000 == 0:
                        print(f"[DEBUG] actions mean={a_mean:.4f} std={a_std:.4f} min={a_min:.4f} max={a_max:.4f} nan={has_nan} inf={has_inf}")
                    if has_nan or has_inf:
                        print("[ERROR] NaN or Inf found in actions! Clamping to zero for safety.")
                        # Clamp to zero to avoid simulator issues; keep running to collect diagnostics
                        actions = torch.zeros_like(a_tensor)
                else:
                    # Could not coerce actions => print type for debugging
                    if task_step % 1000 == 0:
                        print(f"[DEBUG] actions of unexpected type: {type(actions)} - skipping numeric stats")
            except Exception as e:
                print(f"[DEBUG] Failed to compute action stats: {e}")

            # ---- Step -------------------------------------------------------
            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            # ---- Store (skrl record_transition + post_interaction) ----------
            agent.record_transition(
                obs, actions, rewards, next_obs,
                terminated, truncated, infos,
                timestep=task_step,
                timesteps=switch_steps,
            )
            agent.post_interaction(
                timestep=task_step,
                timesteps=switch_steps,
            )

            # ---- Episode tracking -------------------------------------------
            episode_rewards += rewards.squeeze(-1) if rewards.dim() > 1 else rewards
            episode_lengths += 1
            done_mask = (terminated | truncated).squeeze(-1) if (terminated | truncated).dim() > 1 else (terminated | truncated)
            if done_mask.any():
                completed_episode_rewards.extend(episode_rewards[done_mask].tolist())
                completed_episode_lengths.extend(episode_lengths[done_mask].tolist())
                episode_rewards[done_mask] = 0.0
                episode_lengths[done_mask] = 0.0

            obs = next_obs
            task_step += 1

            # ---- Console progress -------------------------------------------
            if task_step % 1000 == 0:
                rollout_ptr = agent._rollout_ptr
                # throughput (env-step rows / sec) and samples/sec (rows * envs)
                now = time.time()
                elapsed = now - _perf_last_time if now > _perf_last_time else 1e-6
                steps_done = task_step - _perf_last_step
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0.0
                samples_per_sec = steps_per_sec * max(1, num_envs)
                # total wall-clock time since loop started
                total_elapsed_s = now - _train_start_time
                total_elapsed_min = total_elapsed_s / 60.0
                total_elapsed_h = total_elapsed_s / 3600.0
                print(
                    f"[FAME] task_step={task_step:7d}  "
                    f"rollout={rollout_ptr:3d}/{agent._rollout_steps}  "
                    f"updates={agent._update_count:6d}  "
                    f"{steps_per_sec:6.1f} rows/s ({samples_per_sec:7.1f} samples/s)  "
                    f"elapsed={total_elapsed_min:.1f}min"
                )
                try:
                    writer.add_scalar("Perf/rows_per_sec", steps_per_sec, task_step)
                    writer.add_scalar("Perf/samples_per_sec", samples_per_sec, task_step)
                    writer.add_scalar("Perf/total_fps", samples_per_sec, task_step)  # rsl_rl-compatible name
                    writer.add_scalar("Perf/walltime_min", total_elapsed_min, task_step)
                    writer.add_scalar("Perf/walltime_hours", total_elapsed_h, task_step)
                except Exception:
                    pass
                _perf_last_time = now
                _perf_last_step = task_step

            # ---- TensorBoard logging ----------------------------------------
            if task_step % write_interval == 0 and completed_episode_rewards:
                mean_rew = sum(completed_episode_rewards) / len(completed_episode_rewards)
                mean_len = sum(completed_episode_lengths) / len(completed_episode_lengths)
                # rsl_rl-compatible tag names (OnPolicyRunner.log) so curves overlay
                writer.add_scalar("Train/mean_reward", mean_rew, task_step)
                writer.add_scalar("Train/mean_episode_length", mean_len, task_step)
                writer.add_scalar("Train/task_index", task_idx, task_step)
                # Wall-clock variants (x = elapsed seconds), mirroring rsl_rl's
                # "Train/mean_reward/time" and "Train/mean_episode_length/time".
                _walltime_s = time.time() - _train_start_time
                writer.add_scalar("Train/mean_reward/time", mean_rew, _walltime_s)
                writer.add_scalar("Train/mean_episode_length/time", mean_len, _walltime_s)
                completed_episode_rewards.clear()
                completed_episode_lengths.clear()

            # Log env info (from infos["log"] if available). Mirror rsl_rl's key scheme:
            # keys already containing "/" (e.g. "Episode_Reward/track_lin_vel_xy_exp") are
            # logged verbatim; bare keys go under "Episode/".
            if "log" in infos:
                for k, v in infos["log"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        tag = k if "/" in k else f"Episode/{k}"
                        writer.add_scalar(tag, v.item(), task_step)

            # ---- Periodic checkpoint ----------------------------------------
            if task_step % ckpt_interval == 0:
                ckpt_path = os.path.join(log_dir, f"checkpoint_{task_step:08d}")
                agent.save(ckpt_path)
                try:
                    _save_pt_checkpoint(ckpt_path)
                except Exception:
                    pass
                print(f"[FAME] Checkpoint saved: {ckpt_path}")

        # ---- End of segment: distil Fast->Meta, checkpoint, exit ----
        print(f"\n[FAME] Segment for task {task_idx+1}/{len(task_list)} complete ({task_step} steps).")
        agent.end_of_env_segment()

        is_last = (task_idx == len(task_list) - 1)
        ckpt_name = "checkpoint_final" if is_last else f"checkpoint_task{task_idx:02d}_end"
        ckpt_path = os.path.join(log_dir, ckpt_name)
        agent.save(ckpt_path)
        try:
            _save_pt_checkpoint(ckpt_path)
        except Exception:
            pass
        print(f"[FAME] Checkpoint saved: {ckpt_path}")

        # Also save to the explicit chaining path so the driver can hand this
        # checkpoint to the next task's --checkpoint without knowing the log dir.
        if args_cli.end_ckpt:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.end_ckpt)), exist_ok=True)
            agent.save(args_cli.end_ckpt)
            try:
                _save_pt_checkpoint(args_cli.end_ckpt)
            except Exception:
                pass
            print(f"[FAME] End-of-segment checkpoint saved: {args_cli.end_ckpt}")

        try:
            writer.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Segment done — this process handles exactly one task, then exits.
    # ------------------------------------------------------------------
    print(f"\n[FAME] Task {start_idx+1}/{len(task_list)} ({task_list[start_idx]}) done.")
    if agent is not None:
        print(f"[FAME] Task sequence : {agent.games}")
        print(f"[FAME] Init choices  : {agent.flag_reg}")

    try:
        env.close()
    except Exception:
        pass

    print("[FAME] Exiting now.", flush=True)
    # Ensure the process is terminated unconditionally. os._exit should
    # normally be sufficient, but on some platforms/process supervisors it
    # may not kill all native threads or child processes. Send SIGKILL to
    # our own PID as a last-resort hard kill.
    try:
        os._exit(0)
    finally:
        try:
            os.kill(os.getpid(), signal.SIGKILL)
        except Exception:
            # If even that fails, fall back to SIGTERM for best-effort.
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                pass


if __name__ == "__main__":
    main()
