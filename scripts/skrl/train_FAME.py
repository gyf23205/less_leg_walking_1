# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Train the FAME continual-RL agent across a sequence of IsaacLab environments.

The script mirrors ``train_TDCLS.py`` for the AppLauncher / Hydra setup,
but replaces the SequentialTrainer loop with a custom continual-learning
loop that:
  1. Runs one environment for ``--switch_steps`` timesteps.
  2. At the switch point, calls ``agent.end_of_env_segment()`` (distils
     Fast → Meta) and then ``agent.on_env_switch()`` (detects best init).
  3. Recreates the IsaacLab environment for the next task.
  4. Repeats until all tasks have been visited.

Command-line examples
---------------------
# 2-task sequence: flat → rough (each for 100 k steps, vectorised 4096 envs)
python scripts/skrl/train_FAME.py \\
    --tasks Less-Leg-Flat-Walking-Direct-v1,Less-AnymalC-Flat-Walking-Direct-v1 \\
    --switch_steps 100000 \\
    --headless

# 3-task cyclic sequence with explicit per-task steps override
python scripts/skrl/train_FAME.py \\
    --tasks Less-AnymalC-Flat-Walking-Direct-v1,Less-Leg-Flat-Walking-Direct-v1,Less-AnymalC-Rough-Walking-Direct-v1 \\
    --switch_steps 200000 \\
    --num_envs 2048 \\
    --headless

# Resume from checkpoint
python scripts/skrl/train_FAME.py \\
    --tasks Less-Leg-Flat-Walking-Direct-v1,Less-Leg-Rough-Walking-Direct-v1 \\
    --switch_steps 100000 \\
    --checkpoint logs/skrl/FAME/2026-03-22_12-00-00_FAME/checkpoint.pt \\
    --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train FAME continual-RL agent with IsaacLab.")
parser.add_argument(
    "--tasks",
    type=str,
    required=True,
    help=(
        "Comma-separated list of IsaacLab gym environment IDs to cycle through, e.g. "
        "'Less-Leg-Flat-Walking-Direct-v1,Less-AnymalC-Flat-Walking-Direct-v1'."
    ),
)
parser.add_argument(
    "--switch_steps",
    type=int,
    default=60_000,
    help=(
        "Number of environment timesteps (rows) to run per task segment. Default 60000 = "
        "2500 rollouts x 24 rollout_steps, matching the rsl_rl methods' 2500 iterations so both "
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
    help="Index into --tasks to start from (used when re-launching for the next task segment).",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Existing log directory to write into (used when re-launching for task idx > 0).",
)
parser.add_argument(
    "--global_step",
    type=int,
    default=0,
    help="Global step offset to continue TensorBoard logging across task segments.",
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
from datetime import datetime

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

# Use the CURRENT task (based on --task_idx) to load env cfg via Hydra.
# When re-launched for task N, --task_idx=N so we load the right env cfg.
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
    """Main continual-learning training loop."""

    # ------------------------------------------------------------------
    # Seeds
    # ------------------------------------------------------------------
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    if seed == -1:
        seed = random.randint(0, 10_000)
    torch.manual_seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # Logging directories
    # ------------------------------------------------------------------
    if args_cli.log_dir:
        # Resuming into an existing run directory (task_idx > 0)
        log_dir = args_cli.log_dir
    else:
        log_root = os.path.abspath(
            os.path.join("logs", "skrl", agent_cfg.get("agent", {}).get("experiment", {}).get("directory", "FAME"))
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root, f"{timestamp}_FAME")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    if args_cli.task_idx == 0:
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------
    # Parse task list and per-task settings
    # ------------------------------------------------------------------
    task_list = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
    switch_steps = args_cli.switch_steps
    device = args_cli.device if args_cli.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Task sequence ({len(task_list)} tasks): {task_list}")
    print(f"[INFO] Steps per task: {switch_steps}")

    # ------------------------------------------------------------------
    # Build first environment to get obs_dim / action_dim
    # ------------------------------------------------------------------
    env_cfg.sim.device = device
    env_cfg.seed = seed
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.log_dir = log_dir

    video_cfg = None
    if args_cli.video:
        video_cfg = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }

    env = _make_env(task_list[args_cli.task_idx], env_cfg, args_cli.num_envs, device, video_cfg)
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    num_envs = env.num_envs
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}  num_envs={num_envs}")

    # ------------------------------------------------------------------
    # Build FAME agent (skrl Agent subclass)
    # ------------------------------------------------------------------
    fame_cfg = FAME_DEFAULT_CONFIG.copy()
    if "agent" in agent_cfg:
        fame_cfg.update({k: v for k, v in agent_cfg["agent"].items() if k not in ("class", "experiment")})
    if "experiment" in agent_cfg.get("agent", {}):
        fame_cfg["experiment"].update(agent_cfg["agent"]["experiment"])
    fame_cfg["experiment"]["directory"] = log_dir

    # ------------------------------------------------------------------
    # Unified TensorBoard x-axis: rescale every add_scalar global_step from
    # env-step rows (task_step) to cumulative minibatch optimizer updates, so
    # FAME's x-axis matches the rsl_rl methods (which count it * epochs * minibatches).
    #   x = task_step * (ppo_epochs * num_mini_batches) / rollout_steps
    # With the pinned config (8 * 8 / 24) this reaches 160k updates at task_step=60000,
    # identical to the rsl_rl runs. Only the fast-actor PPO minibatch updates are counted;
    # FAME's auxiliary meta/distillation optimizer steps have no rsl_rl analog and are excluded.
    _ppo_epochs = int(fame_cfg.get("ppo_epochs", 8))
    _num_mini_batches = int(fame_cfg.get("num_mini_batches", 8))
    _rollout_steps_cfg = int(fame_cfg.get("rollout_steps", 24))
    _xaxis_multiplier = (_ppo_epochs * _num_mini_batches) / _rollout_steps_cfg
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from common.tb_xaxis import patch_writer_gradient_steps

    patch_writer_gradient_steps(_xaxis_multiplier)

    # ------------------------------------------------------------------
    # Keep a SINGLE TensorBoard event file: disable skrl's own writer so
    # Agent.init() does not open a second one under a <timestamp>_FAMEAgent
    # subfolder, then reuse this script's `writer` for the agent's track_data.
    _agent_write_interval = fame_cfg["experiment"].get("write_interval", 1008)
    if _agent_write_interval in ("auto", None):
        _agent_write_interval = 1008
    _agent_write_interval = int(_agent_write_interval)
    fame_cfg["experiment"] = dict(fame_cfg["experiment"])  # avoid mutating the shared default dict
    fame_cfg["experiment"]["write_interval"] = 0

    # ------------------------------------------------------------------
    # Build memories
    # ------------------------------------------------------------------
    # PPO rollout buffer: fixed size = rollout_steps rows × actual num_envs.
    # Lives on the same device as the agent (GPU).
    rollout_steps = fame_cfg.get("rollout_steps", 24)
    torch_device  = torch.device(device)
    cpu_device    = torch.device("cpu")

    memory_rollout   = RandomMemory(memory_size=rollout_steps, num_envs=num_envs, device=torch_device)

    # Distillation memories: CPU, num_envs=1, large capacity.
    f2m_size  = fame_cfg.get("size_fast2meta", 100_000)
    meta_size = fame_cfg.get("size_meta",      500_000)
    memory_fast2meta = RandomMemory(memory_size=f2m_size,  num_envs=1, device=cpu_device)
    memory_meta      = RandomMemory(memory_size=meta_size, num_envs=1, device=cpu_device)

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

    # Route the agent's track_data metrics into this script's single writer
    # (skrl created no writer of its own because write_interval was set to 0 above).
    agent.writer = writer
    agent.write_interval = _agent_write_interval

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        agent.load(resume_path)

    # Helper: save an additional human-friendly .pt checkpoint containing
    # model state_dicts and optimizer states. We save next to the skrl
    # checkpoint file as '<ckpt_path>.pt'. This makes it easy to load with
    # torch.load later.
    def _save_pt_checkpoint(path_base: str) -> None:
        try:
            pt_path = f"{path_base}.pt"
            data = {"models": {}, "optimizers": {}, "meta": {}}
            # model modules are registered in agent.checkpoint_modules
            try:
                for name, module in getattr(agent, "checkpoint_modules", {}).items():
                    try:
                        data["models"][name] = module.state_dict()
                    except Exception:
                        # ignore non-torch modules
                        pass
            except Exception:
                pass

            # optimizers: include common optimizer attributes if present
            for opt_name in [
                "fast_actor_opt",
                "fast_critic_opt",
                "meta_actor_opt",
                "meta_critic_opt",
            ]:
                if hasattr(agent, opt_name):
                    try:
                        data["optimizers"][opt_name] = getattr(agent, opt_name).state_dict()
                    except Exception:
                        pass

            # optionally include a small config / timestamp
            data["meta"]["saved_at"] = time.time()
            data["meta"]["cfg"] = getattr(agent, "cfg", {}) if hasattr(agent, "cfg") else {}

            torch.save(data, pt_path)
            print(f"[FAME] Wrote .pt checkpoint: {pt_path}", flush=True)
        except Exception as e:
            print(f"[FAME] Failed to write .pt checkpoint for {path_base}: {e}", flush=True)

    # No automatic relaunch: child-parent readiness handshake removed.
    # The user will re-launch the script manually for the next task.

    # Write interval (in env-step rows, not per-transition)
    write_interval = agent_cfg.get("agent", {}).get("experiment", {}).get("write_interval", 1000)

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
    # Continual-learning training loop
    # ------------------------------------------------------------------
    task_idx = args_cli.task_idx
    current_task = task_list[task_idx]

    # Notify agent about the current task.
    # task_idx == 0: first environment, no detection needed.
    # task_idx  > 0: resumed after a task switch — run detection to choose
    #               Fast / Meta / Random initialisation for the fast policy.
    is_first = (task_idx == 0)
    eval_fn  = None if is_first else _make_eval_fn(env)
    agent.on_env_switch(current_task, eval_fn=eval_fn, is_first_switch=is_first)

    obs, infos = env.reset()

    print(f"\n[FAME] ══════════════════════════════════════════")
    print(f"[FAME] Starting task {task_idx+1}/{len(task_list)}: {current_task}")
    print(f"[FAME] ══════════════════════════════════════════\n")

    task_step = 0         # steps in the current task segment
    episode_rewards = torch.zeros(num_envs, device=torch_device)
    episode_lengths = torch.zeros(num_envs, device=torch_device)
    completed_episode_rewards = []
    completed_episode_lengths = []
    # Performance counters for throughput monitoring
    _train_start_time = time.time()   # wall-clock time when the training loop begins
    _perf_last_time = time.time()
    _perf_last_step = 0

    while task_idx < len(task_list):
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

        # ---- Checkpoint -------------------------------------------------
        ckpt_interval = agent_cfg.get("agent", {}).get("experiment", {}).get("checkpoint_interval", 20_000)
        if task_step % ckpt_interval == 0:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{task_step:08d}")
            agent.save(ckpt_path)
            try:
                _save_pt_checkpoint(ckpt_path)
            except Exception:
                pass
            print(f"[FAME] Checkpoint saved: {ckpt_path}")

        # ---- Task switch ------------------------------------------------
        if task_step >= switch_steps:
            print(f"\n[FAME] Segment {task_idx+1} complete ({task_step} steps).")

            # Phase 3 & 4: distil Fast→Meta, refresh meta buffer
            agent.end_of_env_segment()

            # Advance to next task
            next_task_idx = task_idx + 1
            if next_task_idx >= len(task_list):
                print("[FAME] All tasks completed — training done.")
                break

            # We no longer perform automatic re-launching. Save a checkpoint
            # and exit so the user can manually start the next task with the
            # same logging directory. This avoids all subprocess/Kit races.
            ckpt_path = os.path.join(log_dir, f"checkpoint_task{task_idx:02d}_end")
            agent.save(ckpt_path)
            try:
                _save_pt_checkpoint(ckpt_path)
            except Exception:
                pass
            print(f"[FAME] Checkpoint saved: {ckpt_path}")
            print(
                f"[FAME] Automatic relaunch disabled. Please relaunch the script manually for task {next_task_idx+1}/{len(task_list)}: {task_list[next_task_idx]}",
                flush=True,
            )

            # Close writer/env (best-effort) and exit immediately to avoid
            # blocking on Kit/plugin shutdown. The user will re-run the
            # script manually for the next task when ready.
            try:
                writer.close()
            except Exception:
                pass
            try:
                env.close()
            except Exception:
                pass
            print("[FAME] Exiting for manual relaunch.", flush=True)
            os._exit(0)

    # ------------------------------------------------------------------
    # Final checkpoint + summary
    # ------------------------------------------------------------------
    final_ckpt = os.path.join(log_dir, "checkpoint_final")
    agent.save(final_ckpt)
    try:
        _save_pt_checkpoint(final_ckpt)
    except Exception:
        pass
    print(f"\n[FAME] Final checkpoint: {final_ckpt}")
    print(f"[FAME] Task sequence : {agent.games}")
    print(f"[FAME] Init choices  : {agent.flag_reg}")

    try:
        writer.close()
    except Exception:
        pass
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
