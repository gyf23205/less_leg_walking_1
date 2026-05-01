# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate / visualise a saved FAME checkpoint.

Usage examples
--------------
# Visualise in the simulator (interactive):
python scripts/skrl/play_FAME.py \
    --task Less-AnymalC-Rough-Walking-Direct-v1 \
    --checkpoint logs/skrl/FAME/<run_dir>/checkpoint_final \
    --num_envs 16

# Record a video:
python scripts/skrl/play_FAME.py \
    --task Less-AnymalC-Rough-Walking-Direct-v1 \
    --checkpoint logs/skrl/FAME/<run_dir>/checkpoint_final \
    --num_envs 1 \
    --video --video_length 500
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a FAME checkpoint with IsaacLab.")
parser.add_argument("--task",       type=str,  required=True,  help="Gym env ID, e.g. Less-AnymalC-Rough-Walking-Direct-v1.")
parser.add_argument("--checkpoint", type=str,  default=None,   help="Path to FAME checkpoint directory (passed to agent.load()).")
parser.add_argument("--num_envs",   type=int,  default=None,   help="Override number of parallel environments.")
parser.add_argument("--seed",       type=int,  default=42,     help="Random seed.")
parser.add_argument("--video",         action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length",  type=int, default=500,  help="Number of steps to record.")
parser.add_argument("--real_time",     action="store_true", default=False, help="Slow down to real-time.")
parser.add_argument("--disable_fabric",action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports (after Isaac Sim is up)."""

import os
import time

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import less_leg_walking_1.tasks  # noqa: F401

from less_leg_walking_1.tasks.direct.less_leg_walking_1.baselines.FAME.agent import (
    FAMEAgent,
    FAME_DEFAULT_CONFIG,
)
from skrl.memories.torch import RandomMemory

_AGENT_CFG_KEY = "skrl_fame_cfg_entry_point"


@hydra_task_config(args_cli.task, _AGENT_CFG_KEY)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Load a FAME checkpoint and run evaluation."""

    # ── Environment setup ──────────────────────────────────────────────
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # get dt before wrapping
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "skrl", "FAME", "play_videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video to:", video_kwargs["video_folder"])
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env)

    device         = env.device
    num_envs       = env.num_envs
    obs_space      = env.observation_space
    act_space      = env.action_space

    # ── Build FAME agent (no replay memory needed for eval) ────────────
    fame_cfg = dict(FAME_DEFAULT_CONFIG)
    if "agent" in agent_cfg:
        fame_cfg.update({k: v for k, v in agent_cfg["agent"].items()
                         if k not in ("class", "experiment")})
    # disable TensorBoard / checkpointing during play
    fame_cfg["experiment"] = {
        "directory": "", "experiment_name": "",
        "write_interval": 0, "checkpoint_interval": 0,
        "store_separately": False, "wandb": False, "wandb_kwargs": {},
    }

    # Tiny memories — not used for training, just to satisfy the API
    dummy_fast = RandomMemory(memory_size=2, num_envs=num_envs, device=device)
    dummy_f2m  = RandomMemory(memory_size=2, num_envs=1,        device=device)
    dummy_meta = RandomMemory(memory_size=2, num_envs=1,        device=device)

    agent = FAMEAgent(
        models={},
        memory=dummy_fast,
        memory_fast2meta=dummy_f2m,
        memory_meta=dummy_meta,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        cfg=fame_cfg,
    )
    agent.init()

    # ── Load checkpoint ────────────────────────────────────────────────
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        # Auto-find the latest checkpoint in logs/skrl/FAME/
        from isaaclab_tasks.utils import get_checkpoint_path
        log_root = os.path.abspath(os.path.join("logs", "skrl", "FAME"))
        resume_path = get_checkpoint_path(log_root, other_dirs=["checkpoints"])

    print(f"[INFO] Loading FAME checkpoint: {resume_path}")
    agent.load(resume_path)
    agent.set_mode("eval")

    # ── Evaluation loop ────────────────────────────────────────────────
    print(f"[INFO] Running evaluation on: {args_cli.task}  ({num_envs} envs)")
    obs, _ = env.reset()
    timestep = 0
    episode_rewards = torch.zeros(num_envs, device=device)
    ep_count = 0

    while simulation_app.is_running():
        t_start = time.time()

        with torch.inference_mode():
            # Use deterministic mean action for evaluation (PPOActor: Normal → unbounded mu)
            dist    = agent.fast_actor(obs)
            actions = dist.mean

        next_obs, rewards, terminated, truncated, _ = env.step(actions)

        episode_rewards += rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        done = (terminated | truncated).squeeze(-1) if (terminated | truncated).dim() > 1 \
               else (terminated | truncated)
        if done.any():
            for r in episode_rewards[done].tolist():
                ep_count += 1
                print(f"[FAME] Episode {ep_count:4d}  return={r:.2f}")
            episode_rewards[done] = 0.0

        obs = next_obs
        timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Video recording complete ({args_cli.video_length} steps).")
            break

        # Real-time pacing
        elapsed = time.time() - t_start
        if args_cli.real_time and dt - elapsed > 0:
            time.sleep(dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
