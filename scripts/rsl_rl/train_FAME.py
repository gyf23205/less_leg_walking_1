# Copyright (c) 2026
# FAME pipeline using rsl_rl OnPolicyRunner (PPO) for the fast learner.
# This script mirrors the skrl `train_FAME.py` continual loop but uses
# the rsl_rl runners where applicable. Automatic re-launch is disabled;
# the script saves an rsl-rl checkpoint and a friendly .pth sidecar at
# the end of each segment so the user can run `distill_FAME.py` to
# train the meta learner (teacher-based distillation) or re-run this
# script for the next task manually.

import sys
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

import argparse
import os
import time
import math
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
from . import cli_args  # relative import to scripts/rsl_rl/cli_args.py

# add argparse arguments
parser = argparse.ArgumentParser(description="Train FAME with rsl_rl PPO (fast learner).")
parser.add_argument("--tasks", type=str, required=True, help="Comma-separated list of IsaacLab task IDs to visit.")
parser.add_argument("--switch_steps", type=int, default=100_000, help="Number of env-step rows to run per task segment.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel envs.")
parser.add_argument("--task_idx", type=int, default=0, help="Index into --tasks to start from (used when re-launching).")
parser.add_argument("--log_dir", type=str, default=None, help="Existing log directory to reuse when re-launching.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from for the fast runner.")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--device", type=str, default=None)
# append rsl_rl cli args
cli_args.add_rsl_rl_args(parser)
# append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# enable cameras automatically if video requested
if getattr(args_cli, "video", False):
    args_cli.enable_cameras = True

# prepare Hydra argv for task config injection
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# rest imports after AppLauncher
import importlib.metadata as metadata
from packaging import version
import platform
import torch
import gymnasium as gym
import omni
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.io import dump_yaml, dump_pickle
from isaaclab.utils.dict import print_dict
import isaaclab_tasks  # noqa: F401
import less_leg_walking_1.tasks  # noqa: F401

# Check rsl-rl version (keep same as other rsl_rl scripts)
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(f"Please install rsl-rl-lib {RSL_RL_VERSION}. Suggested command: {' '.join(cmd)}")
    raise SystemExit(1)

# helper to save a friendly pytorch sidecar
def _save_pth_checkpoint(runner: OnPolicyRunner, path_base: str) -> None:
    try:
        pth_path = f"{path_base}.pth"
        model = None
        if hasattr(runner.alg, 'actor_critic'):
            model = runner.alg.actor_critic
        elif hasattr(runner.alg, 'policy'):
            model = runner.alg.policy
        elif hasattr(runner.alg, 'actor'):
            model = runner.alg.actor
        data = {
            'model': model.state_dict() if model is not None else None,
            'meta': {
                'saved_at': time.time(),
                'runner_cfg': getattr(runner, 'cfg', None),
            }
        }
        torch.save(data, pth_path)
        print(f"[FAME:rsl_rl] Wrote .pth checkpoint: {pth_path}")
    except Exception as e:
        print(f"[FAME:rsl_rl] Failed to write .pth checkpoint {path_base}: {e}")


# The hydra decorator will inject task-specific env/agent configs for the selected task index
_TASK_LIST_GLOBAL = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
_CURRENT_TASK = _TASK_LIST_GLOBAL[args_cli.task_idx]

@hydra_task_config(_CURRENT_TASK, args_cli.agent)
def main(env_cfg, agent_cfg):
    # agent_cfg is an RslRl runner cfg (RslRlOnPolicyRunnerCfg or Distillation)
    # allow CLI overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # set seeds
    if hasattr(agent_cfg, 'seed') and agent_cfg.seed is not None:
        seed = agent_cfg.seed
    else:
        seed = 42

    env_cfg.seed = seed

    # logging directory
    if args_cli.log_dir:
        log_dir = args_cli.log_dir
    else:
        log_root = os.path.abspath(os.path.join('logs', 'rsl_rl', agent_cfg.experiment_name or 'FAME_rslrl'))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.join(log_root, f"{timestamp}_FAME_rslrl")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Save agent/env params for reproducibility
    dump_yaml(os.path.join(log_dir, 'params', 'env.yaml'), env_cfg)
    dump_yaml(os.path.join(log_dir, 'params', 'agent.yaml'), agent_cfg)
    dump_pickle(os.path.join(log_dir, 'params', 'env.pkl'), env_cfg)
    dump_pickle(os.path.join(log_dir, 'params', 'agent.pkl'), agent_cfg)

    # video wrapper
    video_cfg = None
    if getattr(args_cli, 'video', False):
        video_cfg = {
            'video_folder': os.path.join(log_dir, 'videos', 'train'),
            'step_trigger': lambda step: step % args_cli.video_interval == 0,
            'video_length': args_cli.video_length,
            'disable_logger': True,
        }
        print('[INFO] Recording videos during training.')
        print_dict(video_cfg)

    # create env
    env = gym.make(args_cli.tasks.split(',')[args_cli.task_idx], cfg=env_cfg, render_mode='rgb_array' if video_cfg else None)
    # rsl_rl wrapper
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, 'clip_actions', True))

    num_envs = getattr(env, 'num_envs', 1)
    print(f"[INFO] Created env: num_envs={num_envs}")

    # create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device if hasattr(agent_cfg, 'device') else None)
    runner.obs_all = []

    # resume if checkpoint provided
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

    # compute iterations for this segment
    # rsl_rl's iteration collects `num_steps_per_env` rows per env per iteration
    n_steps_per_env = getattr(agent_cfg, 'num_steps_per_env', getattr(agent_cfg, 'num_steps', 24))
    rows_per_iteration = n_steps_per_env * num_envs
    iterations_needed = max(1, math.ceil(args_cli.switch_steps / rows_per_iteration))

    print(f"[FAME:rsl_rl] Running {iterations_needed} iterations for this task segment (~{iterations_needed*rows_per_iteration} rows)")

    # start training (this runs learning loop inside rsl_rl)
    runner.learn(num_learning_iterations=iterations_needed)

    # Save checkpoint(s)
    ckpt_base = os.path.join(log_dir, f"checkpoint_task{args_cli.task_idx:02d}_end")
    runner.save(ckpt_base)
    try:
        _save_pth_checkpoint(runner, ckpt_base)
    except Exception:
        pass
    print(f"[FAME:rsl_rl] Checkpoint saved: {ckpt_base}")

    # close and exit to allow manual relaunch for the next task
    try:
        env.close()
    except Exception:
        pass
    print('[FAME:rsl_rl] Segment complete — exiting for manual relaunch.')
    os._exit(0)

if __name__ == '__main__':
    main()
