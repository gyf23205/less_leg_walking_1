# Simple distillation helper to train a Meta policy using rsl_rl's DistillationRunner.
# Usage: point --teacher_checkpoint to the fast-run checkpoint (.pt/.pth/.zip as produced by rsl-rl)
# and configure the distillation runner via your task's rsl_rl cfg (use DistillationRunner entry-point).

import sys
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

import argparse
import os
from isaaclab.app import AppLauncher

from . import cli_args

parser = argparse.ArgumentParser(description="Distil a Fast checkpoint into a Meta policy using rsl_rl DistillationRunner.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to fast learner checkpoint to use as teacher.")
parser.add_argument("--num_iterations", type=int, default=200, help="Number of distillation iterations to run.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--device", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear hydra args
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
from packaging import version
import platform
import torch
import gymnasium as gym
from datetime import datetime

from rsl_rl.runners import DistillationRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.io import dump_yaml, dump_pickle
import isaaclab_tasks  # noqa: F401
import less_leg_walking_1.tasks  # noqa: F401

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    raise SystemExit(f"Please install rsl-rl-lib=={RSL_RL_VERSION}")

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # logging
    log_root = os.path.abspath(os.path.join('logs', 'rsl_rl', agent_cfg.experiment_name or 'FAME_distill'))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(log_root, f"{timestamp}_distill")
    os.makedirs(log_dir, exist_ok=True)

    # create env and wrapper
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create distillation runner
    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device if hasattr(agent_cfg, 'device') else None)

    # load teacher checkpoint
    print(f"[distill] Loading teacher checkpoint: {args_cli.teacher_checkpoint}")
    runner.load(args_cli.teacher_checkpoint)

    # run distillation
    runner.learn(num_learning_iterations=args_cli.num_iterations)

    # save
    ckpt_base = os.path.join(log_dir, 'checkpoint_distill_end')
    runner.save(ckpt_base)
    print(f"[distill] Saved distillation checkpoint: {ckpt_base}")

if __name__ == '__main__':
    main()
