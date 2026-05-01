# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of a TDCLS agent from skrl.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of a TDCLS agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

from less_leg_walking_1.tasks.direct.less_leg_walking_1.baselines.TDCLS.agent import TDCLS, TDCLS_DEFAULT_CONFIG

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import less_leg_walking_1.tasks  # noqa: F401

# TDCLS always uses this entry point
agent_cfg_entry_point = "skrl_tdcls_cfg_entry_point"
algorithm = "tdcls"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with TDCLS agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # resolve checkpoint path
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # get environment dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    device = env.device
    num_envs = env.num_envs
    observation_space = env.observation_space
    action_space = env.action_space

    # ------------------------------------------------------------------
    # Build models (same logic as train_TDCLS.py)
    # ------------------------------------------------------------------
    from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

    def build_model(model_cfg: dict):
        cfg = model_cfg.copy()
        model_class = cfg.pop("class", "DeterministicMixin")
        if model_class == "GaussianMixin":
            return gaussian_model(
                observation_space=observation_space,
                action_space=action_space,
                device=device,
                **cfg,
            )
        else:
            return deterministic_model(
                observation_space=observation_space,
                action_space=action_space,
                device=device,
                **cfg,
            )

    models_cfg = agent_cfg.get("models", {})
    models = {
        name: build_model(cfg)
        for name, cfg in models_cfg.items()
        if name not in ("separate",) and isinstance(cfg, dict)
    }

    # ------------------------------------------------------------------
    # Instantiate agent (no memory needed for evaluation)
    # ------------------------------------------------------------------
    tdcls_cfg = TDCLS_DEFAULT_CONFIG.copy()
    tdcls_cfg.update({k: v for k, v in agent_cfg["agent"].items() if k != "class"})
    # disable checkpointing and logging during play
    tdcls_cfg["experiment"]["write_interval"] = 0
    tdcls_cfg["experiment"]["checkpoint_interval"] = 0

    tdcls_agent = TDCLS(
        models=models,
        memory=None,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        cfg=tdcls_cfg,
    )

    # load checkpoint and switch to eval mode
    tdcls_agent.load(resume_path)
    tdcls_agent.set_running_mode("eval")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # act() returns (actions, log_prob, outputs_dict)
            actions, _, outputs = tdcls_agent.act(obs, timestep=0, timesteps=0)
            # use deterministic mean actions (no sampling noise)
            actions = outputs.get("mean_actions", actions)
            obs, _, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
