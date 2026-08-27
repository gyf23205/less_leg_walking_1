# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import torch.nn.functional as F  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# DEBUG
parser.add_argument("--task", type=str, default="Less-Leg-Rough-Walking-Direct-v1", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# parser = parser.add_argument("checkpoint", type=str, default="/home/yifan/git/less_leg_walking_1/logs/rsl_rl/less_leg_walking_rough/2025-12-21_22-32-25/model_1999.pt")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
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

# IsaacLab/Isaac Sim dependent imports (require Kit to be started above)
from less_leg_walking_1.tasks.direct.less_leg_walking_1.utils import (
    get_experts_outputs,
    extend_experts_outputs,
)

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from less_leg_walking_1.tasks.direct.less_leg_walking_1.MoE import MoEActorCritic
import rsl_rl.runners.on_policy_runner
# Inject the class so eval("MoEActorCritic") inside the runner can find it
rsl_rl.runners.on_policy_runner.MoEActorCritic = MoEActorCritic

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import less_leg_walking_1.tasks  # noqa: F401

from less_leg_walking_1.tasks.direct.less_leg_walking_1.MoE import MoEActorCritic
# Make the class available in the runner module's namespace
import rsl_rl.runners.on_policy_runner as runner_module
runner_module.MoEActorCritic = MoEActorCritic


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "task1", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # assert False
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # build a runner with freshly initialized weights; how it gets populated depends on
    # whether the checkpoint is a standard rsl-rl checkpoint or train_moe.py's custom
    # "complete_model_with_metadata.pth" (which stores raw 'actor'/'critic' modules).
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    checkpoint = torch.load(resume_path, map_location=agent_cfg.device, weights_only=False)
    # use_custom_actor = isinstance(checkpoint, dict) and "actor" in checkpoint

    # if use_custom_actor:
    #     # custom checkpoint saved by train_moe.py: {"actor": model.actor, "critic": model.critic, "obs_range": ...}
    #     print("[INFO]: Detected custom checkpoint format, loading 'actor' module directly as the policy.")
    #     policy_nn.actor = checkpoint["actor"].to(agent_cfg.device)
    #     if checkpoint.get("critic") is not None:
    #         policy_nn.critic = checkpoint["critic"].to(agent_cfg.device)
    #     policy_nn.eval()

    #     def policy(obs):
    #         with torch.no_grad():
    #             obs_t = policy_nn.get_actor_obs(obs)
    #             obs_t = policy_nn.actor_obs_normalizer(obs_t)
    #             return policy_nn.actor(obs_t)

    # else:

    # --- checkpoint 키 이름 보정 (kaes.0.* -> kae.*) ---
    _policy = runner.alg.policy
    _orig_load = _policy.load_state_dict

    def _remap_load(state_dict, *args, **kwargs):
        fixed, n = {}, 0
        for k, v in state_dict.items():
            if k.startswith("kaes.0."):
                k = k.replace("kaes.0.", "kae.", 1); n += 1
            fixed[k] = v
        print(f"[remap] {n} keys renamed")
        return _orig_load(fixed, *args, **kwargs)

    _policy.load_state_dict = _remap_load
    # ---------------------------------------------------

    runner.load(resume_path)
    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # disable normalizers to avoid 226->256 mismatch during play
    # Note: 'policy' is a bound method, so we must modify 'policy_nn' (the module)
    if hasattr(policy_nn, "actor_obs_normalizer"):
        policy_nn.actor_obs_normalizer = torch.nn.Identity()
    if hasattr(policy_nn, "critic_obs_normalizer"):
        policy_nn.critic_obs_normalizer = torch.nn.Identity()
    if hasattr(policy_nn, "student_obs_normalizer"):
        policy_nn.student_obs_normalizer = torch.nn.Identity()

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # normalizer expects 226-D but we pad to 256; skip it to avoid shape mismatch
    export_policy_as_jit(policy_nn, normalizer=None, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=None, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    def _pad_to_dim(tensor: torch.Tensor, padded_dim: int) -> torch.Tensor:
        if tensor.shape[-1] < padded_dim:
            pad_size = padded_dim - tensor.shape[-1]
            return F.pad(tensor, (0, pad_size), value=1.0)
        return tensor[..., : padded_dim]
    # reset environment
    obs = env.get_observations()
    timestep = 0

    _ret = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    _ep_returns = []
    _TARGET = 400          

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():

            needs_pad = not isinstance(policy_nn, MoEActorCritic)
            if needs_pad:
                obs["policy"] = _pad_to_dim(obs["policy"], 256)

            # if not use_custom_actor:
            #     obs["policy"] = _pad_to_dim(obs["policy"], 256)
            actions = policy(obs)
            # env stepping
            # obs, _, _, _ = env.step(actions)
            obs, rew, dones, extras = env.step(actions)      
            _ret += rew
            _done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if _done_ids.numel() > 0:
                _ep_returns.extend(_ret[_done_ids].tolist())
                _ret[_done_ids] = 0.0
                if len(_ep_returns) >= _TARGET:
                    import statistics
                    print(f"[EVAL] n={len(_ep_returns)}  mean={statistics.mean(_ep_returns):.3f}  "
                        f"std={statistics.pstdev(_ep_returns):.3f}")
                    break

            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
