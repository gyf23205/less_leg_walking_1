# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# testing testing
from __future__ import annotations

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .less_leg_walking_1_env_cfg import LessLegWalkingFlatEnvCfg, LessLegWalkingRoughEnvCfg, AnymalCFlatEnvCfg, AnymalCRoughEnvCfg, AnymalJumpEnvCfg


class LessLegWalkingEnv(DirectRLEnv):
    cfg: LessLegWalkingFlatEnvCfg | LessLegWalkingRoughEnvCfg

    def __init__(self, cfg: LessLegWalkingFlatEnvCfg | LessLegWalkingRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        # Modified for 3 legs: 9 joints instead of 12
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.full_action_for_KAE = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions_KAE = self.full_action_for_KAE.clone()
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        self._last_reward_mean = 0.0

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "stability",
                "forward_progress",
                "action_norm",
                "sensitivity",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        # Updated feet detection for 3-legged robot (exclude RF foot)
        self._feet_ids, _ = self._contact_sensor.find_bodies(["LF_FOOT", "LH_FOOT", "RH_FOOT"])
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # if isinstance(self.cfg, LessLegWalkingRoughEnvCfg):
        #     # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        self._actions[:, [2, 6, 10]] = 0.0
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        self._previous_actions_KAE = self.full_action_for_KAE.clone()
        self.full_action_for_KAE = actions.clone()
        ####### 

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)
        
        augmented_action = self.full_action_for_KAE # !!! augmented_actions != self._actions -- self._actions now has 0s for RF leg joints
                                                    # and augmented_actions has original full 12-dim control input (not 0-ed)

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    # self._actions,
                    augmented_action, # <- now it is simply (original) actions.clone()
                )
                if tensor is not None
            ],
            dim=-1,
        )

        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        
        # joint torques - only for active joints (3 legs)
        joint_indices = torch.tensor([0, 4, 8, 1, 5, 9, 3, 7, 11], device=self.device)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque[:, joint_indices]), dim=1)
        
        # joint acceleration - only for active joints (3 legs)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, joint_indices]), dim=1)
        
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        
        # feet air time - only for 3 active feet
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # Remove command dependency - reward air time regardless of commanded velocity
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
        
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # stability reward - encourage maintaining balance with 3 legs
        # Penalize excessive tilting and reward stable base motion
        base_ang_vel = torch.norm(self._robot.data.root_ang_vel_b[:, :2], dim=1)
        stability = torch.exp(-base_ang_vel / 0.5)  # Exponential reward for low angular velocity
        
        # forward progress reward - strongly encourage forward motion
        forward_velocity = self._robot.data.root_lin_vel_b[:, 0]  # x-velocity in body frame
        forward_progress = torch.clamp(forward_velocity, 0.0, 2.0)  # Reward positive forward motion

        # Give more rewawrd for using KAE ####################################
        # Give more reward for using KAE (observation-based skills)
        # bias_to_skill_reward = torch.zeros(self.num_envs, device=self.device)       

        # weight sensitivty
        # try:
        #     action_norm_penalty = torch.sum(torch.square(self.full_action_for_KAE), dim=1)
        # except: 
        #     action_norm_penalty = torch.zeros(self.num_envs, device=self.device)

        action_norm_penalty = torch.zeros(self.num_envs, device=self.device)

        try:
            current_weights = self._policy_ref.last_expert_weights # [16 experts]
            weight_stability = torch.sum(torch.square(current_weights - self._prev_expert_weights), dim=1)
            self._prev_expert_weights = current_weights.clone()
        except:
            weight_stability = torch.zeros(self.num_envs, device=self.device)
            try:
                current_weights = self._policy_ref.last_expert_weights
                self._prev_expert_weights = current_weights.clone()
            except:    
                pass

        # try:
        #     current_weights = self._policy_ref.last_expert_weights # [16 experts]
        #     action_norm_penalty = torch.sum(torch.square(self.full_action_for_KAE), dim=1)

        #     # Penalize the variance/jitter of the expert selection.
        #     # For a phantom leg, the weights often fluctuate wildly as the network 
        #     # 'searches' for feedback. This dampens that search.
        #     weight_stability = torch.sum(torch.square(current_weights - self._prev_expert_weights), dim=1)

        #     # Update buffer
        #     self._prev_expert_weights = current_weights.clone()
        # except:
        #     weight_stability = torch.zeros(self.num_envs, device=self.device)
        #     action_norm_penalty = torch.zeros(self.num_envs, device=self.device)


        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "stability": stability * self.cfg.stability_reward_scale * self.step_dt,
            "forward_progress": forward_progress * self.cfg.forward_progress_reward_scale * self.step_dt,
            "action_norm": self.cfg.action_norm_scale*action_norm_penalty * self.step_dt,
            "sensitivity": self.cfg.weight_sensitivty_scale*weight_stability * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        reward_without_MoE_only = reward - (self.cfg.action_norm_scale*action_norm_penalty * self.step_dt) \
                            - (self.cfg.weight_sensitivty_scale*weight_stability * self.step_dt)
                
        # Let's return the FULL reward for training, but track core separately
        if not hasattr(self, '_episode_core_reward'):
            self._episode_core_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self, '_episode_full_reward'):
            self._episode_full_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        self._episode_core_reward += reward_without_MoE_only
        self._episode_full_reward += reward
        self._last_reward_mean = reward.mean().item()


        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        # self.episode_length_buf[env_ids] = torch.randint(
        #     0, int(self.max_episode_length), (len(env_ids),), 
        #     device=self.device, dtype=self.episode_length_buf.dtype
        # )
        # timeout_rate = self.reset_time_outs.float().mean().item()
        # print(f"Timeout rate: {timeout_rate:.3f}")


        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands - bias towards forward motion for three-legged walking
        # Forward velocity: 0.2 to 1.5 m/s (mostly forward motion)
        self._commands[env_ids, 0] = torch.empty_like(self._commands[env_ids, 0]).uniform_(0.2, 1.5)
        # Lateral velocity: -0.3 to 0.3 m/s (small lateral motion)  
        self._commands[env_ids, 1] = torch.empty_like(self._commands[env_ids, 1]).uniform_(-0.3, 0.3)
        # Yaw rate: -0.5 to 0.5 rad/s (small turning)
        self._commands[env_ids, 2] = torch.empty_like(self._commands[env_ids, 2]).uniform_(-0.5, 0.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # Ensure the disabled RF leg is properly positioned (folded up)
        # RF joints are at indices [2, 6, 10] for [HAA, HFE, KFE]
        joint_pos[:, 2] = 0.0    # RF_HAA
        joint_pos[:, 6] = -1.2   # RF_HFE (fold hip more)
        joint_pos[:, 10] = 2.0   # RF_KFE (bend knee to tuck leg up)
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # Log core mean reward (matching RSL-RL's Mean reward format)
        if hasattr(self, '_episode_core_reward') and hasattr(self, '_episode_full_reward'):
            core_mean = torch.mean(self._episode_core_reward[env_ids]).item()
            full_mean = torch.mean(self._episode_full_reward[env_ids]).item()
            
            extras["train/core_mean_reward"] = core_mean
            extras["train/full_mean_reward"] = full_mean
            extras["train/MoE_only_reward(penality) contribution"] = full_mean - core_mean
            
            self._episode_core_reward[env_ids] = 0.0
            self._episode_full_reward[env_ids] = 0.0
            
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        self._episode_full_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_full_reward += reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Ensure writable tensors before doing in-place indexed updates. Some
        # tensors may be inference tensors created under torch.inference_mode;
        # cloning them ensures we have regular writable tensors.
        try:
            self._actions = self._actions.clone()
            self._previous_actions = self._previous_actions.clone()
            self._commands = self._commands.clone()
        except Exception:
            # Fall back to fresh zero tensors if cloning fails.
            self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
            self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
            self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            # Clone per-key episode sum tensor before doing indexed writes.
            try:
                self._episode_sums[key] = self._episode_sums[key].clone()
            except Exception:
                pass
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        # AnymalCEnv has no MoE-only penalty terms in its reward dict, so core reward equals full reward
        full_mean = torch.mean(self._episode_full_reward[env_ids]).item()
        extras["train/core_mean_reward"] = full_mean
        extras["train/full_mean_reward"] = full_mean
        extras["train/MoE_only_reward(penality) contribution"] = 0.0
        self._episode_full_reward[env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)



class AnymalJumpEnv(DirectRLEnv):
    cfg: AnymalJumpEnvCfg

    def __init__(self, cfg: AnymalJumpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._num_joints = self._robot.data.joint_pos.shape[1]
        self._actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._joint_pos_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._base_z = torch.zeros(self.num_envs, device=self.device)
        self._jump_height = torch.zeros(self.num_envs, device=self.device)
        self._desired_z = torch.zeros(self.num_envs, device=self.device)
        self._desired_vz = torch.zeros(self.num_envs, device=self.device)
        self._jump_time = torch.zeros(self.num_envs, device=self.device)
        self._episode_peak_lift = torch.zeros(self.num_envs, device=self.device)
        self._episode_lift_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_lift_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_airborne_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_flight_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_min_foot_clearance_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_takeoff_vz_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_takeoff_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_lr_asym_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_roll_pitch_err_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_full_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self._default_joint_pos = self._robot.data.default_joint_pos.clone()
        self._resolve_leg_joint_mapping()
        self._resolve_foot_body_mapping()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "height",
                "takeoff_velocity",
                "airborne_progress",
                "all_feet_airborne_bonus",
                "grounded_flight_penalty",
                "landing_stability",
                "upright",
                "left_right_asym",
                "fore_hind_asym",
                "roll_pitch",
                "lateral_drift",
                "yaw_rate",
                "action",
                "action_rate",
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_leg_joint_mapping(self):
        """Resolve per-leg HFE/KFE joint indices using joint names. Fallback to fixed order if needed."""
        leg_order = ["LF", "LH", "RF", "RH"]
        hfe_ids: list[int] = []
        kfe_ids: list[int] = []
        fallback_mapping = False

        def _find_single_joint(patterns: list[str]) -> int | None:
            for pattern in patterns:
                try:
                    joint_ids, _ = self._robot.find_joints(pattern)
                except Exception:
                    continue
                if len(joint_ids) == 1:
                    return int(joint_ids[0])
            return None

        for leg in leg_order:
            hfe_id = _find_single_joint(
                [f"{leg}_HFE", f".*{leg}.*HFE.*", f".*{leg.lower()}.*hfe.*", f".*{leg.lower()}_hfe.*"]
            )
            kfe_id = _find_single_joint(
                [f"{leg}_KFE", f".*{leg}.*KFE.*", f".*{leg.lower()}.*kfe.*", f".*{leg.lower()}_kfe.*"]
            )
            if hfe_id is None or kfe_id is None:
                fallback_mapping = True
                break
            hfe_ids.append(hfe_id)
            kfe_ids.append(kfe_id)

        if fallback_mapping:
            # Deterministic fallback requested by spec.
            # LF=[0,1,2], LH=[3,4,5], RF=[6,7,8], RH=[9,10,11]
            hfe_ids = [1, 4, 7, 10]
            kfe_ids = [2, 5, 8, 11]
            print("[WARN] ANYMAL HFE/KFE joint mapping failed. Using fixed fallback mapping.")

        self._hfe_ids = torch.tensor(hfe_ids, dtype=torch.long, device=self.device)
        self._kfe_ids = torch.tensor(kfe_ids, dtype=torch.long, device=self.device)

    def _resolve_foot_body_mapping(self):
        """Resolve LF/LH/RF/RH foot rigid-body indices; fail fast if unavailable."""
        if not hasattr(self._robot.data, "body_pos_w"):
            raise RuntimeError("AnymalJumpEnv requires ArticulationData.body_pos_w for foot-clearance rewards.")

        leg_order = ["LF", "LH", "RF", "RH"]
        foot_ids: list[int] = []

        def _find_single_body(patterns: list[str]) -> int | None:
            for pattern in patterns:
                try:
                    body_ids, _ = self._robot.find_bodies(pattern)
                except Exception:
                    continue
                if len(body_ids) == 1:
                    return int(body_ids[0])
            return None

        for leg in leg_order:
            foot_id = _find_single_body(
                [f"{leg}_FOOT", f".*{leg}.*FOOT.*", f".*{leg.lower()}.*foot.*", f".*{leg.lower()}_foot.*"]
            )
            if foot_id is None:
                raise RuntimeError(f"AnymalJumpEnv could not resolve a unique foot body for leg {leg}.")
            foot_ids.append(foot_id)

        self._foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions[:] = self._actions
        self._actions = actions.clone().clamp(-1.0, 1.0)

        cycle_s = self.cfg.jump_active_s + self.cfg.jump_rest_s
        self._jump_time += self.step_dt
        t_cycle = torch.remainder(self._jump_time, cycle_s)

        self._desired_z[:] = self._base_z
        self._desired_vz[:] = 0.0
        jump_mask = t_cycle < self.cfg.jump_active_s
        if torch.any(jump_mask):
            t_jump = t_cycle[jump_mask] / self.cfg.jump_active_s
            phase = math.pi * t_jump
            amplitude = self._jump_height[jump_mask]
            self._desired_z[jump_mask] = self._base_z[jump_mask] + amplitude * torch.sin(phase)
            self._desired_vz[jump_mask] = amplitude * (math.pi / self.cfg.jump_active_s) * torch.cos(phase)

        self._joint_pos_target = self._default_joint_pos + self._actions * self.cfg.action_scale

    def _apply_action(self):
        self._robot.set_joint_position_target(self._joint_pos_target)

    def _get_observations(self) -> dict:
        cycle_s = self.cfg.jump_active_s + self.cfg.jump_rest_s
        t_cycle = torch.remainder(self._jump_time, cycle_s)
        phase = 2.0 * math.pi * t_cycle / cycle_s
        desired_z_error = (self._desired_z - self._robot.data.root_pos_w[:, 2]).unsqueeze(1)
        desired_vz_error = (self._desired_vz - self._robot.data.root_lin_vel_w[:, 2]).unsqueeze(1)
        jump_commands = torch.cat([desired_z_error, desired_vz_error, phase.unsqueeze(1)], dim=-1)
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                jump_commands,
                self._robot.data.joint_pos - self._default_joint_pos,
                self._robot.data.joint_vel,
                height_data,
                self._actions,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        cycle_s = self.cfg.jump_active_s + self.cfg.jump_rest_s
        t_cycle = torch.remainder(self._jump_time, cycle_s)
        jump_mask = t_cycle < self.cfg.jump_active_s
        rest_mask = ~jump_mask
        tau = torch.zeros_like(t_cycle)
        tau[jump_mask] = t_cycle[jump_mask] / self.cfg.jump_active_s
        push_mask = jump_mask & (tau < self.cfg.push_phase_end)
        flight_mask = jump_mask & (tau >= self.cfg.push_phase_end) & (tau < self.cfg.flight_phase_end)
        track_mask = rest_mask | (jump_mask & (tau >= self.cfg.flight_phase_end))

        height_error = self._robot.data.root_pos_w[:, 2] - self._desired_z
        height_track = torch.exp(-torch.square(height_error / self.cfg.height_error_scale))
        height_reward = torch.where(track_mask, height_track, torch.zeros_like(height_track))

        lift = torch.clamp(self._robot.data.root_pos_w[:, 2] - self._base_z, min=0.0)
        landing_pos_stability = torch.exp(-torch.square((self._robot.data.root_pos_w[:, 2] - self._base_z) / 0.08))
        landing_vel_stability = torch.exp(-torch.square(self._robot.data.root_lin_vel_w[:, 2] / 0.6))
        landing_stability = torch.where(
            rest_mask, landing_pos_stability * landing_vel_stability, torch.zeros_like(landing_pos_stability)
        )

        # Per-leg extension proxies from HFE/KFE motion.
        hfe_pos = self._robot.data.joint_pos[:, self._hfe_ids]
        kfe_pos = self._robot.data.joint_pos[:, self._kfe_ids]
        hfe_pos_default = self._default_joint_pos[:, self._hfe_ids]
        kfe_pos_default = self._default_joint_pos[:, self._kfe_ids]
        ext_leg = -((hfe_pos - hfe_pos_default) + (kfe_pos - kfe_pos_default))

        # LF, LH, RF, RH order.
        ext_lf, ext_lh, ext_rf, ext_rh = [ext_leg[:, i] for i in range(4)]
        left_right_asym = torch.abs((ext_lf + ext_lh) - (ext_rf + ext_rh))
        fore_hind_asym = torch.abs((ext_lf + ext_rf) - (ext_lh + ext_rh))

        foot_z = self._robot.data.body_pos_w[:, self._foot_ids, 2]
        ground_z = self._terrain.env_origins[:, 2].unsqueeze(1)
        min_clearance = torch.amin(foot_z - ground_z, dim=1)
        airborne_progress_raw = torch.clamp(
            (min_clearance - self.cfg.foot_clearance_threshold_m) / self.cfg.foot_clearance_scale_m, min=0.0, max=1.0
        )
        all_airborne_raw = (min_clearance > self.cfg.foot_clearance_threshold_m).float()

        takeoff_vz_raw = torch.clamp(
            self._robot.data.root_lin_vel_w[:, 2], min=0.0, max=self.cfg.takeoff_velocity_cap_mps
        ) / self.cfg.takeoff_velocity_cap_mps
        takeoff_velocity = torch.where(push_mask, takeoff_vz_raw, torch.zeros_like(takeoff_vz_raw))
        airborne_progress = torch.where(flight_mask, airborne_progress_raw, torch.zeros_like(airborne_progress_raw))
        all_feet_airborne_bonus = torch.where(flight_mask, all_airborne_raw, torch.zeros_like(all_airborne_raw))
        grounded_flight_penalty = torch.where(
            flight_mask, 1.0 - airborne_progress_raw, torch.zeros_like(airborne_progress_raw)
        )

        roll_pitch_penalty = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        lateral_drift_penalty = torch.sum(torch.square(self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        yaw_rate_penalty = torch.square(self._robot.data.root_ang_vel_b[:, 2])

        upright_reward = torch.clamp(-self._robot.data.projected_gravity_b[:, 2], min=0.0, max=1.0)
        action_scale = torch.where(
            push_mask,
            torch.full_like(tau, self.cfg.action_penalty_pushoff_factor),
            torch.ones_like(tau),
        )
        action_rate_scale = torch.where(
            push_mask,
            torch.full_like(tau, self.cfg.action_rate_penalty_pushoff_factor),
            torch.ones_like(tau),
        )
        action_penalty = torch.sum(torch.square(self._actions), dim=1) * action_scale
        action_rate_penalty = torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * action_rate_scale

        rewards = {
            "height": height_reward * self.cfg.height_reward_scale * self.step_dt,
            "takeoff_velocity": takeoff_velocity * self.cfg.takeoff_velocity_reward_scale * self.step_dt,
            "airborne_progress": airborne_progress * self.cfg.airborne_progress_reward_scale * self.step_dt,
            "all_feet_airborne_bonus": all_feet_airborne_bonus * self.cfg.all_feet_airborne_bonus_scale * self.step_dt,
            "grounded_flight_penalty": grounded_flight_penalty
            * self.cfg.grounded_flight_penalty_scale
            * self.step_dt,
            "landing_stability": landing_stability * self.cfg.landing_stability_reward_scale * self.step_dt,
            "upright": upright_reward * self.cfg.upright_reward_scale * self.step_dt,
            "left_right_asym": left_right_asym * self.cfg.left_right_asym_penalty_scale * self.step_dt,
            "fore_hind_asym": fore_hind_asym * self.cfg.fore_hind_asym_penalty_scale * self.step_dt,
            "roll_pitch": roll_pitch_penalty * self.cfg.roll_pitch_penalty_scale * self.step_dt,
            "lateral_drift": lateral_drift_penalty * self.cfg.lateral_drift_penalty_scale * self.step_dt,
            "yaw_rate": yaw_rate_penalty * self.cfg.yaw_rate_penalty_scale * self.step_dt,
            "action": action_penalty * self.cfg.action_penalty_scale * self.step_dt,
            "action_rate": action_rate_penalty * self.cfg.action_rate_penalty_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Per-episode jump metrics.
        self._episode_peak_lift = torch.maximum(self._episode_peak_lift, lift)
        self._episode_lift_sum += lift
        self._episode_lift_count += 1.0
        self._episode_airborne_sum += torch.where(flight_mask, all_airborne_raw, torch.zeros_like(all_airborne_raw))
        self._episode_flight_count += flight_mask.float()
        self._episode_min_foot_clearance_sum += min_clearance
        self._episode_takeoff_vz_sum += torch.where(
            push_mask,
            takeoff_vz_raw * self.cfg.takeoff_velocity_cap_mps,
            torch.zeros_like(takeoff_vz_raw),
        )
        self._episode_takeoff_count += push_mask.float()
        self._episode_lr_asym_sum += left_right_asym
        self._episode_roll_pitch_err_sum += roll_pitch_penalty
        self._episode_step_count += 1.0

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_full_reward += reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        upright = -self._robot.data.projected_gravity_b[:, 2]
        fallen = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < self.cfg.min_base_height,
            upright < math.cos(self.cfg.max_tilt_rad),
        )
        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_height_error = torch.abs(
            self._robot.data.root_pos_w[env_ids, 2] - self._desired_z[env_ids]
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        # AnymalJumpEnv has no MoE-only penalty terms in its reward dict, so core reward equals full reward
        full_mean = torch.mean(self._episode_full_reward[env_ids]).item()
        extras["train/core_mean_reward"] = full_mean
        extras["train/full_mean_reward"] = full_mean
        extras["train/MoE_only_reward(penality) contribution"] = 0.0
        self._episode_full_reward[env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/fallen"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_height_error"] = final_height_error.item()
        extras["Metrics/peak_lift"] = torch.mean(self._episode_peak_lift[env_ids]).item()
        mean_lift_env = self._episode_lift_sum[env_ids] / torch.clamp(self._episode_lift_count[env_ids], min=1.0)
        extras["Metrics/mean_lift"] = torch.mean(mean_lift_env).item()
        airborne_fraction_env = self._episode_airborne_sum[env_ids] / torch.clamp(self._episode_flight_count[env_ids], min=1.0)
        mean_min_foot_clearance_env = self._episode_min_foot_clearance_sum[env_ids] / torch.clamp(
            self._episode_step_count[env_ids], min=1.0
        )
        mean_takeoff_vz_env = self._episode_takeoff_vz_sum[env_ids] / torch.clamp(self._episode_takeoff_count[env_ids], min=1.0)
        mean_lr_asym_env = self._episode_lr_asym_sum[env_ids] / torch.clamp(self._episode_step_count[env_ids], min=1.0)
        mean_roll_pitch_err_env = self._episode_roll_pitch_err_sum[env_ids] / torch.clamp(
            self._episode_step_count[env_ids], min=1.0
        )
        extras["Metrics/airborne_fraction"] = torch.mean(airborne_fraction_env).item()
        extras["Metrics/mean_min_foot_clearance_m"] = torch.mean(mean_min_foot_clearance_env).item()
        extras["Metrics/mean_takeoff_vz"] = torch.mean(mean_takeoff_vz_env).item()
        extras["Metrics/mean_lr_asymmetry"] = torch.mean(mean_lr_asym_env).item()
        extras["Metrics/mean_roll_pitch_error"] = torch.mean(mean_roll_pitch_err_env).item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._episode_peak_lift[env_ids] = 0.0
        self._episode_lift_sum[env_ids] = 0.0
        self._episode_lift_count[env_ids] = 0.0
        self._episode_airborne_sum[env_ids] = 0.0
        self._episode_flight_count[env_ids] = 0.0
        self._episode_min_foot_clearance_sum[env_ids] = 0.0
        self._episode_takeoff_vz_sum[env_ids] = 0.0
        self._episode_takeoff_count[env_ids] = 0.0
        self._episode_lr_asym_sum[env_ids] = 0.0
        self._episode_roll_pitch_err_sum[env_ids] = 0.0
        self._episode_step_count[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        if self.cfg.joint_pos_noise > 0.0:
            joint_pos += torch.zeros_like(joint_pos).uniform_(-self.cfg.joint_pos_noise, self.cfg.joint_pos_noise)
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample new jump targets and randomize jump phase.
        base_z = default_root_state[:, 2]
        self._base_z[env_ids] = base_z
        self._jump_height[env_ids] = torch.zeros_like(base_z).uniform_(
            self.cfg.jump_height_range[0], self.cfg.jump_height_range[1]
        )
        cycle_s = self.cfg.jump_active_s + self.cfg.jump_rest_s
        self._jump_time[env_ids] = torch.zeros_like(base_z).uniform_(0.0, cycle_s)
        self._desired_z[env_ids] = self._base_z[env_ids]
        self._desired_vz[env_ids] = 0.0
