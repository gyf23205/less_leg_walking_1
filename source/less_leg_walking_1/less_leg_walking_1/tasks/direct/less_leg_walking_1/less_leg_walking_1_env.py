# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .less_leg_walking_1_env_cfg import LessLegWalkingFlatEnvCfg, LessLegWalkingRoughEnvCfg


class LessLegWalkingEnv(DirectRLEnv):
    cfg: LessLegWalkingFlatEnvCfg | LessLegWalkingRoughEnvCfg

    def __init__(self, cfg: LessLegWalkingFlatEnvCfg | LessLegWalkingRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        # Modified for 3 legs: 9 joints instead of 12
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.full_action_for_KAE = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
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
                "bias_to_skill",
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
        self._actions = actions.clone()
        # Create full 12-joint action vector with zeros for the disabled RF leg
        # print("--- CONTROL INPUT MAPPING ---")
        # for i, name in enumerate(self._robot.data.joint_names):
        #     print(f"Index {i}: {name}")        
        # Joint order in ANYmalC: ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']        
        #                             0         1          2         3         4         5         6         7         8         9        10        11

        # # #######[P4] IS THIS CORRECT ORDER??????? - highly likely actions order is not correct
        # full_actions = torch.zeros(self.num_envs, 12, device=self.device)

        # # LF leg: actions[0:3] -> full_actions[0, 4, 8]
        # full_actions[:, [0, 4, 8]] = actions[:, 0:3]
        # # LH leg: actions[3:6] -> full_actions[1, 5, 9]  
        # full_actions[:, [1, 5, 9]] = actions[:, 3:6]
        # # RH leg: actions[6:9] -> full_actions[3, 7, 11]
        # full_actions[:, [3, 7, 11]] = actions[:, 6:9]
        # # RF leg: full_actions[2, 6, 10] remain as zero (disabled)
        # self._processed_actions = self.cfg.action_scale * full_actions + self._robot.data.default_joint_pos

        # full_action_for_KAE = full_actions
        # full_action_for_KAE[:, [2, 6, 10]] = actions[:,9:12]
        # self.full_action_for_KAE = full_action_for_KAE
        # # # ####### SHOULDN'T I SUPPOSE TO SOME HOW GENERATE FULL CONTROL INPUT AND FEED IT INTO KAE?
        
        self._actions[:, [2, 6, 10]] = 0.0
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        self.full_action_for_KAE = actions.clone()
        ####### 

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, LessLegWalkingFlatEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        
        # # Get joint positions and velocities for only the 3 active legs
        # # Joint order in robot: ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
        # # We want: [LF_HAA, LF_HFE, LF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE]
        # joint_indices = torch.tensor([0, 4, 8, 1, 5, 9, 3, 7, 11], device=self.device)
        
        # # Extract joint data for active legs only
        # joint_pos_active = (self._robot.data.joint_pos - self._robot.data.default_joint_pos)[:, joint_indices]
        # joint_vel_active = self._robot.data.joint_vel[:, joint_indices]
        
        # obs = torch.cat(
        #     [
        #         tensor
        #         for tensor in (
        #             self._robot.data.root_lin_vel_b,          # 3
        #             self._robot.data.root_ang_vel_b,          # 3  
        #             self._robot.data.projected_gravity_b,     # 3
        #             self._commands,                           # 3
        #             joint_pos_active,                        # 9 (reduced from 12)
        #             joint_vel_active,                        # 9 (reduced from 12)
        #             height_data,                              # 187 for rough terrain or None for flat
        #             self._actions,                            # 9 (reduced from 12)
        #         )
        #         if tensor is not None
        #     ],
        #     dim=-1,
        # )

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

        # print(observations["policy"].size())
        # temp_a = self._robot.data.joint_pos
        # print("_robot.data.joint_pos: ", temp_a.size())
        # print("joint_pos_active", joint_pos_active.size())
        # temp_b = self._robot.data.joint_vel
        # print("_robot.data.joint_vel: ", temp_b.size())
        # print("joint_vel_active: ", joint_vel_active.size())

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
        bias_to_skill_reward = torch.zeros(self.num_envs, device=self.device)

        # expert_weights = self._policy_ref.extras["expert_weights"]
        # # Calculate L2 norm penalty: sum of squares of the weights
        # l2_penalty = torch.sum(torch.square(expert_weights), dim=-1)
        # bias_to_skill_reward = -l2_penalty # Negative sign to make it a penalty
        # # --- END: MODIFIED SECTION -

        # bias_to_skill_reward = self._policy_ref._last_diversity

        # # # DEBUG: Check if policy reference exists
        # if hasattr(self, '_policy_ref'):

        #     moe_weights = self._policy_ref.last_moe_weights
        #     # print(f"[DEBUG] Found last_moe_weights with shape: {moe_weights.shape}")
            
        #     # Calculate dimensions
        #     total_dim = moe_weights.shape[1]
        #     act_dim = self._actions.shape[1]
        #     obv_dim = total_dim - act_dim
            
        #     if obv_dim > 0:
        #         expert_weights_abs = torch.abs(moe_weights[:, :obv_dim])

        #         # Entropy of expert weight distribution
        #         weights_normalized = expert_weights_abs / (expert_weights_abs.sum(dim=1, keepdim=True) + 1e-8)
        #         entropy = -(weights_normalized * torch.log(weights_normalized + 1e-8)).sum(dim=1)

        #         scale = getattr(self.cfg, "bias_to_skill_reward_scale", 0.0)
        #         bias_to_skill_reward = entropy * float(scale) * self.step_dt

                                 
        ########################################################################


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

            "bias_to_skill": self.cfg.bias_to_skill_reward_scale*bias_to_skill_reward * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        reward_without_bias = reward - self.cfg.bias_to_skill_reward_scale * bias_to_skill_reward * self.step_dt
                
        # Let's return the FULL reward for training, but track core separately
        if not hasattr(self, '_episode_core_reward'):
            self._episode_core_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self, '_episode_full_reward'):
            self._episode_full_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        self._episode_core_reward += reward_without_bias
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
            extras["train/bias_contribution"] = full_mean - core_mean
            
            self._episode_core_reward[env_ids] = 0.0
            self._episode_full_reward[env_ids] = 0.0
            
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
