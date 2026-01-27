# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG, ANYDRIVE_3_LSTM_ACTUATOR_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

##
# Three-legged robot configuration
##

# Create a modified ANYmal-C configuration for three-legged walking
# We'll disable the right front (RF) leg by excluding its joints from actuation
THREE_LEG_ANYMAL_C_CFG = ANYMAL_C_CFG.replace(
    actuators={
        "legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG.replace(
            # Only actuate 3 legs: LF (Left Front), LH (Left Hind), RH (Right Hind)
            # Exclude RF (Right Front) leg joints
            joint_names_expr=["LF_HAA", "LF_HFE", "LF_KFE", "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"],
        )
    },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # Slightly lower starting height for better stability
        joint_pos={
            # Active legs (LF, LH, RH) - adjusted for better walking stance
            "LF_HAA": 0.1, "LH_HAA": 0.0, "RH_HAA": -0.1,  # Slight hip abduction for stability
            "LF_HFE": 0.5, "LH_HFE": -0.3, "RH_HFE": -0.3,  # Asymmetric hip flexion
            "LF_KFE": -0.9, "LH_KFE": 0.6, "RH_KFE": 0.6,   # Adjusted knee angles
            # Disabled leg (RF) - fold it up to avoid ground contact
            "RF_HAA": 0.0,
            "RF_HFE": -1.2,  # Fold the hip more to lift the leg
            "RF_KFE": 2.0,   # Bend the knee to tuck the leg up
        },
    ),
)


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class LessLegWalkingFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12  # With 3 channels being zeroed out for the missing leg
    observation_space = 235  # 235 to 226
    state_space = 0

    logger = "wandb"                    # enable wandb logger
    wandb_project = "koopman_ext"   # your W&B project name
    experiment_name = "IsaacLab"  # used as a folder and W&B group name
    run_name = "three_legged"           # name shown on W&B

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot - use three-legged configuration
    robot: ArticulationCfg = THREE_LEG_ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales - adjusted for three-legged walking with forward motion bias
    lin_vel_reward_scale = 3.0  # Increased to strongly reward forward motion
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -1.0  # Reduced penalty to allow some vertical motion
    ang_vel_reward_scale = -0.02  # Reduced penalty 
    joint_torque_reward_scale = -1.0e-5  # Reduced penalty to allow more torque for walking
    joint_accel_reward_scale = -1.0e-7  # Reduced penalty
    action_rate_reward_scale = -0.005  # Reduced penalty to allow more dynamic actions
    feet_air_time_reward_scale = 1.5  # Increased significantly for better gait
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -1.0  # Further reduced as 3-leg robot needs to tilt
    stability_reward_scale = 0.5  # Reduced to not dominate other rewards
    forward_progress_reward_scale = 2.0  # New reward for forward progress
    bias_to_skill_reward_scale = -0.01

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class LessLegWalkingRoughEnvCfg(LessLegWalkingFlatEnvCfg):
    # env
    observation_space = 235  # Changed from 235 to 226 (reduced by 9 for missing leg)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # # we add a height scanner for perceptive locomotion
    # height_scanner = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
