# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# testing testing 11111

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
    action_space = 12
    observation_space = 235
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

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
    # penalty for large raw action norms (MoE sidecar, keep negative to penalize)
    action_norm_scale = -0.1
    # sensitivity penalty for rapid changes in expert-selection weights (MoE); name kept as used in code
    weight_sensitivty_scale = -0.000
    MoE_magnitude_penality_scale = -0.000


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
    observation_space = 235

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

@configclass
class AnymalCFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 235
    state_space = 0

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
        debug_vis=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # height scanner for perceptive locomotion (used for both flat and rough terrain)
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales
    lin_vel_reward_scale = 5.0 # 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0
    # penalty for large raw action norms (MoE sidecar)
    # action_norm_scale = -0.1
    # sensitivity penalty for expert-selection weights (misspelling preserved)
    # weight_sensitivty_scale = -0.1
    MoE_magnitude_penality_scale = -0.000



@configclass
class AnymalCRoughEnvCfg(AnymalCFlatEnvCfg):
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

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0


@configclass
class AnymalJumpEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 4
    action_space = 12
    observation_space = 235
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

    # terrain
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

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # height scanner for perceptive locomotion (used for both flat and rough terrain)
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # control
    action_scale = 0.45  # [rad]
    joint_pos_noise = 0.05  # [rad]

    # jump task
    jump_height_range = (0.25, 0.45)  # [m] relative to default base height
    jump_active_s = 0.7  # [s] duration of one jump arc
    jump_rest_s = 0.6  # [s] standing phase between jumps
    height_error_scale = 0.12
    push_phase_end = 0.32
    flight_phase_end = 0.82
    foot_clearance_threshold_m = 0.02
    foot_clearance_scale_m = 0.03
    takeoff_velocity_cap_mps = 2.0

    # termination
    min_base_height = 0.20
    max_tilt_rad = 0.90

    # reward scales
    height_reward_scale = 1.5
    takeoff_velocity_reward_scale = 2.5
    airborne_progress_reward_scale = 4.0
    all_feet_airborne_bonus_scale = 2.5
    grounded_flight_penalty_scale = -1.5
    landing_stability_reward_scale = 1.0
    upright_reward_scale = 0.8
    left_right_asym_penalty_scale = -1.0
    fore_hind_asym_penalty_scale = -0.6
    roll_pitch_penalty_scale = -0.6
    lateral_drift_penalty_scale = -0.25
    yaw_rate_penalty_scale = -0.08
    action_penalty_scale = -0.0012
    action_rate_penalty_scale = -0.0025
    action_penalty_pushoff_factor = 0.30
    action_rate_penalty_pushoff_factor = 0.35
    MoE_magnitude_penality_scale = -0.000

@configclass
class AnymalJumpRoughEnvCfg(AnymalJumpEnvCfg):
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

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0