# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


mode = "scratch"  # Choose from "scratch", "residual", "MoE", or "component"

# ############################################
# USE THIS FOR TRAIN_SCRATCH OR RESIDUAL
if mode == "scratch" or mode == "residual":
    from ..res_net import ResCfg, ResActorCritic  # Import ResActorCritic

# USE THIS FOR MoE
if mode == "MoE":
    from ..MoE import MoECfg, MoEActorCritic  # Import both

# USE THIS FOR COMPONENT
if mode == "component":
    from ..baselines.CompoNet import CompoCfg, CompoActorCritic  # Import both
# ############################################

@configclass
class LessLegWalkingFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500  # Increased for more training
    save_interval = 200
    # experiment_name = "less_leg_walking_flat"
    # # experiment_name = "MoE16_less_leg_walking_flat"
    # # experiment_name = "Residual_less_leg_walking_flat"
    # # experiment_name = "Nominal_less_leg_walking_flat"
    # # experiment_name = "Componet_less_leg_walking_flat"
    experiment_name = {
            "scratch": "less_leg_walking_flat",
            "MoE": "MoE16_less_leg_walking_flat",
            "residual": "Residual_less_leg_walking_flat",
            "component": "Component_less_leg_walking_flat"
        }[mode]

    empirical_normalization = False
    # resume = True

    ############################################
    # USE THIS FOR TRAIN_SCRATCH OR RESIDUAL
    if mode == "residual":
        policy = ResCfg()

    # USE THIS FOR MoE
    if mode == "MoE":
        policy = MoECfg()

    # USE THIS FOR COMPONENT
    if mode == "component":
        policy = CompoCfg()

    # # Train from scratch nominal policy
    if mode == "scratch":
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
    ############################################

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,  # Increased entropy for more exploration
        num_learning_epochs=8,  # More learning epochs
        num_mini_batches=8, #4
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LessLegWalkingRoughPPORunnerCfg(LessLegWalkingFlatPPORunnerCfg):
    # experiment_name = "less_leg_walking_rough"
    experiment_name = {
            "scratch": "less_leg_walking_rough",
            "MoE": "MoE16_less_leg_walking_rough",
            "residual": "Residual_less_leg_walking_rough",
            "component": "Component_less_leg_walking_rough"
        }[mode]
    max_iterations = 2500
    
    # Slightly different hyperparameters for rough terrain
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.7, # 1.0
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.007, # 0.005 
        num_mini_batches=8, #5
        num_learning_epochs=8, # 5
        learning_rate=2.0e-4, # 1e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=2.0, # 1.0
    )

@configclass
class AnymalCFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500  # Increased for more training
    save_interval = 200
    # experiment_name = "anymal_c_flat_leg_walking"
    # # experiment_name = "anymal_c_rough_leg_walking"
    # # experiment_name = "Residual_anymal_c_rough_leg_walking"
    # # experiment_name = "Nominal_anymal_c_rough_leg_walking"
    # # experiment_name = "Componet_anymal_c_rough_leg_walking"
    experiment_name = {
            "scratch": "anymal_c_flat_leg_walking",
            "MoE": "MoE16_anymal_c_flat_leg_walking",
            "residual": "Residual_anymal_c_flat_leg_walking",
            "component": "Component_anymal_c_flat_leg_walking"
        }[mode]
    empirical_normalization = False
    # resume = True

    ############################################
    # USE THIS FOR TRAIN_SCRATCH OR RESIDUAL
    if mode == "residual":
        policy = ResCfg()

    # USE THIS FOR MoE
    if mode == "MoE":
        policy = MoECfg()

    # USE THIS FOR COMPONENT
    if mode == "component":
        policy = CompoCfg()

    # # Train from scratch nominal policy
    if mode == "scratch":
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
    ############################################

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,  # Increased entropy for more exploration
        num_learning_epochs=8,  # More learning epochs
        num_mini_batches=8, #4
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class AnymalCRoughPPORunnerCfg(AnymalCFlatPPORunnerCfg):
    # experiment_name = "anymal_c_rough_leg_walking"
    experiment_name = {
            "scratch": "anymal_c_rough_leg_walking",
            "MoE": "MoE16_anymal_c_rough_leg_walking",
            "residual": "Residual_anymal_c_rough_leg_walking", 
            "component": "Component_anymal_c_rough_leg_walking"
        }[mode]
    max_iterations = 2500
    
    # Slightly different hyperparameters for rough terrain
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.7, # 1.0
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.007, # 0.005 
        num_mini_batches=8, #5
        num_learning_epochs=8, # 5
        learning_rate=2.0e-4, # 1e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=2.0, # 1.0
    )

@configclass
class AnymalJumpFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # experiment_name = "anymal_c_jump_flat"
    experiment_name = {
            "scratch": "anymal_c_jump_flat",
            "MoE": "MoE16_anymal_c_jump_flat",
            "residual": "Residual_anymal_c_jump_flat",
            "component": "Component_anymal_c_jump_flat"
        }[mode]
    num_steps_per_env = 24
    max_iterations = 2500 # 4000  # Increased for more training
    save_interval = 200
    empirical_normalization = False

    ############################################
    # USE THIS FOR TRAIN_SCRATCH OR RESIDUAL
    if mode == "residual":
        policy = ResCfg()

    # USE THIS FOR MoE
    if mode == "MoE":
        policy = MoECfg()

    # USE THIS FOR COMPONENT
    if mode == "component":
        policy = CompoCfg()

    # # Train from scratch nominal policy
    if mode == "scratch":
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
    ############################################

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_mini_batches=8, #4
        num_learning_epochs=8,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalJumpRoughPPORunnerCfg(AnymalJumpFlatPPORunnerCfg):
    # experiment_name = "anymal_c_jump_rough"
    experiment_name = {
            "scratch": "anymal_c_jump_rough",
            "MoE": "MoE16_anymal_c_jump_rough",    
            "residual": "Residual_anymal_c_jump_rough",
            "component": "Component_anymal_c_jump_rough"
        }[mode]
    # Slightly different hyperparameters for jump on rough terrain
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, # 1.0
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008, # 0.005
        num_mini_batches=8, #4
        num_learning_epochs=8, # 5
        learning_rate=2.0e-4, # 1e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0, # 1.0
    )