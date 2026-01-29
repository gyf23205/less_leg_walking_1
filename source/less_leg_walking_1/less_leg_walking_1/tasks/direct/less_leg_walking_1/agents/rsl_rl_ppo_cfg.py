# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# ############################################
# USE THIS FOR TRAIN_SCRATCH OR RESIDUAL
# from ..res_net import ResCfg, ResActorCritic  # Import ResActorCritic

# USE THIS FOR MoE
from ..MoE import MoECfg, MoEActorCritic  # Import both
# ############################################


@configclass
class LessLegWalkingFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500  # Increased for more training
    save_interval = 50
    experiment_name = "less_leg_walking_flat"
    # experiment_name = "MoE16_less_leg_walking_flat"
    # experiment_name = "Residual_less_leg_walking_flat"
    # experiment_name = "Nominal_less_leg_walking_flat"
    empirical_normalization = False

    ############################################
    # USE THIS FOR TRAIN_SCRATCH OR RESIDUAL
    # policy = ResCfg()

    # USE THIS FOR MoE
    policy = MoECfg()

    # Train from scratch nominal policy
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    # )
    ############################################

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,  # Increased entropy for more exploration
        num_learning_epochs=8,  # More learning epochs
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LessLegWalkingRoughPPORunnerCfg(LessLegWalkingFlatPPORunnerCfg):
    experiment_name = "less_leg_walking_rough"
    max_iterations = 2000
    
    # Slightly different hyperparameters for rough terrain
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.7, # 1.0
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.007, # 0.005 
        num_mini_batches=5,
        num_learning_epochs=10, # 5
        learning_rate=2.0e-4, # 1e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=2.0, # 1.0
    )