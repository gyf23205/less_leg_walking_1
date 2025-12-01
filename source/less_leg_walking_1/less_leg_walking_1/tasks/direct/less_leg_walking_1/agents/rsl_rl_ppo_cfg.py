# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ..MoE import MoECfg, MoEActorCritic  # Import both



@configclass
class LessLegWalkingFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000  # Increased for more training
    save_interval = 50
    experiment_name = "less_leg_walking_flat"
    empirical_normalization = False
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    # )
    policy = MoECfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,  # Increased entropy for more exploration
        num_learning_epochs=8,  # More learning epochs
        num_mini_batches=4,
        learning_rate=3.0e-4,  # 3.0e-4, -> 1.0e-4 Slightly lower learning rate for stability
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
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Slightly lower entropy for more stable policy
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )