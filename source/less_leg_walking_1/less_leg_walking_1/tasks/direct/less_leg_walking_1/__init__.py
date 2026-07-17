# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
##
# Register Gym environments.
##

gym.register(
    id="Less-Leg-Flat-Walking-Direct-v1",
    entry_point=f"{__name__}.less_leg_walking_1_env:LessLegWalkingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.less_leg_walking_1_env_cfg:LessLegWalkingFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LessLegWalkingFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_tdcls_cfg_entry_point": f"{agents.__name__}:skrl_tdcls_cfg.yaml",
        "skrl_fame_cfg_entry_point": f"{agents.__name__}:skrl_fame_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Less-Leg-Rough-Walking-Direct-v1",
    entry_point=f"{__name__}.less_leg_walking_1_env:LessLegWalkingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.less_leg_walking_1_env_cfg:LessLegWalkingRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LessLegWalkingRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_tdcls_cfg_entry_point": f"{agents.__name__}:skrl_tdcls_cfg.yaml",
        "skrl_fame_cfg_entry_point": f"{agents.__name__}:skrl_fame_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Less-AnymalC-Rough-Walking-Direct-v1",
    entry_point=f"{__name__}.less_leg_walking_1_env:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.less_leg_walking_1_env_cfg:AnymalCRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_tdcls_cfg_entry_point": f"{agents.__name__}:skrl_tdcls_cfg.yaml",
        "skrl_fame_cfg_entry_point": f"{agents.__name__}:skrl_fame_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Less-AnymalC-Flat-Walking-Direct-v1",
    entry_point=f"{__name__}.less_leg_walking_1_env:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.less_leg_walking_1_env_cfg:AnymalCFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_tdcls_cfg_entry_point": f"{agents.__name__}:skrl_tdcls_cfg.yaml",
        "skrl_fame_cfg_entry_point": f"{agents.__name__}:skrl_fame_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Less-AnymalC-Jump-Direct-v1",
    entry_point=f"{__name__}.less_leg_walking_1_env:AnymalJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.less_leg_walking_1_env_cfg:AnymalJumpEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalJumpFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_tdcls_cfg_entry_point": f"{agents.__name__}:skrl_tdcls_cfg.yaml",
        "skrl_fame_cfg_entry_point": f"{agents.__name__}:skrl_fame_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Less-AnymalC-Jump-Rough-Direct-v1",
    entry_point=f"{__name__}.less_leg_walking_1_env:AnymalJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.less_leg_walking_1_env_cfg:AnymalJumpRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalJumpRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_tdcls_cfg_entry_point": f"{agents.__name__}:skrl_tdcls_cfg.yaml",
        "skrl_fame_cfg_entry_point": f"{agents.__name__}:skrl_fame_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)