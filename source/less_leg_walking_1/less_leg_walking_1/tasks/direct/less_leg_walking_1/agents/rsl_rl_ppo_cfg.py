# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import yaml

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# ############################################
# ALGO_METHOD is the single switch that selects the method for every runner config
# below. It drives BOTH:
#   * the policy / actor-critic cfg  -> make_policy_cfg()
#   * the PPO algorithm hyperparams  -> load_algorithm_cfg() reads algo_cfg/<method>.yaml
# Each algo_cfg/<method>.yaml holds a per-terrain block ("flat" / "rough" / "jump");
# every runner config picks its own terrain, while ALGO_METHOD picks the file.
#   "res"     -> ResCfg          + algo_cfg/res.yaml      (residual)
#   "nominal" -> from-scratch AC + algo_cfg/nominal.yaml  (train from scratch)
#   "moe"     -> MoECfg          + algo_cfg/moe.yaml
#   "compo"   -> CompoCfg        + algo_cfg/compo.yaml
ALGO_METHOD = "compo"

_ALGO_CFG_DIR = os.path.join(os.path.dirname(__file__), "algo_cfg")


def make_policy_cfg(method: str = ALGO_METHOD):
    """Return the policy / actor-critic cfg for the chosen method.

    Model modules are imported lazily so that selecting one method does not force
    importing the others (matching the old comment/uncomment-the-import workflow).
    The corresponding ActorCritic class is registered by the train script, not here.
    """
    if method == "res":
        from ..res_net import ResCfg  # noqa: F401  (ResActorCritic is injected by the train script)
        return ResCfg()
    if method == "nominal":
        return RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
    if method == "moe":
        from ..MoE import MoECfg
        return MoECfg()
    if method == "compo":
        from ..baselines.CompoNet import CompoCfg
        return CompoCfg()
    raise ValueError(f"Unknown ALGO_METHOD {method!r}; expected 'res', 'nominal', 'moe' or 'compo'")


def load_algorithm_cfg(terrain: str, method: str = ALGO_METHOD) -> RslRlPpoAlgorithmCfg:
    """Build an RslRlPpoAlgorithmCfg from the <terrain> block of algo_cfg/<method>.yaml.

    ``terrain`` is one of "flat", "rough", "jump" - it selects the per-terrain
    block within the chosen method's YAML file.
    """
    path = os.path.join(_ALGO_CFG_DIR, f"{method}.yaml")
    with open(path, "r") as f:
        params = yaml.safe_load(f)[terrain]
    return RslRlPpoAlgorithmCfg(**params)
# ############################################


@configclass
class LessLegWalkingFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500  # Increased for more training
    save_interval = 200
    experiment_name = "less_leg_walking_flat"
    empirical_normalization = False
    # resume = True

    # Policy is selected by ALGO_METHOD (see top of file): res / nominal / moe / compo
    policy = make_policy_cfg(ALGO_METHOD)

    algorithm = load_algorithm_cfg("flat")


@configclass
class LessLegWalkingRoughPPORunnerCfg(LessLegWalkingFlatPPORunnerCfg):
    experiment_name = "less_leg_walking_rough"
    max_iterations = 2500

    algorithm = load_algorithm_cfg("rough")

@configclass
class AnymalCFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500  # Increased for more training
    save_interval = 200
    experiment_name = "anymal_c_flat_leg_walking"
    empirical_normalization = False
    # resume = True

    # Policy is selected by ALGO_METHOD (see top of file): res / nominal / moe / compo
    policy = make_policy_cfg(ALGO_METHOD)

    algorithm = load_algorithm_cfg("flat")

@configclass
class AnymalCRoughPPORunnerCfg(AnymalCFlatPPORunnerCfg):
    experiment_name = "anymal_c_rough_leg_walking"
    max_iterations = 2500

    algorithm = load_algorithm_cfg("rough")

@configclass
class AnymalJumpFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "anymal_c_jump_flat"
    num_steps_per_env = 24
    max_iterations =4000  # Increased for more training
    save_interval = 200
    empirical_normalization = False

    # Policy is selected by ALGO_METHOD (see top of file): res / nominal / moe / compo
    policy = make_policy_cfg(ALGO_METHOD)

    algorithm = load_algorithm_cfg("jump")


@configclass
class AnymalJumpRoughPPORunnerCfg(AnymalJumpFlatPPORunnerCfg):
    experiment_name = "anymal_c_jump_rough"

    algorithm = load_algorithm_cfg("jump")