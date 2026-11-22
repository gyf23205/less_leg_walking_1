"""Network settings shared by every method, so the methods stay comparable.

Each method's main trainable action network (the "actor MLP") uses these values:
    scratch   -> ActorCritic actor            (rsl_rl_ppo_cfg.make_policy_cfg("nominal"))
    residual  -> the new residual MLP         (res_net.ResCfg.actor_hidden_dims)
    componet  -> the per-task internal policy (CompoNet.CompoCfg.internal_policy_hidden_dims)
    moe (KAE) -> the residual MLP             (MoE.MoECfg.actor_hidden_dims)
    FAME      -> fast and meta actors         (skrl_fame_cfg.yaml hidden_dims; checked in train_FAME.py)
Every critic uses CRITIC_HIDDEN_DIMS. Method-specific extras (attention heads, KAE
experts and gates, frozen earlier policies) are added on top and are not covered here.

Change a value here, never in a single method's config.
"""

ACTOR_HIDDEN_DIMS = [512, 256, 128]
CRITIC_HIDDEN_DIMS = [512, 256, 128]
ACTIVATION = "elu"
# Initial std of the Gaussian action noise (rsl_rl methods; FAME learns a state-dependent std).
INIT_NOISE_STD = 1.0
