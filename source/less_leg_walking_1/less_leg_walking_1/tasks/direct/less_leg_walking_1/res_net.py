from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class ResCfg(RslRlPpoActorCriticCfg):
    """Configuration for the custom MoE policy."""
    # raw_obs_dim: int = 226
    # hidden_dim_moe: list[int] = [512, 256, 128]
    # original_policy_path: str = "/home/yifan/git/less_leg_walking_1/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/walking_policy_new.pth"
    original_policy_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/results/walking_policy_new.pth"
    obs_dim: int = 235
    device: str = "cuda"
    activation: str = "elu"
    class_name: str = "ResActorCritic"
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True

import torch
from torch.serialization import add_safe_globals
from rsl_rl.modules import ActorCritic
try:
    from tensordict import TensorDictBase
except ImportError:  # fallback if tensordict version differs
    from tensordict.tensordict import TensorDictBase

# import sys
from importlib import import_module

class ResActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, n_experts=None, **kwargs):  # Accept additional kwargs from cfg
        # print(kwargs)
        self.n_experts = n_experts
       
        # Extract custom params from kwargs to avoid conflicts
        self.obs_dim = kwargs.pop('obs_dim')
        # self.act_dim = num_actions
        self.original_policy_path = kwargs.pop('original_policy_path') #, "/home/yifan/git/less_leg_walking_1/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/original_policy.pth")
        self.device = kwargs.pop('device')
        activation = kwargs.pop('activation')
        actor_hidden_dims = kwargs.pop('actor_hidden_dims')
        critic_hidden_dims = kwargs.pop('critic_hidden_dims')

        raw_init_noise_std = kwargs.pop("init_noise_std", 0.8)
        if isinstance(raw_init_noise_std, dict):
            init_noise_std = raw_init_noise_std.get("value", 0.8)
        else:
            init_noise_std = raw_init_noise_std

        actor_obs_norm = bool(kwargs.pop("actor_obs_normalization"))
        critic_obs_norm = bool(kwargs.pop("critic_obs_normalization"))

        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            activation=activation,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_obs_normalization=actor_obs_norm,
            critic_obs_normalization=critic_obs_norm,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        # Ensure gradients are enabled for actor and critic
        for param in self.actor.parameters():
            param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True

        # Add a utility to check gradients
        self._check_gradients_enabled()

        self._use_actor_obs_norm = hasattr(self, "actor_obs_normalizer") and self.actor_obs_normalizer is not None
        self._use_critic_obs_norm = hasattr(self, "critic_obs_normalizer") and self.critic_obs_normalizer is not None

        # Load and freeze original policy
        self.original_policy = torch.load(self.original_policy_path, map_location=self.device, weights_only=False)["actor"]
        for param in self.original_policy.parameters():
            param.requires_grad = False


    def forward(self, obs):
        # obs = self.get_actor_obs(obs)
        with torch.no_grad():  
            outputs_init = self.original_policy(obs) # Get hint action from original policy
        res= self.actor(obs) # Get residual from ResPolicy
        # # Pad to 12-dim and keep gradients
        # padded_res = torch.zeros((res.shape[0], 12), device=res.device, dtype=res.dtype)
        # padded_res[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]] = res  # Map 9-dim to 12-dim
        # res = padded_res

        actions = outputs_init + res
        # print(f"outputs shape:{outputs.shape}")
        # mu = outputs[..., : self.act_dim]
        # print(f"mu shape:{mu.shape}")
        # sigma = torch.clamp(torch.exp(outputs[..., self.act_dim:] + res[..., self.act_dim:]), min=1e-6, max=5.0)

        value = self.critic(obs)  # keep shape [B, 1]
        # print(f"action shape:{outputs.shape}", f"mu  shape:{mu.shape}", f"sigma shape:{sigma.shape}", f"value shape:{value.shape}")
        # assert False

        return actions, value

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value  # [B, 1]

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        actions, _ = self.forward(obs)
        return actions

    def update_distribution(self, obs, masks=None, hidden_states=None):
        # Use your MoE forward to define the Gaussian policy
        mu, _ = self.forward(obs)  # [B, act_dim]
        # print(f"mu shape in update_distribution: {mu.shape}")
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mu)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mu)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = torch.distributions.Normal(mu,std)

    def evaluate(self, obs, actions=None, masks=None, hidden_states=None):
        obs = self.get_actor_obs(obs)
        value = self.get_value(obs)
        return value
        # if actions is None:
        #     return value

        # mu, sigma, _ = self.forward(obs)

        # dist = torch.distributions.Normal(mu, sigma)
        # log_prob = dist.log_prob(actions).sum(dim=-1)
        # entropy = dist.entropy().sum(dim=-1)

        # print(f"[DEBUG] Log probabilities: {log_prob.mean().item()}, Entropy: {entropy.mean().item()}")
        # print(f"[DEBUG] log_prob requires_grad: {log_prob.requires_grad}")

        # return value, log_prob, entropy

    def _check_gradients_enabled(self):
        """Utility to check if gradients are enabled for model parameters."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Gradient enabled for: {name}")
            else:
                print(f"WARNING: Gradient disabled for: {name}")

    def log_gradients(self):
        """Log the gradients of the model's parameters."""
        print("[DEBUG] Executing log_gradients for MoEActorCritic")
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm().item()}")
            else:
                print(f"Gradient for {name}: None (parameter is not updated)")