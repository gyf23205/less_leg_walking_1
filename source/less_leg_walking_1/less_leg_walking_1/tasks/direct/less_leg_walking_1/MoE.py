from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class MoECfg(RslRlPpoActorCriticCfg):
    """Configuration for the custom MoE policy."""
    padded_dim: int = 256
    observable_dim: int = 32
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    # kae_path: str = "/home/yifan/git/less_leg_walking_1/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/KAE_original_range.pth"
    kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/new_bound2.pth"
    device: str = "cuda"
    n_experts: int = 1
    p: int = 1
    class_name: str = "MoEActorCritic"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    activation: str = "elu"
    
    # Set explicitly (don't reference other fields)
    # num_actor_obs: int = 256  # Match padded_dim or your obs space
    # num_critic_obs: int = 256
    # num_actions: int = 9
    
    # obs_groups = {
    #     "policy": ["policy"],
    #     "critic": ["policy"],
    # }

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_experts_outputs, extend_experts_outputs
from .Autoencoder import KoopmanAutoencoder_walk
from torch.serialization import add_safe_globals
from rsl_rl.modules import ActorCritic
try:
    from tensordict import TensorDictBase
except ImportError:  # fallback if tensordict version differs
    from tensordict.tensordict import TensorDictBase

# Allowlist for safe loading
add_safe_globals([KoopmanAutoencoder_walk])

import sys
from importlib import import_module
# alias old module path used in checkpoint
sys.modules.setdefault(
    "Autoencoder",
    import_module("less_leg_walking_1.tasks.direct.less_leg_walking_1.Autoencoder"),
)

class MoEActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, n_experts=None, **kwargs):  # Accept additional kwargs from cfg
        # DEBUG
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(num_actor_obs)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # assert False
        self.n_experts = n_experts
       
        # Extract custom params from kwargs to avoid conflicts
        # self.raw_obs_dim = kwargs.pop('raw_obs_dim', 226)
        self.observable_dim = kwargs.pop('observable_dim')
        self.actor_hidden_dims = kwargs.pop('actor_hidden_dims')
        self.critic_hidden_dims = kwargs.pop('critic_hidden_dims')
        self.padded_dim = kwargs.pop('padded_dim')
        # self.obs_range = [(torch.inf, -torch.inf) for _ in range(self.padded_dim)]
        # activation = kwargs.pop("activation", "elu")
        self.act_dim = num_actions
        self.kae_path = kwargs.pop('kae_path')
        self.device = kwargs.pop('device')
        # self.n_experts = kwargs.pop('n_experts', 1)
        self.p = kwargs.pop('p')
        activation = kwargs.pop("activation")
        

        raw_init_noise_std = kwargs.pop("init_noise_std", 0.8)
        if isinstance(raw_init_noise_std, dict):
            init_noise_std = raw_init_noise_std.get("value", 0.8)
        else:
            init_noise_std = raw_init_noise_std



        actor_obs_norm = bool(kwargs.pop("actor_obs_normalization", True))
        critic_obs_norm = bool(kwargs.pop("critic_obs_normalization", True))

        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            activation=activation,
            actor_hidden_dims=self.actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
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

        # remove self.distribution assignment

        # Load and freeze KAE
        self.kae = torch.load(self.kae_path, map_location=self.device, weights_only=False)
        for param in self.kae.parameters():
            param.requires_grad = False
        
        # DEBUG: why need to create actor and critic with Sequential but not directly using KAEAutoencoder_walk?
        # Define trainable MoE layers for actor (outputs mean and std for actions)
        actor_layers = []

        ########
        input_dim = self.padded_dim
        # input_dim = self.padded_dim + self.observable_dim

        for h in self.actor_hidden_dims:
            actor_layers.append(nn.Linear(input_dim, h))
            actor_layers.append(nn.ELU())
            input_dim = h

        actor_layers.append(nn.Linear(input_dim, self.observable_dim+self.act_dim))  # weights for experts
        ########
        self.actor = nn.Sequential(*actor_layers)
        
        # Define critic network (value head)
        critic_layers = []

        ########
        input_dim = self.padded_dim
        # input_dim = self.padded_dim + self.observable_dim
        ########
        
        for h in self.critic_hidden_dims:
            critic_layers.append(nn.Linear(input_dim, h))
            critic_layers.append(nn.ELU())
            input_dim = h
        critic_layers.append(nn.Linear(input_dim, 1))  # Single value output
        self.critic = nn.Sequential(*critic_layers)

        self._cached_mu: torch.Tensor | None = None
        self._cached_sigma: torch.Tensor | None = None

    def _extract_obs_tensor(self, obs):
        if isinstance(obs, TensorDictBase):
            keys = list(obs.keys())
            if "obs" in obs.keys():
                tensor = obs.get("obs")
            elif "policy" in obs.keys():
                tensor = obs.get("policy")
            else:
                tensor = obs.get(keys[0])
        elif isinstance(obs, dict):
            if "obs" in obs:
                tensor = obs["obs"]
            elif "policy" in obs:
                tensor = obs["policy"]
            else:
                tensor = next(iter(obs.values()))
        else:
            tensor = obs
        return tensor

    def _pad_to_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] < self.padded_dim:
            pad_size = self.padded_dim - tensor.shape[-1]
            return F.pad(tensor, (0, pad_size), value=1.0)
        return tensor[..., : self.padded_dim]

    def _prep_obs(self, obs, for_critic: bool = False, return_raw: bool = False):
        obs_tensor = self._extract_obs_tensor(obs).to(self.device, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        raw_obs = self._pad_to_dim(obs_tensor).clone() if return_raw else None

        if for_critic and self._use_critic_obs_norm:
            normalized_obs = self.critic_obs_normalizer(obs_tensor)
        elif not for_critic and self._use_actor_obs_norm:
            normalized_obs = self.actor_obs_normalizer(obs_tensor)
        else:
            normalized_obs = obs_tensor

        normalized_obs = self._pad_to_dim(normalized_obs)

        if return_raw:
            return normalized_obs, raw_obs
        return normalized_obs

    def forward(self, obs):
        padded_obs = self._prep_obs(obs)
        # # Record the maximum range of the padded observations on every channel among all batches
        # obs_range_temp = [(padded_obs[..., i].min().item(), padded_obs[..., i].max().item()) for i in range(padded_obs.shape[-1])]
        # self.obs_range = [(min(self.obs_range[i][0], obs_range_temp[i][0]), max(self.obs_range[i][1], obs_range_temp[i][1])) for i in range(len(obs_range_temp))]
      
        with torch.no_grad():  
            _, latent_z, _ = self.kae(padded_obs)

            latent_z = latent_z.detach()
            if latent_z.ndim == 1:
                latent_z = latent_z.unsqueeze(0)

            experts_outputs = get_experts_outputs(self.kae, latent_z, self.p, self.act_dim)# (Batch, observable_dim, act_dim*2), the last dimension is mean and std
        # print(f"experts_outputs shape: {experts_outputs.shape}")
        extended_experts_outputs = extend_experts_outputs(experts_outputs, self.act_dim)

        weights = self.actor(padded_obs) # isn't this should be pure observation + (KAE output + action_one_hot)?
        # print(f"weights shape: {weights.shape}")
        #######
        # weights = torch.softmax(weights, dim=-1)
        # weights = torch.tanh(weights)
        #######
        
        outputs = torch.sum(weights.view(-1, self.observable_dim+self.act_dim, 1) * extended_experts_outputs, dim=1)
        # print(f"outputs shape: {outputs.shape}")
        mu = outputs[..., : self.act_dim]
        # print(f"mu shape: {mu.shape}")
        # assert False

        # Clamp log-std to a safe range, then exponentiate and sanitize
        log_std = torch.clamp(outputs[..., self.act_dim:], min=-20.0, max=2.0)
        sigma = torch.exp(log_std)
        sigma = torch.clamp(sigma, min=1e-5, max=5.0)
        sigma = torch.where(torch.isfinite(sigma), sigma, torch.full_like(sigma, 1e-3))

        value = self.critic(padded_obs)  # keep shape [B, 1]
        return mu, sigma, value

    def get_value(self, obs):
        _, _, value = self.forward(obs)
        return value  # [B, 1]


    def update_distribution(self, obs, masks=None, hidden_states=None):
        # Use your MoE forward to define the Gaussian policy
        mu, sigma, _ = self.forward(obs)  # [B, act_dim]
        self._action_mean = mu
        self._action_std = sigma
        self.distribution = torch.distributions.Normal(mu,sigma)
        # print(self._cached_mu.requires_grad, self._cached_sigma.requires_grad, flush=True)

    def evaluate(self, obs, actions=None, masks=None, hidden_states=None):
        value = self.get_value(obs)
        if actions is None:
            return value

        mu, sigma, _ = self.forward(obs)

        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        print(f"[DEBUG] Log probabilities: {log_prob.mean().item()}, Entropy: {entropy.mean().item()}")
        print(f"[DEBUG] log_prob requires_grad: {log_prob.requires_grad}")

        return value, log_prob, entropy

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