from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class MoECfg(RslRlPpoActorCriticCfg):
    """Configuration for the custom MoE policy."""
    padded_dim: int = 256
    observable_dim: int = 64
    hidden_dim_moe: list[int] = [512, 256, 128]
    # kae_path: str = "/home/yifan/git/less_leg_walking_1/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/temp_new2.pth"
    kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/new_bound2.pth"
    device: str = "cuda"
    n_experts: int = 1
    p: int = 1
    class_name: str = "MoEActorCritic"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    
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
       
        self.n_experts = n_experts
       
        # Extract custom params from kwargs to avoid conflicts
        self.observable_dim = kwargs.pop('observable_dim', 64)
        self.hidden_dim_moe = kwargs.pop('hidden_dim_moe', [512, 256, 128])
        self.padded_dim = kwargs.pop('padded_dim', 256)
        self.obs_range = [(torch.inf, -torch.inf) for _ in range(self.padded_dim)]
        # activation = kwargs.pop("activation", "elu")
        self.act_dim = num_actions
        # self.kae_path = kwargs.pop('kae_path', "/home/yifan/git/less_leg_walking_1/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/temp_new2.pth")
        # self.kae_path = kwargs.pop('kae_path', "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/new_bound2.pth")
        self.kae_path = kwargs.pop('kae_path', "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/ForMOE_p1_pad256_obv16.pth")
        self.device = kwargs.pop('device', "cuda")
        # self.n_experts = kwargs.pop('n_experts', 1)
        self.p = kwargs.pop('p', 1)
        raw_activation = kwargs.pop("activation", "elu")
        if isinstance(raw_activation, dict):
            activation = raw_activation.get("name", "elu")
        else:
            activation = raw_activation

        raw_actor_hidden = kwargs.pop("actor_hidden_dims", [512, 256, 128])
        if isinstance(raw_actor_hidden, dict):
            actor_hidden_dims = raw_actor_hidden.get("units", [])
        else:
            actor_hidden_dims = raw_actor_hidden
        if not actor_hidden_dims:
            actor_hidden_dims = [512, 256, 128]

        raw_critic_hidden = kwargs.pop("critic_hidden_dims", [512, 256, 128])
        if isinstance(raw_critic_hidden, dict):
            critic_hidden_dims = raw_critic_hidden.get("units", [])
        else:
            critic_hidden_dims = raw_critic_hidden
        if not critic_hidden_dims:
            critic_hidden_dims = [512, 256, 128]

        raw_init_noise_std = kwargs.pop("init_noise_std", 0.8)
        if isinstance(raw_init_noise_std, dict):
            init_noise_std = raw_init_noise_std.get("value", 0.8)
        else:
            init_noise_std = raw_init_noise_std

        # obs_groups = kwargs.pop("obs_groups", {
        #     "policy": ["policy"],
        #     "critic": ["policy"],
        # })

        actor_obs_norm = bool(kwargs.pop("actor_obs_normalization", True))
        critic_obs_norm = bool(kwargs.pop("critic_obs_normalization", True))

        # print(f"actor_obs_normalization: {actor_obs_norm}")
        # print(f"critic_obs_normalization: {critic_obs_norm}")
        # assert False, "Check actor_obs_normalization and critic_obs_normalization values"
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

        # remove self.distribution assignment

        # Load and freeze KAE
        self.kae = torch.load(self.kae_path, map_location=self.device, weights_only=False)
        for param in self.kae.parameters():
            param.requires_grad = False
        
        # Define trainable MoE layers for actor (outputs mean and std for actions)
        actor_layers = []
        input_dim = self.padded_dim

        for h in self.hidden_dim_moe:
            actor_layers.append(nn.Linear(input_dim, h))
            actor_layers.append(nn.ELU())
            input_dim = h

        actor_layers.append(nn.Linear(input_dim, self.observable_dim))  # weights for experts
        self.actor = nn.Sequential(*actor_layers)
        
        # Define critic network (value head)
        critic_layers = []
        input_dim = self.padded_dim

        
        for h in self.hidden_dim_moe:
            critic_layers.append(nn.Linear(input_dim, h))
            critic_layers.append(nn.ELU())
            input_dim = h
        critic_layers.append(nn.Linear(input_dim, 1))  # Single value output
        self.critic = nn.Sequential(*critic_layers)

        self._cached_mu: torch.Tensor | None = None
        self._cached_sigma: torch.Tensor | None = None

    # @property
    # def action_mean(self):
    #     if self._cached_mu is None:
    #         raise RuntimeError("Action mean unavailable. Call act() first.")
    #     return self._cached_mu

    # @property
    # def action_std(self):
    #     if self._cached_sigma is None:
    #         raise RuntimeError("Action std unavailable. Call act() first.")
    #     return self._cached_sigma

    # @property
    # def entropy(self):
    #     if self._cached_mu is None or self._cached_sigma is None:
    #         raise RuntimeError("Entropy unavailable. Call act() first.")
    #     dist = torch.distributions.Normal(self._cached_mu, self._cached_sigma)
    #     return dist.entropy().sum(dim=-1)

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

    def _prep_obs(self, obs):
        obs_tensor = self._extract_obs_tensor(obs).to(self.device, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.shape[-1] < self.padded_dim:
            pad_size = self.padded_dim - obs_tensor.shape[-1]
            padded_obs = F.pad(obs_tensor, (0, pad_size), value=1.0)
        else:
            padded_obs = obs_tensor[..., : self.padded_dim]
        return padded_obs

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
        # extended_experts_outputs = extend_experts_outputs(experts_outputs, self.act_dim)


        weights = self.actor(padded_obs) # isn't this should be pure observation + (KAE output + action_one_hot)?

        #######
        # weights = torch.softmax(weights, dim=-1)
        weights = torch.tanh(weights)
        #######
        
        
        # print(f"weights shape: {weights.shape}")
        outputs = torch.sum(weights.view(-1, self.observable_dim, 1) * experts_outputs, dim=1)
       

        mu = outputs[..., : self.act_dim]
        sigma = torch.clamp(torch.exp(outputs[..., self.act_dim:]), min=1e-6, max=5.0)
        value = self.critic(padded_obs)  # keep shape [B, 1]
        # print(f"action shape:{outputs.shape}", f"mu  shape:{mu.shape}", f"sigma shape:{sigma.shape}", f"value shape:{value.shape}")
        # assert False

        return mu, sigma, value

    def get_value(self, obs):
        _, _, value = self.forward(obs)
        return value  # [B, 1]

    # def get_value(self, obs):
    #     padded_obs = self._prep_obs(obs)
    #     return self.critic(padded_obs)  # shape [B, 1]

    # def act(self, obs, masks=None, hidden_states=None, deterministic=False):
    #     with torch.no_grad():
    #         mu, sigma, _ = self.forward(obs)
    #         self._cached_mu = mu.detach()
    #         self._cached_sigma = sigma.detach()
    #         dist = torch.distributions.Normal(self._cached_mu, self._cached_sigma)
    #         actions = dist.mean if deterministic else dist.sample()
    #     return actions

    # def act(self, obs, masks=None, hidden_states=None, deterministic=False):
    #     mu, sigma, _ = self.forward(obs)     # ✅ keep graph
    #     self._cached_mu = mu
    #     self._cached_sigma = sigma
    #     dist = torch.distributions.Normal(mu, sigma)
    #     actions = dist.mean if deterministic else dist.sample()
    #     return actions

    # def get_actions_log_prob(self, actions):
    #     if self._cached_mu is None or self._cached_sigma is None:
    #         raise RuntimeError("Call act() before querying log-prob.")
    #     dist = torch.distributions.Normal(self._cached_mu, self._cached_sigma)
    #     return dist.log_prob(actions).sum(dim=-1)

    # def get_actions_entropy(self):
    #     if self._cached_mu is None or self._cached_sigma is None:
    #         raise RuntimeError("Call act() before querying entropy.")
    #     dist = torch.distributions.Normal(self._cached_mu, self._cached_sigma)
    #     return dist.entropy().sum(dim=-1)

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
        return value, log_prob, entropy