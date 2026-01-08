from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class MoECfg(RslRlPpoActorCriticCfg):
    """Configuration for the custom MoE policy."""
    padded_dim: int = 256
    observable_dim: int = 16
    actor_hidden_dims: list[int] = [512, 256, 128]
    # actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [512, 256, 128]
    # kae_path: str = "/home/yifan/git/less_leg_walking_1/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/KAE_original_range.pth"
    # kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/new_bound2.pth"
    # kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/ForMOE_p1_pad256_obv64.pth"
    kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/ForMOE_p1_pad256_obv16.pth"
    device: str = "cuda"
    n_experts: int = 1
    p: int = 1
    class_name: str = "MoEActorCritic"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
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
from torch.distributions import Normal
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
    def __init__(self, obs, obs_groups, num_actions, n_experts=None, **kwargs):  # Accept additional kwargs from cfg
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

        # get the observation dimensions
        self.obs_groups = obs_groups
        self.num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            self.num_actor_obs += obs[obs_group].shape[-1]
        self.num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            self.num_critic_obs += obs[obs_group].shape[-1]
        

        raw_init_noise_std = kwargs.pop("init_noise_std", 0.8)
        if isinstance(raw_init_noise_std, dict):
            init_noise_std = raw_init_noise_std.get("value", 0.8)
        else:
            init_noise_std = raw_init_noise_std



        actor_obs_norm = bool(kwargs.pop("actor_obs_normalization", True))
        critic_obs_norm = bool(kwargs.pop("critic_obs_normalization", True))

        super().__init__(
            obs,
            obs_groups,
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
        
        # Define trainable MoE layers for actor
        actor_layers = []

        ########
        # input_dim = self.padded_dim
        input_dim = self.num_actor_obs
        # input_dim = self.observable_dim

        for h in self.actor_hidden_dims:
            actor_layers.append(nn.Linear(input_dim, h))
            actor_layers.append(nn.ELU())
            input_dim = h

        actor_layers.append(nn.Linear(input_dim, self.observable_dim+self.act_dim))  # weights for experts
        ########
        self.actor = nn.Sequential(*actor_layers)

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

    def forward(self, obs): # DEBUG Override all the functions that need actions.
        
        temp = obs.size()
        # print(temp[1])
        assert temp[1]==235, "observation is not 235 dim"
        padded_obs = self._prep_obs(obs)
        with torch.no_grad():  
            _, latent_z, _ = self.kae(padded_obs)

            latent_z = latent_z.detach()
            if latent_z.ndim == 1:
                latent_z = latent_z.unsqueeze(0)

            experts_outputs = get_experts_outputs(self.kae, latent_z, self.p, self.act_dim)# (Batch, observable_dim, act_dim)
            extended_experts_outputs = extend_experts_outputs(experts_outputs, self.act_dim)


        # print(experts_outputs.shape)
        # print(extended_experts_outputs.size())
        
        # weights = self.actor(padded_obs) # isn't this should be pure observation + (KAE output + action_one_hot)?
        weights = self.actor(obs)
        
        outputs = torch.sum(weights.view(-1, self.observable_dim+self.act_dim, 1) * extended_experts_outputs, dim=1)
        actions = outputs[..., : self.act_dim]
        return actions


    def update_distribution(self, obs, masks=None, hidden_states=None):
        # Use your MoE forward to define the Gaussian policy
        mean = self.forward(obs)  # [B, act_dim]

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.forward(obs)

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