from pytest import param
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class MoECfg(RslRlPpoActorCriticCfg):
    """Configuration for the custom MoE policy."""
    padded_dim: int = 256
    observable_dim: int = 16
    actor_hidden_dims: list[int] = [512, 256, 128]
    # actor_hidden_dims: list[int] = [128, 64, 32]
    critic_hidden_dims: list[int] = [512, 256, 128]
    # critic_hidden_dims: list[int] = [1024, 512, 256, 128]

    # kae_path: str = "/home/yifan/git/less_leg_walking_1/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/KAE_original_range.pth"
    # kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/new_bound2.pth"
    kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/ForMOE_p1_pad256_obv16.pth"
    # kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/ForMOE_wtask_p1_pad256_obv16.pth"
    # kae_path: str = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/ForMOE_wdist_p1_pad256_obv16_v2.pth"

    device: str = "cuda"
    n_experts: int = 1
    p: int = 1
    class_name: str = "MoEActorCritic"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    activation: str = "elu"
    init_noise_std: float = 1.0  # ADD THIS - match train_scratch

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
        self.ext = True
        self.n_experts = n_experts
        self.training_steps = 0  # To track training steps for noise scheduling
       
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
        self.identity_weight_decay = kwargs.pop('identity_weight_decay', 0.01)
        self._prev_mlp_weights = None

        # get the observation dimensions
        self.obs_groups = obs_groups
        self.num_actor_obs = 0

        self._step_count = 0  # For tracking steps for KAE scaling

        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            self.num_actor_obs += obs[obs_group].shape[-1]

        self.num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            self.num_critic_obs += obs[obs_group].shape[-1]


        raw_init_noise_std = kwargs.pop("init_noise_std", 1.0)
        if isinstance(raw_init_noise_std, dict):
            init_noise_std = raw_init_noise_std.get("value", 1.0)
        else:
            init_noise_std = raw_init_noise_std

        actor_obs_norm = bool(kwargs.pop("actor_obs_normalization", False))  # Match config default
        critic_obs_norm = bool(kwargs.pop("critic_obs_normalization", False))  # Match config default

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

        # Load and freeze KAE
        self.kae = torch.load(self.kae_path, map_location=self.device, weights_only=False)
        for param in self.kae.parameters():
            param.requires_grad = False

        self.kae.eval()

               # 1. MLP Network (learns residual correction)
        mlp_layers = []
        input_dim = self.num_actor_obs
        for h in [64, 32]:
            mlp_layers.append(nn.Linear(input_dim, h))
            mlp_layers.append(nn.ELU())
            input_dim = h
        mlp_layers.append(nn.Linear(input_dim, self.act_dim))
        self.mlp_network = nn.Sequential(*mlp_layers)
        
        # 2. Expert Weight Network (learns how to use KAE experts)
        expert_weight_layers = []
        input_dim = self.num_actor_obs
        for h in [32]:
            expert_weight_layers.append(nn.Linear(input_dim, h))
            expert_weight_layers.append(nn.ELU())
            input_dim = h
        expert_weight_layers.append(nn.Linear(input_dim, self.observable_dim))
        self.expert_weight_network = nn.Sequential(*expert_weight_layers)
        
        # Initialize expert weights with bias toward 1.0
        with torch.no_grad():
            final_layer = self.expert_weight_network[-1]
            final_layer.weight.data.fill_(0.0)
            final_layer.bias.data = torch.ones(self.observable_dim)
        
        # 3. Gating Network (learns when to trust KAE vs MLP)
        gating_layers = []
        input_dim = self.num_actor_obs  + self.act_dim + self.act_dim
        for h in [64, 32]:
            gating_layers.append(nn.Linear(input_dim, h))
            gating_layers.append(nn.ELU())
            input_dim = h
        gating_layers.append(nn.Linear(input_dim, 1))
        self.gating_network = nn.Sequential(*gating_layers)

        # # Initialize the gate to strongly favor the KAE pathway at the start.
        # # A large positive bias means sigmoid(logit) will be close to 1.0.
        # with torch.no_grad():
        #     self.gating_network[-1].bias.data.fill_(1.0)
        #     # self.mlp_network[-1].weight.data.fill_(0.0)
        #     # self.mlp_network[-1].bias.data.fill_(0.0)


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

    def _prep_obs(self, obs, for_critic: bool = False, return_raw: bool = False, skip_norm: bool = False):
        obs_tensor = self._extract_obs_tensor(obs).to(self.device, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        raw_obs = self._pad_to_dim(obs_tensor).clone() if return_raw else None

        if skip_norm:
            normalized_obs = obs_tensor
        elif for_critic and self._use_critic_obs_norm:
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
        assert temp[1]==235, "observation is not 235 dim"

        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"BAD INPUT obs detected!")
            print(f"obs has NaN: {torch.isnan(obs).any()}")
            print(f"obs has Inf: {torch.isinf(obs).any()}")
            print(f"obs stats: min={obs.min()}, max={obs.max()}")   

        padded_obs = self._prep_obs(obs, skip_norm=True)

        with torch.no_grad():
            _, latent_z, _ = self.kae(padded_obs)
            if latent_z.ndim == 1:
                latent_z = latent_z.unsqueeze(0)
            experts_outputs = get_experts_outputs(self.kae, latent_z, self.p, self.act_dim)  # [B, 16, 12]
        
        # Get weights for each KAE expert using the original, unnormalized 'obs'
        expert_weights = self.expert_weight_network(obs)
        kae_actions = torch.sum(expert_weights.view(-1, self.observable_dim, 1) * experts_outputs, dim=1)
        
        # 2. MLP pathway (residual) - uses the original, unnormalized 'obs'
        mlp_actions = self.mlp_network(obs)
        
        # 3. Gate decides blending - uses the original, unnormalized 'obs'
        gating_input = torch.cat([obs, mlp_actions.detach(), kae_actions.detach()], dim=1)
        gate_logit = self.gating_network(gating_input)
        gate = torch.sigmoid(gate_logit)
        
        # 4. Blend the two pathways
        actions = kae_actions + gate * mlp_actions

        self.last_expert_weights = expert_weights.detach()
            
            
        return actions
            
       
    def act(self, obs, **kwargs):
        # Match base class exactly
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()
        
    def _update_distribution(self, obs, masks=None, hidden_states=None):
        # Use your MoE forward to define the Gaussian policy
        mean = self.forward(obs)  # [B, act_dim]

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        # print(f"std stats: min={std.min()}, max={std.max()}, has_nan={torch.isnan(std).any()}")
        # print(f"noise_std_type: {self.noise_std_type}")

        # create distribution
        self.distribution = Normal(mean, std)

        # # Log gradients periodically during training updates
        # if self.training and hasattr(self, 'training_steps') and self.training_steps % 100 == 0:
        #     # Register a hook to log gradients after backward pass
        #     if not hasattr(self, '_grad_hook_registered'):
        #         def grad_hook(grad):
        #             self._log_gradients_next = True
        #             return grad
        #         self.actor[-1].weight.register_hook(grad_hook)
        #         self._grad_hook_registered = True
            
        #     if hasattr(self, '_log_gradients_next') and self._log_gradients_next:
        #         self.log_gradient_stats()
        #         self._log_gradients_next = False

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

            if torch.isnan(param).any():
                print(f"Parameter {name} has NaN!")

    def log_gradients(self):
        """Log the gradients of the model's parameters."""
        print("[DEBUG] Executing log_gradients for MoEActorCritic")
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm().item()}")
            else:
                print(f"Gradient for {name}: None (parameter is not updated)")

    # def log_gradient_stats(self):
    #     """Log gradient statistics for the final actor layer."""
    #     final_layer = self.actor[-1]  # Last linear layer
        
    #     if final_layer.weight.grad is not None:
    #         with torch.no_grad():
    #             # Gradients for expert weight outputs (first observable_dim)
    #             expert_grad = final_layer.weight.grad[:self.observable_dim, :]
    #             expert_grad_norm = expert_grad.norm().item()
    #             expert_grad_mean = expert_grad.abs().mean().item()
                
    #             # Gradients for identity weight outputs (remaining act_dim)
    #             if self.ext:
    #                 identity_grad = final_layer.weight.grad[self.observable_dim:, :]
    #                 # Exclude RF leg dimensions (2, 6, 10)
    #                 active_dims = [i for i in range(self.act_dim) if i not in [2, 6, 10]]
    #                 identity_grad_active = identity_grad[active_dims, :]
    #                 identity_grad_norm = identity_grad_active.norm().item()
    #                 identity_grad_mean = identity_grad_active.abs().mean().item()
    #             else:
    #                 identity_grad_norm = 0.0
    #                 identity_grad_mean = 0.0
                
    #             # Bias gradients
    #             if final_layer.bias.grad is not None:
    #                 expert_bias_grad = final_layer.bias.grad[:self.observable_dim]
    #                 expert_bias_grad_mean = expert_bias_grad.abs().mean().item()
                    
    #                 if self.ext:
    #                     identity_bias_grad = final_layer.bias.grad[self.observable_dim:]
    #                     identity_bias_grad_active = identity_bias_grad[active_dims]
    #                     identity_bias_grad_mean = identity_bias_grad_active.abs().mean().item()
    #                 else:
    #                     identity_bias_grad_mean = 0.0
                
    #             print(f"\n[Gradient Stats - Step {self.training_steps}]")
    #             print(f"Expert weight grad - norm: {expert_grad_norm:.4f}, mean_abs: {expert_grad_mean:.6f}")
    #             print(f"Identity weight grad - norm: {identity_grad_norm:.4f}, mean_abs: {identity_grad_mean:.6f}")
    #             print(f"Expert bias grad mean_abs: {expert_bias_grad_mean:.6f}")
    #             if self.ext:
    #                 print(f"Identity bias grad mean_abs: {identity_bias_grad_mean:.6f}")
    #             print(f"Expert/Identity gradient ratio: {expert_grad_mean / (identity_grad_mean + 1e-8):.2f}")