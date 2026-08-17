import datetime

from pytest import param
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

import importlib.util
import os
from pathlib import Path

@configclass
class MoECfg(RslRlPpoActorCriticCfg):
    """Configuration for the custom MoE policy."""
    padded_dim: int = 256
    observable_dim: int = 16
    actor_hidden_dims: list[int] = [256, 128, 64] # Residual net
    # actor_hidden_dims: list[int] = [128, 64, 32]
    critic_hidden_dims: list[int] = [512, 256, 128]
    gating_hidden_dims: list[int] = [64, 32] #[64, 32] # gating network
    weight_hidden_dims: list[int] = [32, 16] # wieght network
    # critic_hidden_dims: list[int] = [1024, 512, 256, 128]
    

    # Experiment log    
                            #  res                        g          gbias                w                NOTE
    # 2026-08-12_22-14-44     [512, 128, 64]             [16]        none            [256, 128, 64] 
    # 2026-08-13_09-30-45     [512, 128, 64]             [16]        1               [16] 
    # 2026-08-13_16-53-54     [512, 256, 128]            [8]         none            [64]
    # Consistent problem on less-leg-roguh and jump-rough; but works well for less-leg-flat
    #                         ''                          "0"        none             ''
    #                         [512, 256, 128]            [8]         none            [64]                  w part fix
    # Gate network now does not take KAE and res output as input
    #

    # kae_path: str = "/home/yifan/git/less_leg_walking_1/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/KAEs/ForMOE_p1_pad256_obv16.pth"
    kae_path: str = "/home/joonwon/github/less_leg_walking_1.worktrees/origin-master/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/KAEs/ForMOE_p1_pad256_obv16.pth"
   
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
        self._fwd = 0
        from datetime import datetime
        self._f = open(f"/tmp/gate_{datetime.now():%m%d_%H%M%S}.csv", "w", buffering=1)
        self._temporal_input_dim = 0
       
        # Extract custom params from kwargs to avoid conflicts
        # self.raw_obs_dim = kwargs.pop('raw_obs_dim', 226)
        self.observable_dim = kwargs.pop('observable_dim')
        self.actor_hidden_dims = kwargs.pop('actor_hidden_dims')
        self.critic_hidden_dims = kwargs.pop('critic_hidden_dims')
        self.weight_hidden_dims = kwargs.pop('weight_hidden_dims')
        self.gating_hidden_dims = kwargs.pop('gating_hidden_dims')
        self.padded_dim = kwargs.pop('padded_dim')
        # self.obs_range = [(torch.inf, -torch.inf) for _ in range(self.padded_dim)]
        # activation = kwargs.pop("activation", "elu")
        self.act_dim = num_actions
        # Gate logging: running stats accumulated over each rollout, popped per log iteration
        self.last_gate = None
        self._gate_running_sum = 0.0
        self._gate_running_count = 0
        self.kae_path = kwargs.pop('kae_path')
        self.device = kwargs.pop('device')
        # self.n_experts = kwargs.pop('n_experts', 1)
        self.p = kwargs.pop('p')
        activation = kwargs.pop("activation")
        self.identity_weight_decay = kwargs.pop('identity_weight_decay', 0.01)
        self._prev_mlp_weights = None


        self.crl_mode = (
            os.environ.get("CRL_MODE") == "1"
        )

        self.crl_kae_paths = []

        if self.crl_mode:
            kae_directory = Path(os.environ["CRL_KAE_DIRECTORY"])

            task_names = os.environ["CRL_KAE_TASKS"].split("|")

            for task_name in task_names:
                if not task_name:
                    continue

                kae_file = (
                    kae_directory 
                    / f"{task_name}_KAE.pth"
                )

                if not kae_file.is_file():
                    raise FileNotFoundError(str(kae_file))

                self.crl_kae_paths.append(kae_file)

            if not self.crl_kae_paths:
                raise RuntimeError("No previous-task KAE was provided.")

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

        # # Load and freeze KAE
        # self.kae = torch.load(self.kae_path, map_location=self.device, weights_only=False)
        # for param in self.kae.parameters():
        #     param.requires_grad = False

        # self.kae.eval()

        if not self.crl_mode:
            # Original standalone behavior.
            self.kae = torch.load(
                self.kae_path,
                map_location=self.device,
                weights_only=False,
            )

            for param in self.kae.parameters():
                param.requires_grad = False

            self.kae.eval()

            self.total_modes = (self.observable_dim)

        else:
            kae_approx_file = os.environ["CRL_KAE_APPROX_FILE"]

            module_spec = (importlib.util.spec_from_file_location("crl_kae_approx",kae_approx_file,))

            if (module_spec is None
                or module_spec.loader is None):
                raise ImportError("Unable to load KAE_approx.py: "+ kae_approx_file)

            kae_module = (importlib.util.module_from_spec(module_spec))
            sys.modules[module_spec.name] = (kae_module)
            module_spec.loader.exec_module(kae_module)            
            self.kaes = nn.ModuleList()

            for kae_file in self.crl_kae_paths:
                kae = kae_module.load_kae(
                    kae_file,
                    self.device,
                )

                for param in kae.parameters():
                    param.requires_grad = False

                kae.eval()
                self.kaes.append(kae)

            self.total_modes = sum(
                kae.K.shape[0]
                for kae in self.kaes
            )

            self.n_experts = len(self.kaes)

        # 1. MLP Network (learns residual correction)
        mlp_layers = []
        input_dim = self.num_actor_obs
        for h in self.actor_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, h))
            mlp_layers.append(nn.ELU())
            input_dim = h
        mlp_layers.append(nn.Linear(input_dim, self.act_dim))
        self.mlp_network = nn.Sequential(*mlp_layers)
        
        # # 2. Expert Weight Network (learns how to use KAE experts)
        # expert_weight_layers = []
        # input_dim = self.num_actor_obs
        # for h in self.weight_hidden_dims:
        #     expert_weight_layers.append(nn.Linear(input_dim, h))
        #     expert_weight_layers.append(nn.ELU())
        #     input_dim = h
        # # expert_weight_layers.append(nn.Linear(input_dim, self.observable_dim))
        # # self.expert_weight_network = nn.Sequential(*expert_weight_layers)
        
        # # # Initialize expert weights with bias toward 1.0
        # # with torch.no_grad():
        # #     final_layer = self.expert_weight_network[-1]
        # #     final_layer.weight.data.fill_(0.0)
        # #     final_layer.bias.data = torch.ones(self.observable_dim)

        # if self.crl_mode:
        #     expert_output_dim = (self.total_modes)
        # else:
        #     expert_output_dim = (
        #         self.observable_dim
        #     )

        # expert_weight_layers.append(
        #     nn.Linear(
        #         input_dim,
        #         expert_output_dim,
        #     )
        # )
        # self._temporal_input_dim = input_dim


        # self.expert_weight_network = nn.Sequential(
        #     *expert_weight_layers
        # )

        # 2. Expert weight networks: one independent trunk per KAE.
        #    weight_hidden_dims is now the size of a SINGLE task's network, so the
        #    hidden width never has to cover the accumulated mode count.
        #    Branch order follows self.kaes, matching the concat order of
        #    experts_outputs in forward().
        if self.crl_mode:
            branch_out_dims = [kae.K.shape[0] for kae in self.kaes]
        else:
            branch_out_dims = [self.observable_dim]

        self.expert_weight_networks = nn.ModuleList()
        for out_dim in branch_out_dims:
            layers = []
            input_dim = self.num_actor_obs
            for h in self.weight_hidden_dims:
                layers.append(nn.Linear(input_dim, h))
                layers.append(nn.ELU())
                input_dim = h
            layers.append(nn.Linear(input_dim, out_dim))
            self.expert_weight_networks.append(nn.Sequential(*layers))
        self._temporal_input_dim = input_dim

                
        # 3. Gating Network (learns when to trust KAE vs MLP)
        # self.act_dim, self._temporal_input_dim, self.total_modes
        gating_layers = []
        input_dim = self.num_actor_obs
        for h in self.gating_hidden_dims:
            gating_layers.append(nn.Linear(input_dim, h))
            gating_layers.append(nn.ELU())
            input_dim = h
        gating_layers.append(nn.Linear(input_dim, 1))

        self.gating_network = nn.Sequential(*gating_layers)

        with torch.no_grad():
            self.gating_network[-1].bias.data.fill_(1.0) # <- this is g_bias
        #     # self.mlp_network[-1].weight.data.fill_(0.0)
        #     # self.mlp_network[-1].bias.data.fill_(0.0)

        self.g_min = 0.0 # <- this is g_min
        self.g_max = 1.0 # <- this is g_max


        # KAE-level router. It allocates the unit L1 budget BETWEEN KAEs, while each
        # expert_weight_network (L1-normalised inside its own modes) only decides
        # WHICH modes within that KAE. Because the router is a softmax the total
        # budget stays 1, so the KMD branch magnitude is independent of the number
        # of stacked KAEs. With a single branch this reduces exactly to the previous
        # global-L1 behaviour.
        router_layers = []
        router_input_dim = self.num_actor_obs
        for hidden_dim in self.weight_hidden_dims:
            router_layers.append(nn.Linear(router_input_dim, hidden_dim))
            router_layers.append(nn.ELU())
            router_input_dim = hidden_dim
        router_layers.append(nn.Linear(router_input_dim, len(branch_out_dims)))
        self.kae_router_network = nn.Sequential(*router_layers)

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
        # print("obs size:", temp)
        assert temp[1]==235, "observation is not 235 dim"

        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"BAD INPUT obs detected!")
            print(f"obs has NaN: {torch.isnan(obs).any()}")
            print(f"obs has Inf: {torch.isinf(obs).any()}")
            print(f"obs stats: min={obs.min()}, max={obs.max()}")   

        padded_obs = self._prep_obs(obs, skip_norm=True)

        if not self.crl_mode:
            # Original standalone behavior.
            with torch.no_grad():
                _, latent_z, _ = self.kae(padded_obs)

                if latent_z.ndim == 1:
                    latent_z = (latent_z.unsqueeze(0))

                experts_outputs = (get_experts_outputs(self.kae,latent_z,self.p,self.act_dim,))
            expert_mode_count = (self.observable_dim)

        else:
            all_expert_outputs = []

            with torch.no_grad():
                for kae in self.kaes:
                    _, latent_z, _ = kae(padded_obs)

                    if latent_z.ndim == 1:
                        latent_z = (latent_z.unsqueeze(0))

                    expert_outputs = (get_experts_outputs(
                            kae,
                            latent_z,
                            self.p,
                            self.act_dim,
                        )
                    )

                    all_expert_outputs.append(
                        expert_outputs
                    )

            experts_outputs = torch.cat(
                all_expert_outputs,
                dim=1,
            )

            expert_mode_count = (self.total_modes)

        # 1st
        # # # expert_weights = (self.expert_weight_network(obs))
        # # expert_weights = torch.cat(
        # #     [net(obs) for net in self.expert_weight_networks],
        # #     dim=1,
        # # )

        # # kae_actions = torch.sum(expert_weights.view(-1,expert_mode_count,1,)* experts_outputs,dim=1,)       

        # 2nd
        # # Route, don't scale: L1-normalise the mode weights across ALL KAEs so the
        # # KMD branch magnitude is set by the modes themselves, not by w. This makes
        # # the two branches commensurate before the gate mixes them.
        # expert_weights = torch.cat(
        #     [net(obs) for net in self.expert_weight_networks],
        #     dim=1,
        # )
        # expert_weights = expert_weights / (
        #     expert_weights.abs().sum(dim=1, keepdim=True) + 1e-6
        # )

        # Two-level routing. The softmax router decides HOW MUCH budget each KAE
        # gets; each expert_weight_network, L1-normalised inside its own modes,
        # decides WHICH modes within that KAE. Total budget is still exactly 1, so
        # the KMD branch stays commensurate with res and independent of N. This
        # splits the reallocation problem from 16*N weights down to N logits.
        kae_router = torch.softmax(self.kae_router_network(obs), dim=1)
        branch_weights = []
        for branch_index, weight_net in enumerate(self.expert_weight_networks):
            branch_w = weight_net(obs)
            branch_w = branch_w / (branch_w.abs().sum(dim=1, keepdim=True) + 1e-6)
            branch_weights.append(
                branch_w * kae_router[:, branch_index : branch_index + 1]
            )
        expert_weights = torch.cat(branch_weights, dim=1)



        kae_actions = torch.sum(expert_weights.view(-1,expert_mode_count,1,)* experts_outputs,dim=1,)

        # 2. MLP pathway (residual) - uses the original, unnormalized 'obs'
        mlp_actions = self.mlp_network(obs)

       
        # 3. Gate decides blending - uses the original, unnormalized 'obs'
        # gating_input = torch.cat([obs, mlp_actions.detach(), kae_actions.detach()], dim=1)
        gating_input = torch.cat([obs], dim=1)
        gate_logit = self.gating_network(gating_input)

        # gate = torch.sigmoid(gate_logit)
        gate = self.g_min + (1.0 - self.g_min) * torch.sigmoid(gate_logit)

        # # Track the gate for logging (mean over the batch, averaged across calls per iteration)
        # gate_detached = gate.detach()
        # self.last_gate = gate_detached
        # self._gate_running_sum += gate_detached.mean().item()
        # self._gate_running_count += 1

        # 4. Blend the two pathways
        actions = (
        gate * kae_actions
        + (1.0 - gate) * mlp_actions    )

        if not torch.is_grad_enabled():
            self.last_expert_weights = expert_weights.detach()
            self.last_kae_actions = kae_actions.detach()
            self.last_mlp_actions = mlp_actions.detach()
            self.last_gate = gate.detach()

        gate_log = gate.detach()

        # Only the first ~100 iterations are needed, and .item() forces a GPU sync
        # on the rollout path, so stop logging once we have enough.
        if not torch.is_grad_enabled():
            self._fwd += 1
            _g_std = gate_log.std().item() if gate_log.numel() > 1 else 0.0
            _kae_norm = kae_actions.norm(dim=-1).mean().item()
            _res_norm = mlp_actions.norm(dim=-1).mean().item()
            self._f.write(
                f"{self._fwd},{gate_log.mean().item()},{_g_std},"
                f"{_kae_norm},{_res_norm},{_kae_norm / max(_res_norm, 1e-12)}\n"
            )

        # # --- gate / branch magnitude log -------------------------------------------
        # # Goes into the CRL session folder so concurrent trials never collide.
        # # Columns:
        # #   fwd       forward-pass counter during rollout (24 per PPO iteration)
        # #   g_mean    batch mean of the gate output  g = sigmoid(gate(obs))
        # #   g_std     batch std of g; 0 when the batch holds a single element
        # #   kae_norm  batch mean of ||w . KMD||   (the g-weighted branch, before g)
        # #   res_norm  batch mean of ||res(obs)||  (the residual branch, before 1-g)
        # #   ratio     kae_norm / res_norm
        # _log_dir = Path(os.environ.get("CRL_KAE_LOG_DIRECTORY", "logs/moe_gate"))
        # _log_dir.mkdir(parents=True, exist_ok=True)
        # _task = os.environ.get("CRL_TASK_NAME", "standalone")
        # _index = os.environ.get("CRL_TASK_INDEX", "0")
        # self._f = (_log_dir / f"{_index}_{_task}_gate.csv").open("w", buffering=1)
        # self._f.write("fwd,g_mean,g_std,kae_norm,res_norm,ratio\n")
        # self._fwd = 0


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

        # create distribution
        self.distribution = Normal(mean, std)


    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.forward(obs)

    def pop_gate_stats(self):
        """Return the mean gate value accumulated since the last call, then reset.

        The gate is a scalar in [0, 1] blending the KAE pathway (gate) with the
        MLP residual (1 - gate); returns None if no forward pass has run yet.
        """
        if self._gate_running_count == 0:
            return None
        mean_gate = self._gate_running_sum / self._gate_running_count
        self._gate_running_sum = 0.0
        self._gate_running_count = 0
        return mean_gate

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
