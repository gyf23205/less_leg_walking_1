import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.fx import symbolic_trace

def select_action(obs, policy_net, device):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    probs = F.softmax(policy_net(obs_tensor), dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob.item(), dist

def extend_experts_outputs(experts_outputs, act_dim):
    extends = torch.diag(torch.ones(act_dim, dtype=experts_outputs.dtype, device=experts_outputs.device)).tile(experts_outputs.shape[0], 1, 1)
    extended_outputs = torch.cat([experts_outputs, extends], dim=1)
    return extended_outputs

def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    advantages = []
    gae = 0
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    return advantages

def process_trajectory(value_net, obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs, device):
    next_value = value_net(torch.tensor(obs, dtype=torch.float32).to(device)).item()
    advantages = compute_gae(rew_buf, val_buf, done_buf, next_value)
    returns = [adv + v for adv, v in zip(advantages, val_buf)]
    obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(device)
    act_tensor = torch.tensor(act_buf, dtype=torch.int64).to(device)
    logp_tensor = torch.tensor(logp_buf, dtype=torch.float32).to(device)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
    ret_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
    dataset = TensorDataset(obs_tensor, act_tensor, logp_tensor, adv_tensor, ret_tensor)
    return dataset

def get_experts_outputs(kae, z, p, act_dim, conjugate=False):
    ko = kae.K
    eigvals, eigvec_left = torch.linalg.eig(ko.T)
    # print(f"eigvals conj shape: {eigvals.conj().shape}", f"eigvec_left conj shape: {eigvec_left.conj().shape}")
    # assert False
    eigvals = eigvals.conj().unsqueeze(0)
    eigvec_left = eigvec_left.conj().T.unsqueeze(0)
    # print(f"eigvals shape: {eigvals.shape}", f"eigvec_left shape: {eigvec_left.shape}")
    # assert False
    eigvec_left_inv = torch.linalg.inv(eigvec_left)

    B = kae.decoder.linear.weight.detach().clone()
    B = B.to(torch.complex64)
    v = (B @ eigvec_left_inv) # kae dim x encoder dim

    phi = torch.sum(eigvec_left * z.to(torch.complex64).unsqueeze(1), dim=-1)
    # phi = eigvec_left @ z.to(torch.complex64)
    if conjugate:
        expert_outputs = (((eigvals**p)*phi).unsqueeze(1)*v).conj()
    else:
        expert_outputs = ((eigvals**p)*phi).unsqueeze(1)*v

    output = expert_outputs[:, :act_dim*2, :].transpose(1, 2).real
    return output.to(dtype=torch.float32, copy=False) # (Batch, observable_dim(+act_dim if using the extended experts), act_dim*2), the last dimension is mean and std

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def extract_model_structure_and_shapes(model: nn.Module):
    traced = symbolic_trace(model)
    modules = dict(model.named_modules())
    model_structure = []
    param_shapes = []

    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        layer = modules[node.target]
        layer_type = type(layer)

        if isinstance(layer, nn.Conv2d):
            model_structure.append({
                "type": "conv2d",
                "params": [layer.weight.shape, layer.bias.shape],
                "stride": layer.stride,
                "padding": layer.padding,
            })
            param_shapes.extend([layer.weight.shape, layer.bias.shape])
        elif isinstance(layer, nn.Linear):
            model_structure.append({
                "type": "linear",
                "params": [layer.weight.shape, layer.bias.shape],
            })
            param_shapes.extend([layer.weight.shape, layer.bias.shape])
        elif isinstance(layer, nn.ReLU):
            model_structure.append({"type": "relu"})
        elif isinstance(layer, nn.Tanh):
            model_structure.append({"type": "tanh"})
        elif isinstance(layer, nn.MaxPool2d):
            model_structure.append({
                "type": "maxpool2d",
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
            })
        elif isinstance(layer, nn.Flatten):
            model_structure.append({"type": "flatten"})
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    return model_structure, param_shapes

def functional_forward(x: torch.Tensor, p_vec: torch.Tensor, model_structure, param_shapes):
    device = x.device
    idx = 0
    for layer in model_structure:
        layer_type = layer["type"]

        if "params" in layer:
            layer_params = []
            for shape in layer["params"]:
                numel = torch.prod(torch.tensor(shape, device=device)).item()
                param = p_vec[idx:idx + numel].view(shape).to(device)
                layer_params.append(param)
                idx += numel
        else:
            layer_params = []

        if layer_type == "conv2d":
            weight, bias = layer_params
            x = F.conv2d(x, weight, bias, stride=layer["stride"], padding=layer["padding"])
        elif layer_type == "linear":
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            weight, bias = layer_params
            x = F.linear(x, weight, bias)
        elif layer_type == "relu":
            x = F.relu(x)
        elif layer_type == "tanh":
            x = torch.tanh(x)
        elif layer_type == "maxpool2d":
            x = F.max_pool2d(x, kernel_size=layer["kernel_size"], stride=layer["stride"])
        elif layer_type == "flatten":
            x = x.view(x.size(0), -1)
        else:
            raise NotImplementedError(f"Unsupported layer: {layer_type}")

    return x