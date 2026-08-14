from pathlib import Path
import random
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader


class Encoder_walk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, observable_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, observable_dim, state_dim):
        super().__init__()
        self.linear = nn.Linear(observable_dim, state_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class KoopmanAutoencoder_walk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim, device):
        super().__init__()
        self.encoder = Encoder_walk(state_dim, hidden_dim, observable_dim)
        self.decoder = Decoder(observable_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.observable_dim = observable_dim
        self.K = torch.randn(observable_dim, observable_dim, device=device)

    def forward(self, x):
        z = self.encoder(x)
        z_next = torch.matmul(z, self.K.T)
        y_hat = self.decoder(z_next)
        x_hat = self.decoder(z)
        return x_hat, z, y_hat

    def compute_koopman_operator(self, latent_X, latent_Y, device):
        del device
        latent_X = latent_X.view(-1, latent_X.size(-1)).T
        latent_Y = latent_Y.view(-1, latent_Y.size(-1)).T
        self.K = latent_Y @ torch.linalg.pinv(latent_X, rcond=1e-5)


def _install_legacy_autoencoder_module():
    module = types.ModuleType("Autoencoder")
    module.Encoder_walk = Encoder_walk
    module.Decoder = Decoder
    module.KoopmanAutoencoder_walk = KoopmanAutoencoder_walk
    sys.modules["Autoencoder"] = module
    sys.modules["less_leg_walking_1.tasks.direct.less_leg_walking_1.Autoencoder"] = module


def load_kae(checkpoint_file, device):
    _install_legacy_autoencoder_module()
    kae = torch.load(checkpoint_file, map_location=device, weights_only=False)

    if not isinstance(kae, nn.Module):
        raise TypeError(f"KAE file must contain a full torch.nn.Module: {checkpoint_file}")
    if not hasattr(kae, "K") or not torch.is_tensor(kae.K):
        raise TypeError(f"Loaded KAE has no tensor Koopman operator K: {checkpoint_file}")

    kae = kae.to(device)
    kae.K = kae.K.to(device)
    kae.eval()
    return kae


def koopman_loss(x, x_hat, latent_x, y_seq_states, y_seq_latents, p, action_dim, model):
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(x_hat, x)
    state_pred_loss = 0.0
    latent_pred_loss = 0.0
    action_loss = 0.0

    Ks = [torch.linalg.matrix_power(model.K, step) for step in range(1, p + 1)]

    for k in range(p):
        pred_lat_k = latent_x @ Ks[k].T
        pred_y_k = model.decoder(pred_lat_k)
        state_pred_loss += mse_loss(pred_y_k, y_seq_states[:, k, :])
        latent_pred_loss += mse_loss(pred_lat_k, y_seq_latents[:, k, :])

        if k == p - 1:
            action_loss += mse_loss(
                pred_y_k[:, :action_dim],
                y_seq_states[:, k, :action_dim],
            )

    state_pred_loss /= p
    latent_pred_loss /= p
    return recon_loss, state_pred_loss, latent_pred_loss, action_loss


def compute_l_kae(
    kae,
    aug_input,
    aug_output,
    c1,
    c2,
    c3,
    p,
    device,
    aug_input_all,
    aug_output_all,
    inner,
    batch_size,
    action_dim,
):
    del aug_output, aug_input_all

    batch_count = aug_input.size(0)
    start = inner * batch_size
    end = start + batch_count

    x = aug_input.to(device).squeeze(1)
    x_hat, latent_x, _ = kae(x)

    y_seq_states = []
    y_seq_latents = []
    for k in range(p):
        if end + k > len(aug_output_all):
            break

        yk = aug_output_all[start + k : end + k].to(device).squeeze(1)
        _, latent_yk, _ = kae(yk)
        y_seq_states.append(yk)
        y_seq_latents.append(latent_yk)

    if not y_seq_states:
        raise RuntimeError("No future states available for multi-step supervision.")

    if len(y_seq_states) < p:
        last_y = y_seq_states[-1]
        last_latent_y = y_seq_latents[-1]
        for _ in range(p - len(y_seq_states)):
            y_seq_states.append(last_y)
            y_seq_latents.append(last_latent_y)

    y_seq_states = torch.stack(y_seq_states, dim=1)
    y_seq_latents = torch.stack(y_seq_latents, dim=1)

    recon_loss, state_pred_loss, latent_pred_loss, action_loss = koopman_loss(
        x,
        x_hat,
        latent_x,
        y_seq_states,
        y_seq_latents,
        p,
        action_dim,
        kae,
    )
    loss_kae = c1 * recon_loss + c2 * state_pred_loss + c3 * latent_pred_loss
    return loss_kae, action_loss, recon_loss, state_pred_loss, latent_pred_loss


def _set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _fit_observation_distribution(observations, device):
    if not torch.is_tensor(observations):
        raise TypeError(f"Observations must be a torch.Tensor, got {type(observations).__name__}.")
    if observations.ndim != 2:
        raise RuntimeError(f"Observations must have shape [N, obs_dim], got {tuple(observations.shape)}.")
    if observations.shape[0] < 2:
        raise RuntimeError("At least two observations are required to estimate covariance.")
    if not torch.isfinite(observations).all():
        raise FloatingPointError("Observation data contains NaN or Inf.")

    observations64 = observations.to(dtype=torch.float64, device="cpu")
    mu64 = observations64.mean(dim=0)
    centered = observations64 - mu64
    sigma64 = centered.T @ centered / (observations64.shape[0] - 1)
    sigma64 = 0.5 * (sigma64 + sigma64.T)
    sigma64 += 1e-5 * torch.eye(sigma64.shape[0], dtype=sigma64.dtype)

    mu = mu64.to(dtype=torch.float32, device=device)
    sigma = sigma64.to(dtype=torch.float32, device=device)
    return MultivariateNormal(mu, covariance_matrix=sigma)


def _policy_mean(policy, observations):
    if not hasattr(policy, "actor_obs_normalizer"):
        raise AttributeError(
            "The integrated CRL policy must expose actor_obs_normalizer so KAE data "
            "uses the same preprocessing as policy rollout."
        )
    if not callable(getattr(policy, "forward", None)):
        raise TypeError("The integrated CRL policy must provide a callable forward method.")

    normalized_observations = policy.actor_obs_normalizer(observations)
    if not torch.is_tensor(normalized_observations):
        raise TypeError(
            "actor_obs_normalizer must return a torch.Tensor, got "
            f"{type(normalized_observations).__name__}."
        )
    if normalized_observations.shape != observations.shape:
        raise RuntimeError(
            "actor_obs_normalizer changed observation shape from "
            f"{tuple(observations.shape)} to {tuple(normalized_observations.shape)}."
        )

    policy_output = policy.forward(normalized_observations)
    if not torch.is_tensor(policy_output):
        raise TypeError(
            f"Policy forward must return torch.Tensor, got {type(policy_output).__name__}."
        )
    if policy_output.ndim != 2:
        raise RuntimeError(
            f"Policy output must have shape [N, action_dim], got {tuple(policy_output.shape)}."
        )
    if policy_output.shape[0] != observations.shape[0]:
        raise RuntimeError(
            "Policy output batch size does not match policy input batch size: "
            f"{policy_output.shape[0]} != {observations.shape[0]}."
        )
    if not torch.isfinite(policy_output).all():
        raise FloatingPointError("Policy output contains NaN or Inf.")
    return policy_output


def _generate_training_data(
    distribution,
    policy,
    sample_count,
    obs_dim,
    padded_dimension,
    device,
):
    training_data = []
    action_dim = None

    with torch.no_grad():
        for _ in range(sample_count):
            random_input = distribution.sample((1,)).to(device)
            policy_output = _policy_mean(policy, random_input)

            if action_dim is None:
                action_dim = policy_output.shape[-1]
                if padded_dimension < obs_dim or padded_dimension < action_dim:
                    raise ValueError(
                        "padded_dimension must be at least both observation and action dimensions: "
                        f"padded_dimension={padded_dimension}, obs_dim={obs_dim}, action_dim={action_dim}."
                    )
            elif policy_output.shape[-1] != action_dim:
                raise RuntimeError(
                    "Policy output dimension changed during data generation: "
                    f"expected {action_dim}, got {policy_output.shape[-1]}."
                )

            pad_input = torch.ones(1, padded_dimension - obs_dim, device=device)
            pad_output = torch.ones(1, padded_dimension - action_dim, device=device)
            aug_input = torch.cat([random_input, pad_input], dim=1)
            aug_output = torch.cat([policy_output, pad_output], dim=1)
            training_data.append((aug_input.detach().cpu(), aug_output.detach().cpu()))

    if action_dim is None:
        raise RuntimeError("No KAE training samples were generated.")
    return training_data, action_dim


def train_and_save_kae(
    task_name,
    policy,
    observation_file,
    kae_directory,
    device="cuda",
    observable_dim=16,
    padded_dimension=256,
    hidden_dim=256,
    sample_count=15000,
    batch_size=2048,
    num_epochs=20000,
    learning_rate=1e-5,
    kae_coefficient=0.1,
    action_coefficient=0.9,
    c1=1.0,
    c2=1.0,
    c3=1.0,
    p=1,
    seed=10,
):
    device = torch.device(device)
    observation_file = Path(observation_file)
    kae_directory = Path(kae_directory)

    if not observation_file.is_file():
        raise FileNotFoundError(str(observation_file))
    if p < 1:
        raise ValueError(f"p must be at least 1, got {p}.")
    if batch_size % p != 0:
        raise ValueError(f"batch_size must be divisible by p: batch_size={batch_size}, p={p}.")

    kae_directory.mkdir(parents=True, exist_ok=True)
    _set_deterministic_seed(seed)

    observations = torch.load(observation_file, map_location="cpu", weights_only=False)
    if not torch.is_tensor(observations):
        raise TypeError(
            f"Observation file must contain torch.Tensor, got {type(observations).__name__}."
        )
    observations = observations.float().reshape(-1, observations.shape[-1])
    obs_dim = observations.shape[-1]

    distribution = _fit_observation_distribution(observations, device)
    policy = policy.to(device)
    policy.eval()

    training_data, action_dim = _generate_training_data(
        distribution,
        policy,
        sample_count,
        obs_dim,
        padded_dimension,
        device,
    )

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    aug_input_all = torch.cat(
        [training_data[index][0].unsqueeze(0) for index in range(len(training_data))]
    ).to(device)
    aug_output_all = torch.cat(
        [training_data[index][1].unsqueeze(0) for index in range(len(training_data))]
    ).to(device)

    kae = KoopmanAutoencoder_walk(
        state_dim=padded_dimension,
        hidden_dim=hidden_dim,
        observable_dim=observable_dim,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(kae.parameters(), lr=learning_rate)
    kae.train()

    import time
    start = time.perf_counter()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_kae_loss = 0.0
        running_action_loss = 0.0
        running_recon_loss = 0.0
        running_state_loss = 0.0
        running_latent_loss = 0.0

        for inner, (aug_input, aug_output) in enumerate(train_loader):
            loss_kae, loss_action, recon_loss, state_loss, latent_loss = compute_l_kae(
                kae,
                aug_input,
                aug_output,
                c1,
                c2,
                c3,
                p,
                device,
                aug_input_all,
                aug_output_all,
                inner,
                batch_size,
                action_dim,
            )
            loss = kae_coefficient * loss_kae + action_coefficient * loss_action

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite KAE loss at epoch {epoch + 1}, batch {inner + 1}: {loss.item()}"
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kae.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            running_kae_loss += loss_kae.detach().cpu().item()
            running_action_loss += loss_action.detach().cpu().item()
            running_recon_loss += recon_loss.detach().cpu().item()
            running_state_loss += state_loss.detach().cpu().item()
            running_latent_loss += latent_loss.detach().cpu().item()

            # This intentionally remains inside autograd, matching the original notebook.
            _, latent_input_all, _ = kae(aug_input_all)
            _, latent_output_all, _ = kae(aug_output_all)
            kae.compute_koopman_operator(latent_input_all, latent_output_all, device)

            if not torch.isfinite(kae.K).all():
                raise FloatingPointError(
                    f"Koopman operator contains NaN or Inf at epoch {epoch + 1}, batch {inner + 1}."
                )

        num_batches = len(train_loader)
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(
                f"[KAE] {task_name} epoch {epoch + 1}/{num_epochs} "
                f"loss={running_loss / num_batches:.6f} "
                f"loss_kae={running_kae_loss / num_batches:.6f} "
                f"action={running_action_loss / num_batches:.6f} "
                f"recon={running_recon_loss / num_batches:.6f} "
                f"state={running_state_loss / num_batches:.6f} "
                f"latent={running_latent_loss / num_batches:.6f}"
            )

    kae.eval()
    kae = kae.cpu()
    kae.K = kae.K.detach().cpu()
    _install_legacy_autoencoder_module()

    output_file = kae_directory / f"{task_name}_KAE.pth"
    torch.save(kae, output_file)
    print(f"[KAE] Saved: {output_file}")

    elapsed = time.perf_counter() - start
    print(f"Elapsed time for training KAE: {elapsed:.4f} seconds")

    return output_file