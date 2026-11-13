from pathlib import Path
import csv
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

# ---- KAE-side sizes and constants (the only place they are set) ----------------------------
BLOCK_SIZE = 2                   # real block size of K; must divide observable_dim
OBSERVER_HIDDEN_DIMS = [64, 32, 16]  # ModeObserver MLP hidden widths (ELU)
OBSERVER_LOSS_WEIGHT = 0.1       # weight of the observer error in the KAE loss
# -------------------------------------------------------------------------------------------


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


class KoopmanAutoencoder_walk_var(nn.Module):
    """Encoder_walk + Decoder with a block-diagonal, trainable K.

    Row convention as in KAE_approx.py: z_next = z @ K.T. blocks[k] is the k-th block of K itself,
    K = blockdiag(blocks[0], ..., blocks[n_units-1]), each block_size x block_size, initialised to
    the identity. Unit k reads latent coordinates [k*block_size, (k+1)*block_size).
    """

    def __init__(self, state_dim, hidden_dim, observable_dim, device, block_size=BLOCK_SIZE):
        super().__init__()
        if block_size < 1 or observable_dim % block_size:
            raise ValueError(
                f"observable_dim ({observable_dim}) must be divisible by block_size ({block_size})."
            )
        del device  # parameters follow .to(device) like every other module
        self.encoder = Encoder_walk(state_dim, hidden_dim, observable_dim)
        self.decoder = Decoder(observable_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.observable_dim = observable_dim
        self.block_size = block_size
        self.n_units = observable_dim // block_size
        self.blocks = nn.Parameter(torch.eye(block_size).repeat(self.n_units, 1, 1))

    @property
    def K(self):
        """Dense [d, d] view of the blocks; differentiable, built on demand."""
        return torch.block_diag(*self.blocks)

    def evolve(self, z, steps):
        """z @ K.T applied `steps` times as batched block products."""
        units = z.reshape(z.shape[0], self.n_units, self.block_size)
        for _ in range(int(steps)):
            units = torch.einsum("nki,kji->nkj", units, self.blocks)
        return units.reshape(z.shape[0], -1)

    def unit_outputs(self, z, act_dim, p):
        """Mode outputs [N, n_units, act_dim]: unit k gives B[:act_dim, block_k] (R_k^p z_k).
        Summing over units reproduces decoder(z @ (K^p).T)[:, :act_dim] exactly.
        """
        evolved = self.evolve(z, p).reshape(z.shape[0], self.n_units, self.block_size)
        weight = self.decoder.linear.weight[:act_dim]  # [act_dim, d]
        weight = weight.reshape(act_dim, self.n_units, self.block_size)
        return torch.einsum("nki,aki->nka", evolved, weight)

    def forward(self, x):
        z = self.encoder(x)
        z_next = self.evolve(z, 1)
        y_hat = self.decoder(z_next)
        x_hat = self.decoder(z)
        return x_hat, z, y_hat


class _GradReverse(torch.autograd.Function):
    """Identity forward, negated gradient backward (gradient reversal layer)."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


def standardised(units):
    """Per unit and per action dim, zero mean / unit variance over the batch."""
    mean = units.mean(dim=0, keepdim=True)
    std = units.std(dim=0, keepdim=True)
    return (units - mean) / (std + 1e-6)


class ModeObserver(nn.Module):
    """One MLP per unit: predicts the standardised output of unit k from the standardised outputs
    of the other units. Its own loss is the prediction MSE (it minimises); the KAE sees the same
    MSE through _GradReverse (it maximises). n_units < 2 gives no observer (loss 0).
    """

    def __init__(self, n_units, act_dim, hidden_dims):
        super().__init__()
        self.n_units = n_units
        self.heads = nn.ModuleList()
        if n_units < 2:
            return
        for _ in range(n_units):
            layers = []
            width = (n_units - 1) * act_dim
            for hidden in hidden_dims:
                layers.append(nn.Linear(width, hidden))
                layers.append(nn.ELU())
                width = hidden
            layers.append(nn.Linear(width, act_dim))
            self.heads.append(nn.Sequential(*layers))

    def forward(self, units):
        """units: [N, n_units, act_dim] mode outputs (raw). Returns mean prediction MSE."""
        if self.n_units < 2:
            return units.sum() * 0.0
        units = standardised(_GradReverse.apply(units))
        errors = []
        for k, head in enumerate(self.heads):
            others = torch.cat([units[:, :k], units[:, k + 1 :]], dim=1).reshape(units.shape[0], -1)
            errors.append(nn.functional.mse_loss(head(others), units[:, k]))
        return torch.stack(errors).mean()


def _install_legacy_autoencoder_module():
    """Dense KAEs saved by the A pipeline (e.g. the original task's) pickle their classes under
    'crl_kae_approx' or 'Autoencoder'. Resolve them from the sibling KAE_approx.py instead of
    duplicating the dense class here."""
    if "crl_kae_approx" not in sys.modules:
        legacy_file = Path(__file__).with_name("KAE_approx.py")
        spec = importlib.util.spec_from_file_location("crl_kae_approx", legacy_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["crl_kae_approx"] = module
        spec.loader.exec_module(module)
    sys.modules["crl_kae_approx"]._install_legacy_autoencoder_module()


def load_kae(checkpoint_file, device):
    _install_legacy_autoencoder_module()
    kae = torch.load(checkpoint_file, map_location=device, weights_only=False)

    if not isinstance(kae, nn.Module):
        raise TypeError(f"KAE file must contain a full torch.nn.Module: {checkpoint_file}")
    if not hasattr(kae, "K") or not torch.is_tensor(kae.K):
        raise TypeError(f"Loaded KAE has no tensor Koopman operator K: {checkpoint_file}")

    kae = kae.to(device)
    if not hasattr(kae, "blocks"):
        kae.K = kae.K.to(device)  # dense legacy KAE: K is a plain tensor attribute
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
    observer,
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
    observer_loss = observer(kae.unit_outputs(latent_x, action_dim, p))
    return loss_kae, action_loss, recon_loss, state_pred_loss, latent_pred_loss, observer_loss


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
    num_epochs=20000,#20000,
    learning_rate=1e-4,
    kae_coefficient=0.1,
    action_coefficient=0.9,
    c1=1.0,
    c2=1.0,
    c3=1.0,
    p=1,
    seed=777,
    log_directory=None,
    block_size=BLOCK_SIZE,
    observer_hidden_dims=OBSERVER_HIDDEN_DIMS,
    observer_weight=OBSERVER_LOSS_WEIGHT,
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
    if block_size < 1 or observable_dim % block_size:
        raise ValueError(
            f"observable_dim ({observable_dim}) must be divisible by block_size ({block_size})."
        )
    if observer_weight < 0:
        raise ValueError(f"observer_weight must be nonnegative, got {observer_weight}.")

    # kae_directory.mkdir(parents=True, exist_ok=True)
    # _set_deterministic_seed(seed)

    kae_directory.mkdir(parents=True, exist_ok=True)
    if log_directory is None:
        log_directory = os.environ.get("CRL_KAE_LOG_DIRECTORY") or (kae_directory / "kae_logs")
    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    history_file = log_directory / f"{task_name}_kae_history.csv"
    summary_file = log_directory / f"{task_name}_kae_summary.json"
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

    kae = KoopmanAutoencoder_walk_var(
        state_dim=padded_dimension,
        hidden_dim=hidden_dim,
        observable_dim=observable_dim,
        device=device,
        block_size=block_size,
    ).to(device)
    observer = ModeObserver(kae.n_units, action_dim, observer_hidden_dims).to(device)
    optimizer = torch.optim.Adam(
        list(kae.parameters()) + list(observer.parameters()), lr=learning_rate
    )

    # kae.train()
    # import time
    # start = time.perf_counter()
    # for epoch in range(num_epochs):

    kae.train()
    observer.train()
    start = time.perf_counter()
    history_handle = history_file.open("w", newline="", encoding="utf-8")
    history_writer = csv.writer(history_handle)
    history_writer.writerow(
        ["epoch", "loss", "loss_kae", "action", "recon", "state", "latent", "observer",
        "elapsed_s"]
    )
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_kae_loss = 0.0
        running_action_loss = 0.0
        running_recon_loss = 0.0
        running_state_loss = 0.0
        running_latent_loss = 0.0
        running_observer_loss = 0.0

        for inner, (aug_input, aug_output) in enumerate(train_loader):
            loss_kae, loss_action, recon_loss, state_loss, latent_loss, observer_loss = compute_l_kae(
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
                observer,
            )
            # observer_loss enters once: the observer descends it, the KAE ascends it (_GradReverse).
            loss = (
                kae_coefficient * loss_kae
                + action_coefficient * loss_action
                + observer_weight * observer_loss
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite KAE loss at epoch {epoch + 1}, batch {inner + 1}: {loss.item()}"
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(kae.parameters()) + list(observer.parameters()), max_norm=1.0
            )
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            running_kae_loss += loss_kae.detach().cpu().item()
            running_action_loss += loss_action.detach().cpu().item()
            running_recon_loss += recon_loss.detach().cpu().item()
            running_state_loss += state_loss.detach().cpu().item()
            running_latent_loss += latent_loss.detach().cpu().item()
            running_observer_loss += observer_loss.detach().cpu().item()

            # No EDMD refit: the block-diagonal K is a parameter updated by optimizer.step().
            if not torch.isfinite(kae.blocks).all():
                raise FloatingPointError(
                    f"Koopman blocks contain NaN or Inf at epoch {epoch + 1}, batch {inner + 1}."
                )

        # num_batches = len(train_loader)
        # if epoch == 0 or (epoch + 1) % 100 == 0:
        #     print(
        #         f"[KAE] {task_name} epoch {epoch + 1}/{num_epochs} "
        #         f"loss={running_loss / num_batches:.6f} "
        #         f"loss_kae={running_kae_loss / num_batches:.6f} "
        #         f"action={running_action_loss / num_batches:.6f} "
        #         f"recon={running_recon_loss / num_batches:.6f} "
        #         f"state={running_state_loss / num_batches:.6f} "
        #         f"latent={running_latent_loss / num_batches:.6f}"
        #     )

        num_batches = len(train_loader)
        with torch.no_grad():
            spectral_radius = torch.linalg.eigvals(kae.K).abs().max().item()
        history_writer.writerow([
            epoch + 1,
            running_loss / num_batches,
            running_kae_loss / num_batches,
            running_action_loss / num_batches,
            running_recon_loss / num_batches,
            running_state_loss / num_batches,
            running_latent_loss / num_batches,
            running_observer_loss / num_batches,
            time.perf_counter() - start,
        ])
        if epoch == 0 or (epoch + 1) % 100 == 0:
            history_handle.flush()
            print(
                f"[KAE] {task_name} epoch {epoch + 1}/{num_epochs} "
                f"loss={running_loss / num_batches:.6f} "
                f"loss_kae={running_kae_loss / num_batches:.6f} "
                f"action={running_action_loss / num_batches:.6f} "
                f"recon={running_recon_loss / num_batches:.6f} "
                f"state={running_state_loss / num_batches:.6f} "
                f"latent={running_latent_loss / num_batches:.6f} "
                f"observer={running_observer_loss / num_batches:.6f}"
            )

    history_handle.close()
    kae.eval()
    observer.eval()

    # Evaluate reconstruction on the real rollout observations, not on the sampled
    # training data, so the number reflects states the policy actually visits.
    with torch.no_grad():
        eval_count = min(4096, observations.shape[0])
        real = observations[:eval_count].to(device)
        real_pad = torch.ones(eval_count, padded_dimension - obs_dim, device=device)
        real_aug = torch.cat([real, real_pad], dim=1)
        real_hat, real_latent, _ = kae(real_aug)
        eval_observer = observer(kae.unit_outputs(real_latent, action_dim, p)).item()
        recon_mse = nn.MSELoss()(real_hat, real_aug).item()
        recon_rel = (torch.norm(real_hat - real_aug) / torch.norm(real_aug)).item()
        eigenvalues = torch.linalg.eigvals(kae.K)

    elapsed = time.perf_counter() - start
    summary = {
        "task_name": task_name,
        "session_id": os.environ.get("CRL_SESSION_ID"),
        "task_index": os.environ.get("CRL_TASK_INDEX"),
        "observation_file": str(observation_file),
        "num_epochs": num_epochs,
        "sample_count": sample_count,
        "batch_size": batch_size,
        "observable_dim": observable_dim,
        "kae_type": "block_diagonal",
        "block_size": block_size,
        "n_units": kae.n_units,
        "observer_hidden_dims": list(observer_hidden_dims),
        "observer_weight": observer_weight,
        "padded_dimension": padded_dimension,
        "hidden_dim": hidden_dim,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "learning_rate": learning_rate,
        "seed": seed,
        "final_loss": running_loss / num_batches,
        "final_loss_kae": running_kae_loss / num_batches,
        "final_action_loss": running_action_loss / num_batches,
        "final_recon_loss": running_recon_loss / num_batches,
        "final_state_loss": running_state_loss / num_batches,
        "final_latent_loss": running_latent_loss / num_batches,
        "final_observer_loss": running_observer_loss / num_batches,
        "eval_observer_loss_real_obs": eval_observer,
        "eval_recon_mse_real_obs": recon_mse,
        "eval_recon_relative_error": recon_rel,
        "elapsed_seconds": elapsed,
    }

    kae = kae.cpu()
    output_file = kae_directory / f"{task_name}_KAE.pth"
    torch.save(kae, output_file)
    summary["kae_file"] = str(output_file)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[KAE] Saved   : {output_file}")
    print(f"[KAE] History : {history_file}")
    print(f"[KAE] Summary : {summary_file}")
    print(f"[KAE] eval  recon_rel={recon_rel:.4f}  rho(K)={spectral_radius:.4f}  observer={eval_observer:.4f}")
    print(f"Elapsed time for training KAE: {elapsed:.4f} seconds")
    return output_file