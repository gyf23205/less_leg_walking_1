import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from .Autoencoder import KoopmanAutoencoder_walk


def fit_observation_gaussian(observation_file, covariance_jitter=1e-5):
    observations = torch.load(observation_file, map_location="cpu")
    observations = observations.float()
    observations = observations.reshape(-1, observations.shape[-1])

    mean = observations.mean(dim=0)
    centered = observations - mean
    covariance = centered.T @ centered
    covariance = covariance / max(observations.shape[0] - 1, 1)

    covariance = 0.5 * (covariance + covariance.T)
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    minimum_eigenvalue = torch.linalg.eigvalsh(covariance).min().item()

    if minimum_eigenvalue < covariance_jitter:
        covariance = covariance + (
            covariance_jitter - minimum_eigenvalue
        ) * identity

    return mean, covariance


def run_policy(policy, observations):
    if hasattr(policy, "act_inference"):
        actions = policy.act_inference(observations)
    else:
        actions = policy(observations)

    if isinstance(actions, tuple):
        actions = actions[0]

    return actions


def generate_training_data(
    policy,
    mean,
    covariance,
    sample_count,
    padded_dimension,
    device,
    generation_batch_size=4096,
):
    distribution = MultivariateNormal(
        mean.to(device),
        covariance_matrix=covariance.to(device),
    )

    padded_inputs = []
    padded_outputs = []

    policy.eval()

    if mean.shape[0] > padded_dimension:
        raise ValueError("Observation dimension exceeds padded dimension.")

    with torch.no_grad():
        generated = 0

        while generated < sample_count:
            current_size = min(
                generation_batch_size,
                sample_count - generated,
            )

            observations = distribution.sample((current_size,))
            actions = run_policy(policy, observations)

            if actions.shape[-1] > padded_dimension:
                raise ValueError("Action dimension exceeds padded dimension.")

            padded_observations = F.pad(
                observations,
                (0, padded_dimension - observations.shape[-1]),
                value=1.0,
            )
            padded_actions = F.pad(
                actions,
                (0, padded_dimension - actions.shape[-1]),
                value=1.0,
            )

            padded_inputs.append(padded_observations.cpu())
            padded_outputs.append(padded_actions.cpu())
            generated += current_size

    return torch.cat(padded_inputs), torch.cat(padded_outputs)


def update_koopman_operator(kae, data_loader, device):
    latent_inputs = []
    latent_outputs = []

    kae.eval()

    with torch.no_grad():
        for inputs, outputs in data_loader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            latent_inputs.append(kae.encoder(inputs).cpu())
            latent_outputs.append(kae.encoder(outputs).cpu())

    latent_inputs = torch.cat(latent_inputs).to(device)
    latent_outputs = torch.cat(latent_outputs).to(device)
    kae.compute_koopman_operator(latent_inputs, latent_outputs, device)
    kae.train()


def train_and_save_kae(
    task_name,
    policy,
    observation_file,
    kae_directory,
    observable_dim=16,
    padded_dimension=256,
    hidden_dim=256,
    sample_count=50000,
    batch_size=2048,
    num_epochs=3000,
    learning_rate=1e-3,
    second_learning_rate=1e-4,
    third_learning_rate=1e-5,
    second_schedule_epoch=1000,
    third_schedule_epoch=2000,
    kae_coefficient=0.1,
    action_coefficient=0.9,
    gradient_clip=1.0,
    device="cuda",
):
    device = torch.device(device)
    kae_directory = Path(kae_directory)
    kae_directory.mkdir(parents=True, exist_ok=True)

    observation_file = Path(observation_file)
    if not observation_file.is_file():
        raise FileNotFoundError(str(observation_file))

    policy = policy.to(device)
    policy.eval()

    mean, covariance = fit_observation_gaussian(observation_file)

    inputs, outputs = generate_training_data(
        policy,
        mean,
        covariance,
        sample_count,
        padded_dimension,
        device,
    )

    observation_dim = mean.shape[0]
    action_dim = run_policy(
        policy,
        mean.unsqueeze(0).to(device),
    ).shape[-1]

    dataset = TensorDataset(inputs, outputs)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    kae = KoopmanAutoencoder_walk(
        state_dim=padded_dimension,
        hidden_dim=hidden_dim,
        observable_dim=observable_dim,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(
        kae.parameters(),
        lr=learning_rate,
    )

    best_loss = float("inf")
    best_kae = None

    for epoch in range(num_epochs):
        if epoch > third_schedule_epoch:
            current_learning_rate = third_learning_rate
        elif epoch > second_schedule_epoch:
            current_learning_rate = second_learning_rate
        else:
            current_learning_rate = learning_rate

        for group in optimizer.param_groups:
            group["lr"] = current_learning_rate

        update_koopman_operator(kae, data_loader, device)

        running_loss = 0.0

        for padded_observations, padded_actions in data_loader:
            padded_observations = padded_observations.to(device)
            padded_actions = padded_actions.to(device)

            reconstructed, latent_input, predicted = kae(
                padded_observations
            )
            latent_target = kae.encoder(padded_actions)
            predicted_latent = latent_input @ kae.K.T

            reconstruction_loss = F.mse_loss(
                reconstructed,
                padded_observations,
            )
            prediction_loss = F.mse_loss(
                predicted,
                padded_actions,
            )
            latent_loss = F.mse_loss(
                predicted_latent,
                latent_target,
            )
            action_loss = F.mse_loss(
                predicted[:, :action_dim],
                padded_actions[:, :action_dim],
            )

            kae_loss = (
                reconstruction_loss
                + prediction_loss
                + latent_loss
            )
            loss = (
                kae_coefficient * kae_loss
                + action_coefficient * action_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                kae.parameters(),
                gradient_clip,
            )
            optimizer.step()

            running_loss += loss.detach().item()

        average_loss = running_loss / len(data_loader)
        update_koopman_operator(kae, data_loader, device)

        if average_loss < best_loss:
            best_loss = average_loss
            best_kae = copy.deepcopy(kae).cpu()
            best_kae.K = best_kae.K.detach().cpu()

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(
                "[KAE]",
                task_name,
                "epoch",
                epoch + 1,
                "loss",
                average_loss,
            )

    if best_kae is None:
        raise RuntimeError("KAE training did not produce a model.")

    output_file = kae_directory / (task_name + "_KAE.pth")
    torch.save(best_kae, output_file)

    print("[KAE] Saved:", output_file)
    print("[KAE] Observation dimension:", observation_dim)
    print("[KAE] Action dimension:", action_dim)

    return output_file