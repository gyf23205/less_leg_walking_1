from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset


class EncoderWalk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim):
        super().__init__()
        self.network = nn.Sequential(
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
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, observable_dim, state_dim):
        super().__init__()
        self.linear = nn.Linear(observable_dim, state_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class KoopmanAutoencoderWalk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim, device):
        super().__init__()
        self.encoder = EncoderWalk(state_dim, hidden_dim, observable_dim)
        self.decoder = Decoder(observable_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.observable_dim = observable_dim
        self.register_buffer(
            "K",
            torch.randn(observable_dim, observable_dim, device=device),
        )

    def forward(self, x):
        latent = self.encoder(x)
        next_latent = latent @ self.K.T
        predicted_output = self.decoder(next_latent)
        reconstructed_input = self.decoder(latent)
        return reconstructed_input, latent, predicted_output

    def update_koopman_operator(self, latent_input, latent_output):
        latent_input = latent_input.reshape(-1, latent_input.shape[-1]).T
        latent_output = latent_output.reshape(-1, latent_output.shape[-1]).T
        new_operator = latent_output @ torch.linalg.pinv(
            latent_input,
            rcond=1e-5,
        )
        self.K.copy_(new_operator)


def train_and_save_kae(
    task_name,
    policy,
    observation_file,
    kae_directory,
    device="cuda",
    observable_dim=16,
    padded_dimension=256,
    hidden_dim=256,
    sample_count=50000,
    batch_size=2048,
    num_epochs=3000,
    learning_rate=1e-3,
):
    device = torch.device(device)
    observation_file = Path(observation_file)
    kae_directory = Path(kae_directory)

    if not observation_file.is_file():
        raise FileNotFoundError(str(observation_file))

    kae_directory.mkdir(parents=True, exist_ok=True)

    observations = torch.load(
        observation_file,
        map_location="cpu",
        weights_only=False,
    ).float()
    observations = observations.reshape(-1, observations.shape[-1])

    if observations.shape[0] < 2:
        raise ValueError("At least two observations are required.")

    observation_dim = observations.shape[-1]
    if observation_dim >= padded_dimension:
        raise ValueError(
            f"observation_dim={observation_dim} must be smaller than "
            f"padded_dimension={padded_dimension}."
        )

    mean = observations.mean(dim=0)
    centered = observations - mean
    covariance = centered.T @ centered
    covariance = covariance / (observations.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance + 1e-5 * torch.eye(
        observation_dim,
        dtype=covariance.dtype,
    )

    distribution = MultivariateNormal(
        mean.to(device),
        covariance_matrix=covariance.to(device),
    )

    policy = policy.to(device)
    policy.eval()

    with torch.no_grad():
        sampled_observations = distribution.sample((sample_count,))
        target_actions = policy(sampled_observations)
        if isinstance(target_actions, tuple):
            target_actions = target_actions[0]

    if target_actions.ndim != 2:
        raise ValueError(
            f"Policy output must have shape [N, action_dim], "
            f"but received {tuple(target_actions.shape)}."
        )

    action_dim = target_actions.shape[-1]
    if action_dim >= padded_dimension:
        raise ValueError(
            f"action_dim={action_dim} must be smaller than "
            f"padded_dimension={padded_dimension}."
        )

    padded_observations = F.pad(
        sampled_observations,
        (0, padded_dimension - observation_dim),
        value=1.0,
    )
    padded_actions = F.pad(
        target_actions,
        (0, padded_dimension - action_dim),
        value=1.0,
    )

    training_loader = DataLoader(
        TensorDataset(padded_observations, padded_actions),
        batch_size=batch_size,
        shuffle=False,
    )

    kae = KoopmanAutoencoderWalk(
        state_dim=padded_dimension,
        hidden_dim=hidden_dim,
        observable_dim=observable_dim,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(kae.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        if epoch > 2000:
            current_learning_rate = 1e-5
        elif epoch > 1000:
            current_learning_rate = 1e-4
        else:
            current_learning_rate = learning_rate

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate

        running_loss = 0.0

        for batch_observations, batch_actions in training_loader:
            batch_observations = batch_observations.to(device)
            batch_actions = batch_actions.to(device)

            reconstructed, latent_observations, predicted_actions = kae(
                batch_observations
            )
            latent_actions = kae.encoder(batch_actions)
            predicted_latent = latent_observations @ kae.K.T

            reconstruction_loss = mse(
                reconstructed,
                batch_observations,
            )
            prediction_loss = mse(
                predicted_actions,
                batch_actions,
            )
            latent_loss = mse(
                predicted_latent,
                latent_actions,
            )
            action_loss = mse(
                predicted_actions[:, :action_dim],
                batch_actions[:, :action_dim],
            )

            loss = (
                0.1
                * (
                    reconstruction_loss
                    + prediction_loss
                    + latent_loss
                )
                + 0.9 * action_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kae.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                all_latent_observations = kae.encoder(padded_observations)
                all_latent_actions = kae.encoder(padded_actions)
                kae.update_koopman_operator(
                    all_latent_observations,
                    all_latent_actions,
                )

        average_loss = running_loss / len(training_loader)

        if not torch.isfinite(torch.tensor(average_loss)):
            raise RuntimeError("Non-finite KAE loss detected.")

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(
                "[KAE]",
                task_name,
                "epoch",
                epoch + 1,
                "/",
                num_epochs,
                "loss",
                f"{average_loss:.6f}",
            )

    kae.eval()
    kae = kae.cpu()

    output_file = kae_directory / f"{task_name}_KAE.pth"
    checkpoint = {
        "format_version": 1,
        "state_dim": padded_dimension,
        "hidden_dim": hidden_dim,
        "observable_dim": observable_dim,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "state_dict": kae.state_dict(),
    }
    torch.save(checkpoint, output_file)

    print("[KAE] Saved:", output_file)
    return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Approximate a trained policy with a Koopman autoencoder."
    )
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--policy_file", required=True)
    parser.add_argument("--observation_file", required=True)
    parser.add_argument("--kae_directory", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--observable_dim", type=int, default=16)
    parser.add_argument("--padded_dimension", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sample_count", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()

    policy_data = torch.load(
        args.policy_file,
        map_location=args.device,
        weights_only=False,
    )
    policy = policy_data["actor"]

    train_and_save_kae(
        task_name=args.task_name,
        policy=policy,
        observation_file=args.observation_file,
        kae_directory=args.kae_directory,
        device=args.device,
        observable_dim=args.observable_dim,
        padded_dimension=args.padded_dimension,
        hidden_dim=args.hidden_dim,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()