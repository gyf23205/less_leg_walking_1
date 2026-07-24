import argparse
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

    def forward(self, inputs):
        return self.network(inputs)


class Decoder(nn.Module):
    def __init__(self, observable_dim, state_dim):
        super().__init__()
        self.linear = nn.Linear(observable_dim, state_dim, bias=False)

    def forward(self, inputs):
        return self.linear(inputs)


class KoopmanAutoencoderWalk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim):
        super().__init__()
        self.encoder = EncoderWalk(state_dim, hidden_dim, observable_dim)
        self.decoder = Decoder(observable_dim, state_dim)
        self.register_buffer("K", torch.randn(observable_dim, observable_dim))

    def forward(self, inputs):
        latent = self.encoder(inputs)
        predicted_latent = latent @ self.K.T
        return (
            self.decoder(latent),
            latent,
            self.decoder(predicted_latent),
        )

    def update_koopman_operator(self, latent_inputs, latent_outputs):
        latent_inputs = latent_inputs.reshape(-1, latent_inputs.shape[-1]).T
        latent_outputs = latent_outputs.reshape(-1, latent_outputs.shape[-1]).T
        self.K = latent_outputs @ torch.linalg.pinv(
            latent_inputs,
            rcond=1e-5,
        )

def load_kae(checkpoint_file, device):
    import sys
    import types

    legacy_module = types.ModuleType("Autoencoder")

    legacy_module.Encoder_walk = (
        EncoderWalk
    )

    legacy_module.Decoder = Decoder

    legacy_module.KoopmanAutoencoder_walk = (
        KoopmanAutoencoderWalk
    )

    if "Autoencoder" not in sys.modules:
        sys.modules["Autoencoder"] = (
            legacy_module
        )

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported KAE checkpoint type: "
            + str(type(checkpoint))
        )

    model = KoopmanAutoencoderWalk(
        checkpoint["state_dim"],
        checkpoint["hidden_dim"],
        checkpoint["observable_dim"],
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    return model.to(device)

def get_observation_tensor(observations):
    while not torch.is_tensor(observations):
        if not hasattr(observations, "keys"):
            raise TypeError(
                "Observation history must contain a Tensor."
            )

        keys = list(observations.keys())

        if "policy" in keys:
            observations = observations["policy"]
            continue

        if "obs" in keys:
            observations = observations["obs"]
            continue

        tensor_keys = []

        for key in keys:
            value = observations[key]

            if torch.is_tensor(value):
                tensor_keys.append(key)

        if len(tensor_keys) != 1:
            raise KeyError(
                "Could not identify the policy observation Tensor. "
                f"Available keys: {keys}"
            )

        observations = observations[tensor_keys[0]]

    return observations


def train_and_save_kae(
    task_name,
    policy,
    observation_file,
    kae_directory,
    device="cuda",
    observable_dim=16,
    padded_dimension=256,
    hidden_dim=256,
    sample_count=10000,
    batch_size=2048,
    num_epochs=3000, #total epoch
    epochs_schedule = [1000, 2000],
    epochs_schedule_lr = [1e-3, 1e-4, 1e-5]
    ):

    device = torch.device(device)
    loaded_observations = torch.load(observation_file, map_location="cpu", weights_only=False)
    observations = get_observation_tensor(loaded_observations)
    observations = observations.float()
    observations = observations.reshape(-1, observations.shape[-1])

    mean = torch.mean(        observations,        dim=0,    )
    covariance = torch.cov(        observations.T    )
    covariance = covariance + (        1e-5        * torch.eye(observations.shape[-1],dtype=observations.dtype,device=observations.device))

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

    observation_dim = sampled_observations.shape[-1]
    action_dim = target_actions.shape[-1]

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
    

    loader = DataLoader(
        TensorDataset(padded_observations, padded_actions),
        batch_size=batch_size,
        shuffle=False,
    )
    model = KoopmanAutoencoderWalk(
        padded_dimension,
        hidden_dim,
        observable_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        if epoch > epochs_schedule[1]:
            learning_rate = epochs_schedule_lr[2]
        elif epoch > epochs_schedule[0]:
            learning_rate = epochs_schedule_lr[1]
        else:
            learning_rate = epochs_schedule_lr[0]

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_prediction_loss = 0.0
        running_latent_loss = 0.0
        running_action_loss = 0.0

        for batch_observations, batch_actions in loader:
            reconstructed, latent_observations, predicted_actions = model(
                batch_observations
            )
            latent_actions = model.encoder(
                batch_actions
            )

            predicted_latent = (
                latent_observations @ model.K.T
            )

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

            kae_loss = (
                reconstruction_loss
                + prediction_loss
                + latent_loss
            )
            loss = 0.1 * kae_loss + 0.9 * action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.detach().item()
            running_reconstruction_loss += (
                reconstruction_loss.detach().item()
            )
            running_prediction_loss += (
                prediction_loss.detach().item()
            )
            running_latent_loss += (
                latent_loss.detach().item()
            )
            running_action_loss += (
                action_loss.detach().item()
            )

            with torch.no_grad():
                all_input_latent = model.encoder(padded_observations)
                all_output_latent = model.encoder(padded_actions)

            model.update_koopman_operator(all_input_latent,all_output_latent)

        batch_count = len(loader)

        average_loss = (
            running_loss / batch_count
        )

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(
                "[KAE]",
                task_name,
                "epoch",
                epoch + 1,
                "total",
                average_loss,
                "    ||  reconstruction",
                running_reconstruction_loss / batch_count,
                "prediction",
                running_prediction_loss / batch_count,
                "latent",
                running_latent_loss / batch_count,
                "action",
                running_action_loss / batch_count,
            )

    kae_directory = Path(kae_directory)
    kae_directory.mkdir(parents=True, exist_ok=True)
    output_file = kae_directory / f"{task_name}_KAE.pth"
    checkpoint = {
        "state_dim": padded_dimension,
        "hidden_dim": hidden_dim,
        "observable_dim": observable_dim,
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(checkpoint, output_file)
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--kae-dir", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    policy_data = torch.load(
        arguments.policy,
        map_location=arguments.device,
        weights_only=False,
    )
    if isinstance(policy_data, nn.Module):
        policy = policy_data
    elif policy_data.get("policy") is not None:
        policy = policy_data["policy"]
    else:
        policy = policy_data["actor"]

    train_and_save_kae(
        arguments.task,
        policy,
        arguments.observations,
        arguments.kae_dir,
        arguments.device,
    )


if __name__ == "__main__":
    main()