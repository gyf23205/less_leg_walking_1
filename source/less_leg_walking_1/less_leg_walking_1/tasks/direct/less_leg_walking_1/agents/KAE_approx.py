from pathlib import Path
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset


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
        self.encoder = Encoder_walk(
            state_dim,
            hidden_dim,
            observable_dim,
        )
        self.decoder = Decoder(
            observable_dim,
            state_dim,
        )
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.observable_dim = observable_dim
        self.K = torch.randn(
            observable_dim,
            observable_dim,
            device=device,
        )

    def forward(self, x):
        z = self.encoder(x)
        z_next = torch.matmul(z, self.K.T)
        y_hat = self.decoder(z_next)
        x_hat = self.decoder(z)
        return x_hat, z, y_hat

    def compute_koopman_operator(
        self,
        latent_X,
        latent_Y,
        device,
    ):
        del device
        latent_X = latent_X.view(
            -1,
            latent_X.size(-1),
        ).T
        latent_Y = latent_Y.view(
            -1,
            latent_Y.size(-1),
        ).T
        self.K = latent_Y @ torch.linalg.pinv(
            latent_X,
            rcond=1e-5,
        )

# def run_policy(policy, observations):
#     if hasattr(policy, "act_inference"):
#         policy_output = policy.act_inference(
#             observations
#         )
#     else:
#         policy_output = policy(
#             observations
#         )

#     if isinstance(policy_output, tuple):
#         policy_output = policy_output[0]

#     if not torch.is_tensor(policy_output):
#         raise TypeError(
#             "Policy output must be a Tensor."
#         )

#     return policy_output


def _install_legacy_autoencoder_module():
    module = types.ModuleType("Autoencoder")
    module.Encoder_walk = Encoder_walk
    module.Decoder = Decoder
    module.KoopmanAutoencoder_walk = KoopmanAutoencoder_walk
    sys.modules["Autoencoder"] = module
    sys.modules[
        "less_leg_walking_1.tasks.direct.less_leg_walking_1.Autoencoder"
    ] = module


def load_kae(checkpoint_file, device):
    _install_legacy_autoencoder_module()

    kae = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(kae, nn.Module):
        raise TypeError(
            f"KAE file must contain a full torch.nn.Module: "
            f"{checkpoint_file}"
        )

    kae = kae.to(device)
    kae.K = kae.K.to(device)
    kae.eval()
    return kae


def koopman_loss(
    x,
    x_hat,
    latent_x,
    y_seq_states,
    y_seq_latents,
    p,
    action_dim,
    model,
):
    mse_loss = nn.MSELoss()

    recon_loss = mse_loss(x_hat, x)
    state_pred_loss = 0.0
    latent_pred_loss = 0.0
    action_loss = 0.0

    Ks = [
        torch.linalg.matrix_power(
            model.K,
            step,
        )
        for step in range(1, p + 1)
    ]

    for k in range(p):
        pred_lat_k = latent_x @ Ks[k].T
        pred_y_k = model.decoder(pred_lat_k)

        state_pred_loss = (
            state_pred_loss
            + mse_loss(
                pred_y_k,
                y_seq_states[:, k, :],
            )
        )
        latent_pred_loss = (
            latent_pred_loss
            + mse_loss(
                pred_lat_k,
                y_seq_latents[:, k, :],
            )
        )

        if k == p - 1:
            action_loss = (
                action_loss
                + mse_loss(
                    pred_y_k[:, :action_dim],
                    y_seq_states[
                        :,
                        k,
                        :action_dim,
                    ],
                )
            )

    state_pred_loss = state_pred_loss / p
    latent_pred_loss = latent_pred_loss / p

    return (
        recon_loss,
        state_pred_loss,
        latent_pred_loss,
        action_loss,
    )


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
    del aug_output
    del aug_input_all

    B = aug_input.size(0)
    start = inner * batch_size
    end = start + B

    x = aug_input.to(device).squeeze(1)
    x_hat, latent_x, _ = kae(x)

    y_seq_states = []
    y_seq_latents = []

    for k in range(p):
        if end + k > len(aug_output_all):
            break

        yk = (
            aug_output_all[
                start + k : end + k
            ]
            .to(device)
            .squeeze(1)
        )
        _, latent_yk, _ = kae(yk)

        y_seq_states.append(yk)
        y_seq_latents.append(latent_yk)

    if not y_seq_states:
        raise RuntimeError(
            "No future states available for multi-step supervision."
        )

    if len(y_seq_states) < p:
        last_y = y_seq_states[-1]
        last_ly = y_seq_latents[-1]

        for _ in range(
            p - len(y_seq_states)
        ):
            y_seq_states.append(last_y)
            y_seq_latents.append(last_ly)

    y_seq_states = torch.stack(
        y_seq_states,
        dim=1,
    )
    y_seq_latents = torch.stack(
        y_seq_latents,
        dim=1,
    )

    (
        recon_loss,
        state_pred_loss,
        koopman_pred_loss,
        loss_action,
    ) = koopman_loss(
        x,
        x_hat,
        latent_x,
        y_seq_states,
        y_seq_latents,
        p,
        action_dim,
        kae,
    )

    loss_kae = (
        c1 * recon_loss
        + c2 * state_pred_loss
        + c3 * koopman_pred_loss
    )

    return loss_kae, loss_action


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
    num_epochs=5,
    learning_rate=1e-3,
    kae_coefficient=0.1,
    action_coefficient=0.9,
    c1=1.0,
    c2=1.0,
    c3=1.0,
    p=1,
):
    device = torch.device(device)
    observation_file = Path(
        observation_file
    )
    kae_directory = Path(
        kae_directory
    )

    if not observation_file.is_file():
        raise FileNotFoundError(
            str(observation_file)
        )

    kae_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    observations = torch.load(
        observation_file,
        map_location="cpu",
        weights_only=False,
    ).float()

    observations = observations.reshape(
        -1,
        observations.shape[-1],
    )

    obs_dim = observations.shape[-1]

    mu = observations.mean(dim=0)
    centered = observations - mu
    sigma = centered.T @ centered
    sigma = sigma / (
        observations.shape[0] - 1
    )
    sigma = 0.5 * (
        sigma + sigma.T
    )
    sigma = sigma + 1e-5 * torch.eye(
        obs_dim,
        dtype=sigma.dtype,
    )

    distribution = MultivariateNormal(
        mu.to(device),
        covariance_matrix=sigma.to(device),
    )

    policy = policy.to(device)
    policy.eval()

    with torch.no_grad():
        random_input = distribution.sample((sample_count,))
        policy_output = policy(random_input)

        if isinstance(policy_output, tuple):
            policy_output = policy_output[0]

        if not torch.is_tensor(policy_output):
            raise TypeError("Policy output must be a Tensor.")

    action_dim = policy_output.shape[-1]

    if action_dim > padded_dimension:
        raise ValueError(
            f"Action dimension {action_dim} exceeds padded dimension "
            f"{padded_dimension}."
        )

    aug_input = F.pad(
        random_input,
        (
            0,
            padded_dimension - obs_dim,
        ),
        value=1.0,
    )

    aug_output = F.pad(
        policy_output,
        (
            0,
            padded_dimension
            - action_dim,
        ),
        value=1.0,
    )

    aug_input_all = (
        aug_input.unsqueeze(1)
    )
    aug_output_all = (
        aug_output.unsqueeze(1)
    )

    train_loader = DataLoader(
        TensorDataset(
            aug_input_all,
            aug_output_all,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    kae = KoopmanAutoencoder_walk(
        state_dim=padded_dimension,
        hidden_dim=hidden_dim,
        observable_dim=observable_dim,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(kae.parameters(), lr=learning_rate)

    with torch.no_grad():
        _, latent_input_all, _ = kae(aug_input_all.squeeze(1))
        _, latent_output_all, _ = kae(aug_output_all.squeeze(1))
        kae.compute_koopman_operator(
            latent_input_all,
            latent_output_all,
            device,
        )

    for epoch in range(num_epochs):
        if epoch > 2000:
            current_lr = 1e-5
        elif epoch > 1000:
            current_lr = 1e-4
        else:
            current_lr = (
                learning_rate
            )

        for group in (
            optimizer.param_groups
        ):
            group["lr"] = current_lr

        running_loss = 0.0
        running_action_loss = 0.0

        for inner, (
            batch_input,
            batch_output,
        ) in enumerate(
            train_loader
        ):
            (
                loss_kae,
                loss_action,
            ) = compute_l_kae(
                kae,
                batch_input,
                batch_output,
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

            loss = (
                kae_coefficient
                * loss_kae
                + action_coefficient
                * loss_action
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                kae.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
            )
            running_action_loss += (
                loss_action.item()
            )

            with torch.no_grad():
                (
                    _,
                    latent_input_all,
                    _,
                ) = kae(
                    aug_input_all.squeeze(
                        1
                    )
                )
                (
                    _,
                    latent_output_all,
                    _,
                ) = kae(
                    aug_output_all.squeeze(
                        1
                    )
                )

                kae.compute_koopman_operator(
                    latent_input_all,
                    latent_output_all,
                    device,
                )

        if (
            epoch == 0
            or (epoch + 1) % 100
            == 0
        ):
            num_batches = len(
                train_loader
            )
            print(
                "[KAE]",
                task_name,
                "epoch",
                epoch + 1,
                "loss",
                running_loss
                / num_batches,
                "action_loss",
                running_action_loss
                / num_batches,
            )

    kae.eval()
    kae = kae.cpu()
    kae.K = kae.K.detach().cpu()

    _install_legacy_autoencoder_module()

    output_file = (
        kae_directory
        / f"{task_name}_KAE.pth"
    )

    torch.save(
        kae,
        output_file,
    )

    print(
        "[KAE] Saved:",
        output_file,
    )

    return output_file