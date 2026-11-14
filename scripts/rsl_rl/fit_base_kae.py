"""Fit the starting KAE of the KAE-MoE continual-RL chain to the shared base policy.

componet and residual start their task2 chains from base_models/rough_walking.pth. This
script gives the KAE-MoE chain the same starting knowledge: it rolls out that base policy
on its own task, records policy observations the same way train_moe_CRL.ObservationLogger
does (the last ``--max_samples`` rows), and fits a KAE with KAE_approx.train_and_save_kae,
using the same defaults train_moe.py uses for every later task.

Output (read by train_moe_CRL.py as ORIGINAL_TASK):
    <task dir>/KAEs/Base-Rough-Walking_KAE.pth
    <task dir>/KAEs/Base-Rough-Walking_observations.pt
    <task dir>/KAEs/kae_logs/Base-Rough-Walking_kae_{history.csv,summary.json}

Usage:
    python scripts/rsl_rl/fit_base_kae.py --headless
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fit the starting KAE to the shared base policy.")
parser.add_argument("--task", default="Less-AnymalC-Rough-Walking-Direct-v1", help="Task the base policy was trained on.")
parser.add_argument("--num_envs", type=int, default=1024, help="Parallel environments (task2 training uses 1024).")
parser.add_argument("--steps", type=int, default=300, help="Rollout steps; only the last --max_samples rows are kept.")
parser.add_argument("--max_samples", type=int, default=30000, help="Observation rows kept (ObservationLogger default).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--name", default="Base-Rough-Walking", help="KAE key; must equal ORIGINAL_TASK in train_moe_CRL.py.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.util  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import less_leg_walking_1.tasks  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[2]
TASK_DIR = REPO / "source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1"
BASE_MODEL = TASK_DIR / "base_models/rough_walking.pth"
KAE_ROOT = TASK_DIR / "KAEs"
KAE_APPROX_FILE = TASK_DIR / "agents/KAE_approx.py"


class BasePolicy(nn.Module):
    """Exposes the interface KAE_approx._policy_mean expects: actor_obs_normalizer and forward.

    The base model was trained without observation normalization, so the normalizer is the identity.
    """

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor
        self.actor_obs_normalizer = nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    torch.manual_seed(args_cli.seed)

    checkpoint = torch.load(BASE_MODEL, map_location=device, weights_only=False)
    policy = BasePolicy(checkpoint["actor"]).to(device).eval()

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))

    obs = env.get_observations()["policy"]
    expected = policy.actor[0].in_features
    if obs.shape[-1] != expected:
        raise RuntimeError(f"Base actor expects {expected} observations, {args_cli.task} gives {obs.shape[-1]}.")

    # Spread episodes over their whole length, as training does with init_at_random_ep_len=True.
    unwrapped = env.unwrapped
    unwrapped.episode_length_buf = torch.randint_like(unwrapped.episode_length_buf, high=int(unwrapped.max_episode_length))

    # Ring buffer of the last max_samples observation rows, as in ObservationLogger.
    rows = []
    kept = 0
    with torch.inference_mode():
        for step in range(args_cli.steps):
            rows.append(obs.detach().cpu())
            kept += obs.shape[0]
            while kept - rows[0].shape[0] >= args_cli.max_samples:
                kept -= rows.pop(0).shape[0]
            obs = env.step(policy(obs))[0]["policy"]
            if (step + 1) % 50 == 0:
                print(f"[BASE-KAE] rollout step {step + 1}/{args_cli.steps}")
    observations = torch.cat(rows)[-args_cli.max_samples:].clone()
    env.close()

    KAE_ROOT.mkdir(parents=True, exist_ok=True)
    observation_file = KAE_ROOT / f"{args_cli.name}_observations.pt"
    torch.save(observations, observation_file)
    print(f"[BASE-KAE] Saved {tuple(observations.shape)} observations to {observation_file}")

    spec = importlib.util.spec_from_file_location("crl_kae_approx", KAE_APPROX_FILE)
    kae_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kae_module
    spec.loader.exec_module(kae_module)

    torch.set_grad_enabled(True)
    kae_module.train_and_save_kae(
        task_name=args_cli.name,
        policy=policy,
        observation_file=observation_file,
        kae_directory=KAE_ROOT,
        device=device,
        log_directory=KAE_ROOT / "kae_logs",
    )
    print(f"[BASE-KAE] Saved {KAE_ROOT / (args_cli.name + '_KAE.pth')}")


if __name__ == "__main__":
    main()
    simulation_app.close()
