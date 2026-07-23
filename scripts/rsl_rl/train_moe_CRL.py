import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ORIGINAL_TASK = "Less-AnymalC-Rough-Walking-Direct-v1" # Assume the original task's 'KAE' is given
TRAIN_TASKS = [ # train in this order
     "Less-Leg-Flat-Walking-Direct-v1",
    # "Less-Leg-Rough-Walking-Direct-v1",
     "Less-AnymalC-Jump-Direct-v1",
    # "Less-AnymalC-Jump-Rough-Direct-v1",
]

TASK_DIRECTORY = Path(
    "/home/joonwon/github/less_leg_walking_1.worktrees/origin-master/"
    "source/less_leg_walking_1/less_leg_walking_1/tasks/direct/"
    "less_leg_walking_1"
)

KAE_DIRECTORY = TASK_DIRECTORY / "KAEs"
KAE_APPROX_FILE = (
    TASK_DIRECTORY
    / "agents"
    / "KAE_approx.py"
)

class ObservationLogger:
    def __init__(
        self,
        env,
        log_dir,
        max_samples=30000,
    ):
        self.env = env
        self.original_step = env.step
        self.log_dir = Path(log_dir)
        self.max_samples = max_samples

        self.buffer = None
        self.sample_count = 0
        self.write_index = 0

        env.step = self.step

    def get_policy_observation(self):
        observations = self.env.get_observations()

        if isinstance(observations, tuple):
            observations = observations[0]

        if hasattr(observations, "get"):
            policy_observation = observations.get("policy")

            if policy_observation is not None:
                observations = policy_observation

        if not torch.is_tensor(observations):
            raise TypeError(
                "Policy observation must be a Tensor."
            )

        return observations

    def step(self, actions):
        observations = self.get_policy_observation()

        observations = observations.detach()
        observations = observations.reshape(
            -1,
            observations.shape[-1],
        )
        observations = observations.cpu()

        if self.buffer is None:
            self.buffer = torch.empty(
                self.max_samples,
                observations.shape[-1],
                dtype=observations.dtype,
            )

        if observations.shape[0] >= self.max_samples:
            observations = observations[
                -self.max_samples:
            ]

            self.buffer.copy_(observations)
            self.sample_count = self.max_samples
            self.write_index = 0

        else:
            first_count = min(
                observations.shape[0],
                self.max_samples - self.write_index,
            )

            self.buffer[
                self.write_index:
                self.write_index + first_count
            ].copy_(
                observations[:first_count]
            )

            remaining_count = (
                observations.shape[0] - first_count
            )

            if remaining_count > 0:
                self.buffer[
                    :remaining_count
                ].copy_(
                    observations[first_count:]
                )

            self.write_index = (
                self.write_index
                + observations.shape[0]
            ) % self.max_samples

            self.sample_count = min(
                self.max_samples,
                self.sample_count
                + observations.shape[0],
            )

        return self.original_step(actions)

    def save(self):
        if self.buffer is None:
            raise RuntimeError(
                "No observations were recorded."
            )

        observation_file = (
            self.log_dir / "observations.pt"
        )

        observations = self.buffer[
            :self.sample_count
        ].clone()

        torch.save(
            observations,
            observation_file,
        )

        self.buffer = None

        return observation_file


def get_run_directories():
    run_directories = set()
    for model_file in Path("logs/rsl_rl").rglob(
        "complete_model_with_metadata.pth"
    ):
        run_directories.add(model_file.parent)
    return run_directories

def copy_results(task_name, run_directory):
    policy_file = run_directory / "complete_model_with_metadata.pth"
    observation_file = run_directory / "observations.pt"
    if not policy_file.is_file():
        raise FileNotFoundError(str(policy_file))
    if not observation_file.is_file():
        raise FileNotFoundError(str(observation_file))
    shutil.copy2(
        policy_file,
        KAE_DIRECTORY / f"{task_name}_policy.pth",
    )
    shutil.copy2(
        observation_file,
        KAE_DIRECTORY / f"{task_name}_observations.pt",
    )

def main():
    KAE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    original_kae_file = KAE_DIRECTORY / f"{ORIGINAL_TASK}_KAE.pth"
    if not original_kae_file.is_file():
        raise FileNotFoundError(str(original_kae_file))
    if not KAE_APPROX_FILE.is_file():
        raise FileNotFoundError(str(KAE_APPROX_FILE))

    train_script = Path(__file__).with_name("train_moe.py")
    if not train_script.is_file():
        raise FileNotFoundError(str(train_script))

    train_arguments = sys.argv[1:]
    completed_tasks = [ORIGINAL_TASK]

    for task_name in TRAIN_TASKS:
        runs_before = get_run_directories()
        command = [
            sys.executable,
            str(train_script),
            "--task",
            task_name,
        ]
        command.extend(train_arguments)

        environment = os.environ.copy()
        environment["CRL_MODE"] = "1"
        environment["CRL_OBSERVATION_LOG"] = "1"
        environment["CRL_TRAIN_KAE"] = "1"
        environment["CRL_TASK_NAME"] = task_name
        environment["CRL_KAE_TASKS"] = "|".join(completed_tasks)
        environment["CRL_KAE_DIRECTORY"] = str(KAE_DIRECTORY)
        environment["CRL_KAE_APPROX_FILE"] = str(KAE_APPROX_FILE)

        print("[CRL] Training:", task_name)
        print("[CRL] Previous KAEs:", completed_tasks)
        subprocess.run(command, env=environment, check=True)

        new_runs = get_run_directories() - runs_before
        if len(new_runs) != 1:
            raise RuntimeError("New training run was not identified.")

        run_directory = new_runs.pop()
        copy_results(task_name, run_directory)

        task_kae_file = KAE_DIRECTORY / f"{task_name}_KAE.pth"
        if not task_kae_file.is_file():
            raise FileNotFoundError(str(task_kae_file))

        completed_tasks.append(task_name)
        print("[CRL] Completed:", task_name)

if __name__ == "__main__":
    main()