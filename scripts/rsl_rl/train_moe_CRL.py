import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ORIGINAL_TASK = "Less-AnymalC-Rough-Walking-Direct-v1"
TRAIN_TASKS = [
    "Less-Leg-Flat-Walking-Direct-v1",
    # "Less-Leg-Rough-Walking-Direct-v1",
    # "Less-AnymalC-Jump-Direct-v1",
    # "Less-AnymalC-Jump-Rough-Direct-v1",
]

KAE_DIRECTORY = Path(
    "/home/joonwon/github/less_leg_walking_1.worktrees/origin-master/"
    "source/less_leg_walking_1/less_leg_walking_1/tasks/direct/"
    "less_leg_walking_1/agents/KAEs"
)
KAE_APPROX_FILE = KAE_DIRECTORY.parent / "KAE_approx.py"
ORIGINAL_POLICY_FILE = KAE_DIRECTORY / f"{ORIGINAL_TASK}_policy.pth"
ORIGINAL_OBSERVATION_FILE = KAE_DIRECTORY / f"{ORIGINAL_TASK}_observations.pt"


class ObservationLogger:
    def __init__(self, env, log_dir):
        self.env = env
        self.original_step = env.step
        self.log_dir = Path(log_dir)
        self.observations = []
        env.step = self.step

    def get_policy_observation(self):
        observations = self.env.get_observations()
        if isinstance(observations, tuple):
            observations = observations[0]
        if hasattr(observations, "get"):
            policy_observation = observations.get("policy")
            if policy_observation is not None:
                observations = policy_observation
        return observations

    def step(self, actions):
        observations = self.get_policy_observation()
        self.observations.append(observations.detach().cpu())
        return self.original_step(actions)

    def save(self):
        if not self.observations:
            raise RuntimeError("No policy observations were recorded.")
        output_file = self.log_dir / "observations.pt"
        torch.save(torch.cat(self.observations, dim=0), output_file)
        return output_file


def find_run_files():
    return set(Path("logs/rsl_rl").rglob("complete_model_with_metadata.pth"))


def run_original_approximation():
    command = [
        sys.executable,
        str(KAE_APPROX_FILE),
        "--task",
        ORIGINAL_TASK,
        "--policy",
        str(ORIGINAL_POLICY_FILE),
        "--observations",
        str(ORIGINAL_OBSERVATION_FILE),
        "--kae-dir",
        str(KAE_DIRECTORY),
        "--device",
        os.environ.get("CRL_KAE_DEVICE", "cuda"),
    ]
    subprocess.run(command, check=True)


def copy_results(task_name, model_file):
    run_directory = model_file.parent
    observation_file = run_directory / "observations.pt"
    if not observation_file.is_file():
        raise FileNotFoundError(str(observation_file))
    shutil.copy2(
        model_file,
        KAE_DIRECTORY / f"{task_name}_policy.pth",
    )
    shutil.copy2(
        observation_file,
        KAE_DIRECTORY / f"{task_name}_observations.pt",
    )


def main():
    KAE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    run_original_approximation()

    train_script = Path(__file__).with_name("train_moe.py")
    train_arguments = sys.argv[1:]
    completed_tasks = [ORIGINAL_TASK]

    for task_name in TRAIN_TASKS:
        runs_before = find_run_files()
        environment = os.environ.copy()
        environment["CRL_MODE"] = "1"
        environment["CRL_TASK_NAME"] = task_name
        environment["CRL_KAE_TASKS"] = "|".join(completed_tasks)
        environment["CRL_KAE_DIRECTORY"] = str(KAE_DIRECTORY)
        environment["CRL_KAE_APPROX_FILE"] = str(KAE_APPROX_FILE)

        command = [
            sys.executable,
            str(train_script),
            "--task",
            task_name,
        ]
        command.extend(train_arguments)

        print("[CRL] Training:", task_name)
        subprocess.run(command, env=environment, check=True)

        new_runs = find_run_files() - runs_before
        if len(new_runs) != 1:
            raise RuntimeError("New training run was not identified.")

        model_file = new_runs.pop()
        copy_results(task_name, model_file)

        kae_file = KAE_DIRECTORY / f"{task_name}_KAE.pth"
        if not kae_file.is_file():
            raise FileNotFoundError(str(kae_file))

        completed_tasks.append(task_name)
        print("[CRL] Completed:", task_name)


if __name__ == "__main__":
    main()