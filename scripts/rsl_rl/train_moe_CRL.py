import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

original_task = "Less-AnymalC-Rough-Walking-Direct-v1"
train_tasks = [
    #"Less-AnymalC-Rough-Walking-Direct-v1",
    "Less-Leg-Flat-Walking-Direct-v1",
    #"Less-Leg-Rough-Walking-Direct-v1",
    #"Less-AnymalC-Jump-Direct-v1",
    #"Less-AnymalC-Jump-Rough-Direct-v1",
]

kae_path = Path(
    "/home/joonwon/github/less_leg_walking_1.worktrees/origin-master/source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/KAEs"
)


class ActionLogger:
    def __init__(self, env, log_dir):
        self.original_step = env.step
        self.actions = []
        self.log_dir = log_dir
        env.step = self.step

    def step(self, actions):
        self.actions.append(actions.detach().cpu())
        return self.original_step(actions)

    def save(self):
        action_file = os.path.join(self.log_dir, "actions.pt")
        torch.save(torch.stack(self.actions), action_file)

def get_run_directories():
    run_directories = set()
    for model_file in Path("logs/rsl_rl").glob("*/*/model_*.pt"):
        run_directories.add(model_file.parent)
    return run_directories

def get_final_model(run_directory):
    final_model = None
    final_iteration = -1
    for model_file in run_directory.glob("model_*.pt"):
        iteration_text = model_file.stem.replace("model_", "")
        if not iteration_text.isdigit():
            continue

        iteration = int(iteration_text)
        if iteration > final_iteration:
            final_model = model_file
            final_iteration = iteration

    if final_model is None:
        raise FileNotFoundError("Final checkpoint was not found.")
    return final_model

def copy_results(task_name, run_directory):
    model_file = get_final_model(run_directory)
    action_file = run_directory / "actions.pt"
    if not action_file.is_file():
        raise FileNotFoundError(str(action_file))

    shutil.copy2(model_file, kae_path / (task_name + "_policy.pt"))
    shutil.copy2(action_file, kae_path / (task_name + "_actions.pt"))

def main():
    kae_path.mkdir(parents=True, exist_ok=True)
    original_kae = list(kae_path.glob("*" + original_task + "*.pth"))
    if len(original_kae) == 0:
        raise FileNotFoundError("Original-task KAE was not found.")

    train_script = Path(__file__).with_name("train_moe.py")
    train_arguments = sys.argv[1:]

    for task_name in train_tasks:
        runs_before = get_run_directories()
        command = [sys.executable, str(train_script), "--task", task_name]
        command.extend(train_arguments)

        environment = os.environ.copy()
        environment["CRL_ACTION_LOG"] = "1"
        environment["CRL_TASK_NAME"] = task_name

        print("[CRL] Training:", task_name)
        subprocess.run(command, env=environment, check=True)

        new_runs = get_run_directories() - runs_before
        if len(new_runs) != 1:
            raise RuntimeError("New experiment/time log was not identified.")

        run_directory = new_runs.pop()
        copy_results(task_name, run_directory)
        print("[CRL] Saved:", task_name)

if __name__ == "__main__":
    main()