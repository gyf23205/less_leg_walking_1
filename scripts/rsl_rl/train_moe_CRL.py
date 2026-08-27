# NOTE
# TO do:
# 1. CRL continuation: make a certain step is reproducible, our be continued
#    1.5 Make each terminal run separate s.t. I can run several trials at the same time.
# 2. print and save KAE train result. 


import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ORIGINAL_TASK = "Less-AnymalC-Rough-Walking-Direct-v1" # Assume the original task's 'KAE' is given

# TRAIN_TASKS = [ # train in this order
#     "Less-AnymalC-Jump-Rough-Direct-v1",
# ]

TRAIN_TASKS = [ # train in this order
    "Less-AnymalC-Jump-Direct-v1",
    # "Less-AnymalC-Rough-Walking-Direct-v1",
    "Less-AnymalC-Flat-Walking-Direct-v1",
    "Less-Leg-Flat-Walking-Direct-v1",
    "Less-AnymalC-Jump-Rough-Direct-v1",
    "Less-Leg-Rough-Walking-Direct-v1",
]

TASK_DIRECTORY = Path("source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1")
KAE_ROOT = TASK_DIRECTORY / "KAEs"            # shared; holds the original task KAE only
SESSION_ROOT = KAE_ROOT / "sessions"          # per-session artifacts
LOG_ROOT = Path("logs/rsl_rl")
KAE_APPROX_FILE = TASK_DIRECTORY / "agents" / "KAE_approx.py"

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

USAGE = """\
train_moe_CRL.py [--session ID] [--resume | --start-from N|TASK] [--list-sessions] [--crl-help]
           [... every other argument is forwarded to train_moe.py ...]

  --session ID       Session name. Defaults to sYYYYmmdd_HHMMSS.
                     KAEs, policies, observations and logs are kept per session,
                     so several trials can run concurrently.
  --resume           Continue this session from crl_state.json.
  --start-from N     Start at the N-th task (1-based). A task name also works.
                     The *_KAE.pth of every preceding task must already exist
                     in the session directory.
  --list-sessions    Print all sessions and their progress.
"""


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
    
# --------------------------------------------------------------------------------------
# Session management
# --------------------------------------------------------------------------------------
def session_directory(session_id):
    return SESSION_ROOT / session_id


def state_file(session_id):
    return session_directory(session_id) / "crl_state.json"


def load_state(session_id):
    path = state_file(session_id)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(session_id, state):
    path = state_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def list_sessions():
    if not SESSION_ROOT.is_dir():
        print("No session found:", SESSION_ROOT)
        return
    directories = sorted(p for p in SESSION_ROOT.iterdir() if p.is_dir())
    if not directories:
        print("No session found:", SESSION_ROOT)
        return
    for directory in directories:
        state = load_state(directory.name)
        done = state["completed_tasks"][1:] if state else []
        remaining = TRAIN_TASKS[len(done)] if len(done) < len(TRAIN_TASKS) else "(all done)"
        print(f"{directory.name:<24} {len(done)}/{len(TRAIN_TASKS)}  next: {remaining}")


def resolve_start_index(state, start_from):
    """Return the index into TRAIN_TASKS at which this invocation should start."""
    if start_from is None:
        completed = state.get("completed_tasks", [ORIGINAL_TASK])[1:]
        if completed != TRAIN_TASKS[:len(completed)]:
            raise RuntimeError(
                f"completed_tasks in crl_state.json does not follow TRAIN_TASKS order: {completed}"
            )
        return len(completed)
    if start_from.isdigit():
        index = int(start_from) - 1
    elif start_from in TRAIN_TASKS:
        index = TRAIN_TASKS.index(start_from)
    else:
        raise ValueError(
            f"Invalid --start-from value: {start_from}\n"
            f"Expected 1..{len(TRAIN_TASKS)} or one of {TRAIN_TASKS}"
        )
    if not 0 <= index <= len(TRAIN_TASKS):
        raise ValueError(f"--start-from out of range: {start_from}")
    return index


def verify_previous_artifacts(kae_directory, completed_tasks):
    missing = [
        task for task in completed_tasks
        if not (kae_directory / f"{task}_KAE.pth").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Resuming requires the KAE of every preceding task in the session directory.\n"
            f"  session directory : {kae_directory}\n"
            f"  missing files     : {[f'{t}_KAE.pth' for t in missing]}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--crl-help", action="store_true")
    return parser.parse_known_args()

# --------------------------------------------------------------------------------------
# Run directory resolution
# --------------------------------------------------------------------------------------
def get_run_directories():
    run_directories = set()
    for model_file in Path("logs/task1").rglob(
        "complete_model_with_metadata.pth"
    ):
        run_directories.add(model_file.parent)
    return run_directories


def resolve_new_run(runs_before, runs_after, session_id, started_at):
    """Pick this session's own run even when several trials run concurrently."""
    candidates = runs_after - runs_before
    tagged = {d for d in candidates if session_id in d.name}
    if tagged:
        candidates = tagged
    else:
        candidates = {d for d in candidates if d.stat().st_mtime >= started_at - 5.0}
    if not candidates:
        raise RuntimeError("New training run was not identified.")
    if len(candidates) > 1:
        newest = max(candidates, key=lambda d: d.stat().st_mtime)
        print(f"[CRL][WARN] {len(candidates)} new runs found; picking the newest: {newest}")
        print("[CRL][WARN] The CRL_SESSION_ID -> run_name patch in train_moe.py may be missing.")
        return newest
    return candidates.pop()


def copy_results(task_name, run_directory, kae_directory):
    policy_file = run_directory / "complete_model_with_metadata.pth"
    observation_file = run_directory / "observations.pt"
    if not policy_file.is_file():
        raise FileNotFoundError(str(policy_file))
    if not observation_file.is_file():
        raise FileNotFoundError(str(observation_file))
    shutil.copy2(policy_file, kae_directory / f"{task_name}_policy.pth")
    shutil.copy2(observation_file, kae_directory / f"{task_name}_observations.pt")


def main():
    arguments, train_arguments = parse_arguments()

    if arguments.crl_help:
        print(USAGE)
        return
    if arguments.list_sessions:
        list_sessions()
        return

    SESSION_ROOT.mkdir(parents=True, exist_ok=True)

    session_id = arguments.session or datetime.now().strftime("s%Y%m%d_%H%M%S")
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError(f"Session id may only contain alphanumerics and . _ - : {session_id}")

    kae_directory = session_directory(session_id)
    kae_log_directory = kae_directory / "kae_logs"
    resuming = arguments.resume or arguments.start_from is not None

    if kae_directory.exists() and not resuming:
        raise FileExistsError(
            f"Session already exists: {kae_directory}\n"
            "Use --resume to continue, or --start-from to begin at a specific task."
        )
    kae_directory.mkdir(parents=True, exist_ok=True)
    kae_log_directory.mkdir(parents=True, exist_ok=True)

    # Copy the original KAE into the session so the session is self-contained.
    original_source = KAE_ROOT / f"{ORIGINAL_TASK}_KAE.pth"
    original_target = kae_directory / f"{ORIGINAL_TASK}_KAE.pth"
    if not original_target.is_file():
        if not original_source.is_file():
            raise FileNotFoundError(str(original_source))
        shutil.copy2(original_source, original_target)

    if not KAE_APPROX_FILE.is_file():
        raise FileNotFoundError(str(KAE_APPROX_FILE))
    train_script = Path(__file__).with_name("train_moe.py")
    if not train_script.is_file():
        raise FileNotFoundError(str(train_script))

    state = load_state(session_id) if resuming else None
    if state is None:
        state = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "original_task": ORIGINAL_TASK,
            "train_tasks": list(TRAIN_TASKS),
            "completed_tasks": [ORIGINAL_TASK],
            "runs": {},
        }

    start_index = resolve_start_index(state, arguments.start_from)
    state["completed_tasks"] = [ORIGINAL_TASK] + TRAIN_TASKS[:start_index]
    state["train_arguments"] = train_arguments
    verify_previous_artifacts(kae_directory, state["completed_tasks"])
    save_state(session_id, state)

    next_task = TRAIN_TASKS[start_index] if start_index < len(TRAIN_TASKS) else "(none)"
    print(f"[CRL] session     : {session_id}")
    print(f"[CRL] session dir : {kae_directory}")
    print(f"[CRL] start from  : #{start_index + 1} {next_task}")
    print(f"[CRL] completed   : {state['completed_tasks']}")

    for index in range(start_index, len(TRAIN_TASKS)):
        task_name = TRAIN_TASKS[index]
        runs_before = get_run_directories()
        started_at = time.time()

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
        environment["CRL_KAE_TASKS"] = "|".join(state["completed_tasks"])
        environment["CRL_KAE_DIRECTORY"] = str(kae_directory)          # session directory
        environment["CRL_KAE_APPROX_FILE"] = str(KAE_APPROX_FILE)
        environment["CRL_SESSION_ID"] = session_id                     # new
        environment["CRL_KAE_LOG_DIRECTORY"] = str(kae_log_directory)  # new
        environment["CRL_TASK_INDEX"] = str(index + 1)                 # new

        print(f"\n[CRL] ({index + 1}/{len(TRAIN_TASKS)}) Training: {task_name}")
        print("[CRL] Previous KAEs:", state["completed_tasks"])
        subprocess.run(command, env=environment, check=True)

        run_directory = resolve_new_run(
            runs_before, get_run_directories(), session_id, started_at
        )
        copy_results(task_name, run_directory, kae_directory)

        task_kae_file = kae_directory / f"{task_name}_KAE.pth"
        if not task_kae_file.is_file():
            raise FileNotFoundError(str(task_kae_file))

        state["completed_tasks"].append(task_name)
        state["runs"][task_name] = str(run_directory)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_state(session_id, state)  # saved per task so --resume works after a crash
        print(f"[CRL] Completed: {task_name}  (run: {run_directory})")

    print(f"\n[CRL] Session {session_id} finished.")


if __name__ == "__main__":
    main()