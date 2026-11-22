"""Root folder of the multi-seed continual-RL logs, shared by every train script.

Runs are written to <root>/<task name>/<method>/<seed>. The root is DEFAULT_LOG_ROOT unless the
CRL_LOG_ROOT environment variable is set (e.g. to a throwaway folder for smoke tests).

Experiments so far:
    logs/task2_v1_best_own   first 5-seed run; each method with its own tuned actor size, FAME a larger budget
    logs/task2_aligned       shared network (agents/shared_net_cfg.py) and per-task budget for every method
"""

import os

DEFAULT_LOG_ROOT = "logs/task2_aligned"


def crl_log_root() -> str:
    return os.environ.get("CRL_LOG_ROOT", DEFAULT_LOG_ROOT)
