"""Figures that compare the continual-learning methods of a multi-seed experiment (Isaac Sim is not needed).

Input: the TensorBoard files of ``<root>/<task>/<method>/<seed>[_<suffix>]/`` (layout of
``scripts/rsl_rl/run_task2_seeds.sh``). ``<root>`` defaults to the log root of ``scripts/common/log_root.py``.
Only the tags that every method logs are used: ``Train/mean_reward`` (mean episode return) and
``Train/mean_episode_length``. The x-axis is the cumulative number of PPO mini-batch updates for every method
(``scripts/common/tb_xaxis.py``).

Each run is interpolated onto a common grid of ``--points`` steps per task, from the latest first logged step to
the earliest last logged step of the runs on that task. A run that stops before 95% of the longest run on its task
is incomplete and left out (with a warning).

Metrics, per task i (from the episode return of the training episodes, stochastic policy):

- normalized return ``r_norm = (r - lo_i) / (hi_i - lo_i)``; ``lo_i``, ``hi_i``: minimum and maximum of all
  plotted runs on task i, so the values depend on the plotted methods.
- ``AUC_i``: mean of ``r_norm`` over the training of task i (area under the curve divided by its length).
- forward transfer (Wołczyk et al., 2021, "Continual World", arXiv:2105.10919, Eq. 2), with ``scratch``
  (a new policy per task) as baseline: ``FT_i = (AUC_i - AUC_i^b) / (1 - AUC_i^b)``. FT_i > 0: the method learns
  task i faster or better than a new policy; FT_i <= 1.
- final return: mean of ``r`` over the last ``--final_fraction`` of the training of task i.

Values are computed per seed; figures show the mean over seeds and the standard deviation (band / error bar).
Forgetting and transfer matrices need each stage's policy evaluated on every task, which the logs do not have.

Figures (PNG and PDF in ``--output_dir``):

- ``learning_curves``: episode return and episode length over the training of each task.
- ``final_return``: final episode return per task and method, with the value of each seed.
- ``forward_transfer``: forward transfer per method and task (diverging: blue above ``scratch``, red below).
- ``summary``: mean AUC, forward transfer and normalized final return over the tasks.

Example (Isaac Lab Python environment, it has ``tensorboard`` and ``matplotlib``):

    python scripts/visulization/plot_crl_results.py --root logs/task2_aligned
    python scripts/visulization/plot_crl_results.py --root logs/task2_v1_best_own --seeds 40 41 \\
        --output_dir docs/figures/task2_v1 --output_json docs/task2_v1_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from tensorboard.compat.proto import event_pb2  # noqa: E402
from tensorboard.util import tensor_util  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.log_root import crl_log_root  # noqa: E402

# Task folders in the order of the continual-learning chain (scripts/rsl_rl/run_task2_seeds.sh).
TASKS = [
    "anymal_c_jump_flat",
    "less_leg_walking_flat",
    "anymal_c_rough_leg_walking",
    "anymal_c_jump_rough",
    "less_leg_walking_rough",
]
REWARD = "Train/mean_reward"
LENGTH = "Train/mean_episode_length"
BASELINE = "scratch"
COMPLETE_FRACTION = 0.95

# -- style (same as CrazySim's scripts/skrl/plot_continual_metrics.py) ----------------------------------------------
# Method colors: CrazySim's validated categorical slots, so a method has the same color in both projects. The pairs
# residual/kae_moe and residual/FAME are only 6-8 CVD ΔE apart: every method also has its own marker.
METHODS = {
    # log folder: (label, color, marker)
    "scratch": ("scratch", "#898781", "o"),
    "residual": ("residual", "#1baf7a", "^"),
    "componet": ("componet", "#eda100", "v"),
    "moe": ("kae_moe", "#e87ba4", "P"),
    "FAME": ("FAME", "#008300", "X"),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# diverging: red (below the reference) - neutral gray - blue (above)
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def label(method: str) -> str:
    return METHODS[method][0]


def color(method: str) -> str:
    return METHODS[method][1]


def marker(method: str) -> str:
    return METHODS[method][2]


def ink_on(fill) -> str:
    """Primary ink or white, whichever reads on ``fill``."""
    r, g, b = to_rgb(fill)
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


def signed(value: float, digits: int = 2) -> str:
    """``+0.47`` / ``-1.23``; values that round to zero print as ``0.00`` (no ``-0.00``)."""
    if round(value, digits) == 0:
        return f"{0.0:.{digits}f}"
    return f"{value:+.{digits}f}"


def updates_formatter(value, _):
    return f"{value / 1000:.0f}K" if abs(value) >= 1000 else f"{value:.0f}"


def save(fig, output_dir: str, name: str):
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {os.path.join(output_dir, name)}.png/.pdf")


# -- data -----------------------------------------------------------------------------------------------------------
def discover(root: str, methods: list[str], seeds: list[int] | None) -> dict[str, dict[str, dict[int, str]]]:
    """``{task: {method: {seed: run directory}}}``; a run folder is ``<seed>`` or ``<seed>_<suffix>``."""
    runs = {}
    for task in TASKS:
        for method in methods:
            method_dir = os.path.join(root, task, method)
            if not os.path.isdir(method_dir):
                continue
            for name in sorted(os.listdir(method_dir)):
                run_dir = os.path.join(method_dir, name)
                leading = name.split("_", 1)[0]
                if not (os.path.isdir(run_dir) and leading.isdigit()):
                    continue
                if seeds and int(leading) not in seeds:
                    continue
                if not any(f.startswith("events.out.tfevents.") for f in os.listdir(run_dir)):
                    print(f"[WARN] No event file, skipped: {run_dir}")
                    continue
                runs.setdefault(task, {}).setdefault(method, {})[int(leading)] = run_dir
    return runs


def read_tags(run_dir: str, tags: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{tag: (steps, values)}`` over all event files of a run; later files override earlier ones (resumed runs).

    Reads the TFRecord files directly (8-byte length, 4-byte length CRC, data, 4-byte data CRC; CRCs not checked) and
    parses only the records that contain one of ``tags``: FAME logs ~20 tags at every step, which makes
    TensorBoard's ``EventAccumulator`` about 30x slower. A record cut off at the end (run still writing) is ignored.
    """
    keys = [tag.encode() for tag in tags]
    values = {tag: {} for tag in tags}
    for name in sorted(f for f in os.listdir(run_dir) if f.startswith("events.out.tfevents.")):
        with open(os.path.join(run_dir, name), "rb") as f:
            buffer = f.read()
        position = 0
        while position + 12 <= len(buffer):
            (length,) = struct.unpack_from("<Q", buffer, position)
            start, end = position + 12, position + 12 + length
            if end + 4 > len(buffer):
                break
            position = end + 4
            record = buffer[start:end]
            if not any(key in record for key in keys):
                continue
            event = event_pb2.Event.FromString(record)
            for value in event.summary.value:
                if value.tag in values:
                    scalar = (value.simple_value if value.WhichOneof("value") == "simple_value"
                              else float(tensor_util.make_ndarray(value.tensor)))
                    values[value.tag][event.step] = scalar
    out = {}
    for tag, series in values.items():
        if not series:
            raise ValueError(f"{run_dir} has no tag {tag}")
        steps = np.array(sorted(series))
        out[tag] = (steps, np.array([series[s] for s in steps]))
    return out


def load_task(task_runs: dict[str, dict[int, str]], task: str, points: int) -> dict:
    """Curves of every complete run of a task on a common step grid: ``{"steps", "methods": {m: {seed: {tag: y}}}}``."""
    raw = {m: {s: read_tags(d, (REWARD, LENGTH)) for s, d in seeds.items()} for m, seeds in task_runs.items()}
    longest = max(tags[REWARD][0][-1] for seeds in raw.values() for tags in seeds.values())
    for method, seeds in raw.items():
        for seed in [s for s, tags in seeds.items() if tags[REWARD][0][-1] < COMPLETE_FRACTION * longest]:
            last = seeds[seed][REWARD][0][-1]
            print(f"[WARN] {task}/{method}/{seed} is incomplete ({last} of {longest} updates), left out")
            del seeds[seed]
    raw = {m: seeds for m, seeds in raw.items() if seeds}
    runs = [tags for seeds in raw.values() for tags in seeds.values()]
    start = max(max(tags[tag][0][0] for tag in (REWARD, LENGTH)) for tags in runs)
    end = min(min(tags[tag][0][-1] for tag in (REWARD, LENGTH)) for tags in runs)
    grid = np.linspace(start, end, points)
    return {
        "steps": grid,
        "methods": {
            m: {s: {tag: np.interp(grid, *tags[tag]) for tag in (REWARD, LENGTH)} for s, tags in seeds.items()}
            for m, seeds in raw.items()
        },
    }


def forward_transfer(auc: float, baseline_auc: float) -> float:
    return (auc - baseline_auc) / (1.0 - baseline_auc) if baseline_auc < 1.0 else float("nan")


def task_metrics(data: dict, final_fraction: float) -> dict:
    """Per method, per seed: AUC of the normalized return, final return (raw and normalized), forward transfer."""
    curves = [run[REWARD] for seeds in data["methods"].values() for run in seeds.values()]
    lo, hi = float(np.min(curves)), float(np.max(curves))
    span = hi - lo if hi > lo else 1.0
    tail = max(1, int(round(final_fraction * len(data["steps"]))))
    per_seed = {}
    for method, seeds in data["methods"].items():
        per_seed[method] = {}
        for seed, run in seeds.items():
            normalized = (run[REWARD] - lo) / span
            per_seed[method][seed] = {
                "auc": float(normalized.mean()),
                "final_return": float(run[REWARD][-tail:].mean()),
                "final_normalized": float(normalized[-tail:].mean()),
            }
    if BASELINE in per_seed:
        baseline_auc = float(np.mean([v["auc"] for v in per_seed[BASELINE].values()]))
        for method, seeds in per_seed.items():
            for values in seeds.values():
                # the baseline's own FT is relative to its seed mean: its mean is 0 by definition
                values["ft"] = forward_transfer(values["auc"], baseline_auc)
    summary = {}
    for method, seeds in per_seed.items():
        summary[method] = {"seeds": sorted(seeds)}
        for key in next(iter(seeds.values())):
            values = np.array([v[key] for v in seeds.values()])
            summary[method][key] = {"mean": float(values.mean()), "std": float(values.std()),
                                    "per_seed": values.tolist()}
    return {"return_range": [lo, hi], "updates": float(data["steps"][-1]), "methods": summary}


# -- figures --------------------------------------------------------------------------------------------------------
def method_handles(methods: list[str], counts: dict[str, int] | None = None) -> list[Line2D]:
    def text(m):
        return f"{label(m)} (n={counts[m]})" if counts else label(m)
    return [Line2D([], [], color=color(m), marker=marker(m), markersize=6, markeredgecolor=SURFACE,
                   label=text(m)) for m in methods]


def plot_learning_curves(data: dict, tasks: list[str], methods: list[str], output_dir: str):
    rows = [(REWARD, "episode return (mean)"), (LENGTH, "episode length [steps]")]
    fig, axes = plt.subplots(len(rows), len(tasks), figsize=(3.1 * len(tasks), 2.6 * len(rows) + 0.8),
                             squeeze=False)
    counts = {}
    for column, task in enumerate(tasks):
        steps = data[task]["steps"]
        for method in methods:
            seeds = data[task]["methods"].get(method)
            if not seeds:
                continue
            counts[method] = max(counts.get(method, 0), len(seeds))
            for r, (tag, _) in enumerate(rows):
                stacked = np.stack([run[tag] for run in seeds.values()])
                mean, std = stacked.mean(axis=0), stacked.std(axis=0)
                ax = axes[r, column]
                if len(seeds) > 1:
                    ax.fill_between(steps, mean - std, mean + std, color=color(method), alpha=0.15, linewidth=0)
                ax.plot(steps, mean, color=color(method), marker=marker(method), markersize=5,
                        markevery=max(1, len(steps) // 8), markeredgecolor=SURFACE, markeredgewidth=0.8)
        axes[0, column].set_title(task, loc="left", fontsize=9)
        axes[-1, column].set_xlabel("PPO mini-batch updates")
    for r, (_, ylabel) in enumerate(rows):
        axes[r, 0].set_ylabel(ylabel)
    for ax in axes.flat:
        ax.grid(axis="y")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(updates_formatter))
    shown = [m for m in methods if m in counts]
    fig.legend(handles=method_handles(shown, counts), loc="upper center", ncol=len(shown), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Training curves (mean over seeds, band: ± 1 std; training episodes, stochastic policy)",
                 x=0.01, y=1.05, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir, "learning_curves")


def plot_final_return(metrics: dict, tasks: list[str], methods: list[str], output_dir: str, final_fraction: float):
    # episode return differs per task: one panel per task, each with its own x scale
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.1 * len(tasks), 0.42 * len(methods) + 1.4), squeeze=False)
    y = np.arange(len(methods))
    for ax, task in zip(axes[0], tasks):
        results = metrics[task]["methods"]
        high = max(max(results[m]["final_return"]["per_seed"]) for m in results)
        low = min(0.0, min(min(results[m]["final_return"]["per_seed"]) for m in results))
        span = max(high - low, 1e-9)
        for yi, method in zip(y, methods):
            if method not in results:
                ax.text(0, yi, "  no run", va="center", ha="left", color=MUTED, fontsize=8)
                continue
            final = results[method]["final_return"]
            ax.barh(yi, final["mean"], height=0.6, color=color(method), zorder=2)
            if len(final["per_seed"]) > 1:
                ax.errorbar(final["mean"], yi, xerr=final["std"], color=INK_SECONDARY, linewidth=1, capsize=2,
                            zorder=4)
            ax.scatter(final["per_seed"], np.full(len(final["per_seed"]), yi), s=9, color=SURFACE,
                       edgecolors=INK, linewidths=0.6, zorder=5)
            # label right of the bar, the seeds and the zero line (a negative bar ends left of zero)
            right = max(final["per_seed"] + [final["mean"] + final["std"], 0.0])
            ax.text(right + 0.03 * span, yi, f"{final['mean']:.1f}", va="center", ha="left", color=INK_SECONDARY,
                    fontsize=8)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=3)
        ax.set_xlim(low - (0.05 * span if low < 0 else 0.0), high + 0.25 * span)
        ax.set_title(task, loc="left", fontsize=9)
        ax.grid(axis="x", zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(y, [label(m) for m in methods] if ax is axes[0, 0] else [])
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.set_xlabel("episode return (mean)")
    fig.suptitle(f"Final episode return (last {final_fraction:.0%} of training; bar: seed mean, "
                 "line: ± 1 std, dots: seeds)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, output_dir, "final_return")


def plot_forward_transfer(metrics: dict, tasks: list[str], methods: list[str], output_dir: str):
    methods = [m for m in methods if m != BASELINE]  # FT of the baseline is 0 by definition
    data = np.array([[metrics[t]["methods"][m]["ft"]["mean"] if m in metrics[t]["methods"] else np.nan
                      for t in tasks] for m in methods])
    table = np.column_stack([data, np.nanmean(data, axis=1)])
    limit = max(np.nanmax(np.abs(table)), 1e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)  # equal arms: same color distance per unit
    fig, ax = plt.subplots(figsize=(1.3 * len(tasks) + 3.0, 0.45 * len(methods) + 2.0))
    image = ax.imshow(np.ma.masked_invalid(table), cmap=DIVERGING, norm=norm, aspect="auto")
    ax.axvline(len(tasks) - 0.5, color=SURFACE, linewidth=4)  # a surface gap before the mean column
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if not np.isfinite(table[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            ax.text(j, i, signed(table[i, j]), ha="center", va="center", fontsize=8,
                    color=ink_on(DIVERGING(norm(table[i, j]))), fontweight="bold" if j == len(tasks) else "normal")
    ax.set_xticks(range(len(tasks) + 1), tasks + ["mean"], rotation=30, ha="right")
    ax.set_yticks(range(len(methods)), [label(m) for m in methods])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    bar.set_label(f"forward transfer (0: as {BASELINE})")
    ax.set_title(f"Forward transfer of the normalized episode return vs {BASELINE} (seed mean)", loc="left")
    fig.tight_layout()
    save(fig, output_dir, "forward_transfer")


def plot_summary(metrics: dict, tasks: list[str], methods: list[str], output_dir: str):
    panels = [("auc", "AUC of normalized return"), ("ft", f"Forward transfer vs {BASELINE}"),
              ("final_normalized", "Normalized final return")]
    panels = [p for p in panels if all(p[0] in metrics[t]["methods"][m] for t in tasks for m in metrics[t]["methods"])]
    names = []
    for m in methods:
        done = sum(m in metrics[t]["methods"] for t in tasks)
        names.append(label(m) if done == len(tasks) else f"{label(m)} ({done}/{len(tasks)} tasks)")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.42 * len(methods) + 1.3), sharey=True,
                             squeeze=False)
    y = np.arange(len(methods))
    for ax, (key, title) in zip(axes[0], panels):
        values = np.array([np.mean([metrics[t]["methods"][m][key]["mean"] for t in tasks if m in metrics[t]["methods"]])
                           for m in methods])
        ax.barh(y, values, height=0.6, color=[color(m) for m in methods], zorder=2)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=3)
        span = max(np.abs(values).max(), 1e-9)
        ax.set_xlim(min(0.0, values.min()) - (0.15 * span if values.min() < 0 else 0), max(0.0, values.max()) + 0.3 * span)
        for yi, method, value in zip(y, methods, values):
            if key == "ft" and method == BASELINE:
                ax.text(0, yi, "  baseline", va="center", ha="left", color=MUTED, fontsize=8)
                continue
            ha = "left" if value >= 0 else "right"
            ax.text(value + (0.03 if value >= 0 else -0.03) * span, yi, signed(value) if key == "ft" else
                    f"{value:.2f}", va="center", ha=ha, color=INK_SECONDARY, fontsize=8)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0, 0].set_yticks(y, names)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Continual learning: mean over the tasks (seed mean per task)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, output_dir, "summary")


# -- tables ---------------------------------------------------------------------------------------------------------
def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def print_tables(metrics: dict, tasks: list[str], methods: list[str]):
    def cell(task, method, key, fmt):
        result = metrics[task]["methods"].get(method)
        if result is None or key not in result:
            return "-"
        value = result[key]
        text = fmt(value["mean"])
        return text + (f" ± {value['std']:.2f}" if len(value["per_seed"]) > 1 else "") + f" (n={len(value['per_seed'])})"

    for key, title, fmt in (("final_return", "Final episode return", lambda v: f"{v:.2f}"),
                            ("auc", "AUC of the normalized episode return", lambda v: f"{v:.3f}"),
                            ("ft", f"Forward transfer vs {BASELINE}", signed)):
        rows = [[label(m)] + [cell(t, m, key, fmt) for t in tasks] for m in methods]
        print(f"\n{title} (seed mean ± std)\n" + markdown_table(["method"] + tasks, rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="Experiment log root (default: scripts/common/log_root.py).")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                        help="Method folders to compare.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to use (default: all found).")
    parser.add_argument("--points", type=int, default=200, help="Grid points per task for interpolation.")
    parser.add_argument("--final_fraction", type=float, default=0.05,
                        help="Fraction of the training at its end that gives the final return.")
    parser.add_argument("--output_dir", default=None, help="Directory for the figures (default: <root>/figures).")
    parser.add_argument("--output_json", default=None, help="JSON file for all metrics.")
    args = parser.parse_args()

    root = args.root or crl_log_root()
    output_dir = args.output_dir or os.path.join(root, "figures")
    os.makedirs(output_dir, exist_ok=True)

    runs = discover(root, args.methods, args.seeds)
    if not runs:
        raise SystemExit(f"No runs found under {root}/<task>/<method>/<seed>")
    tasks = [t for t in TASKS if t in runs]
    methods = [m for m in args.methods if any(m in runs[t] for t in tasks)]
    print(f"[INFO] Root {root}: tasks {tasks}")
    for method in methods:
        print(f"[INFO]   {method}: " + ", ".join(f"{t} {sorted(runs[t].get(method, {}))}" for t in tasks))

    data = {task: load_task(runs[task], task, args.points) for task in tasks}
    metrics = {task: task_metrics(data[task], args.final_fraction) for task in tasks}

    plot_learning_curves(data, tasks, methods, output_dir)
    plot_final_return(metrics, tasks, methods, output_dir, args.final_fraction)
    if all(BASELINE in metrics[t]["methods"] for t in tasks):
        plot_forward_transfer(metrics, tasks, methods, output_dir)
    else:
        print(f"[WARN] No complete {BASELINE} run on every task: forward transfer skipped")
    plot_summary(metrics, tasks, methods, output_dir)
    print_tables(metrics, tasks, methods)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"root": root, "tasks": tasks, "final_fraction": args.final_fraction, "metrics": metrics},
                      f, indent=2)
        print(f"\n[INFO] Metrics written to {args.output_json}")


if __name__ == "__main__":
    main()
