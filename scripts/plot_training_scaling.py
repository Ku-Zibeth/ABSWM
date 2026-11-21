"""Render the S1 and S2 validation plots from an existing scaling run."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, PercentFormatter


def read(output, name):
    with (output/f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def learning_plot(output, curves):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    rows = defaultdict(list)
    for r in curves:
        if r["method"].startswith("A"):
            rows[r["seed"]].append(r)
    for seed, epochs in sorted(rows.items()):
        original = {int(float(r["epoch"])): r for r in epochs if r["method"] == "A0"}
        resume_epoch = min(original, key=lambda i: float(original[i]["val_score"]))
        continuation = {float(r["epoch"])+resume_epoch: r for r in epochs if r["method"] != "A0"}
        for ax, key, title in zip(axes, ("train_total", "val_total"), ("Training", "Validation")):
            xs = sorted(original)
            color = ax.plot(xs, [float(original[i][key]) for i in xs], "--", label=f"Seed {seed}")[0].get_color()
            xs = [resume_epoch, *sorted(continuation)]
            ys = [float(original[resume_epoch][key]), *[float(continuation[i][key]) for i in xs[1:]]]
            ax.plot(xs, ys, color=color)
            ax.scatter(xs[0], ys[0], s=19, color=color, zorder=3)
            ax.set(xlabel="Joint-training epoch", ylabel="Curve MSE", title=title)
            ax.grid(alpha=.22)
    axes[0].legend(title="Dashed = archived fit", frameon=False)
    fig.savefig(output/"epoch_loss.png", dpi=170)
    plt.close(fig)


def scale_plot(output, window, training):
    fig, plots = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    keys = ("action_wheel_rmse", "force_rmse", "action_regret_ns", "ranking_accuracy", "high_fnr")
    titles = ("Hold-relative wheel RMSE (rad/s)", "Tire force RMSE (N)", "Impulse regret (N·s)",
              "Pairwise ranking accuracy", "High-slip false negative rate")
    train_time = defaultdict(list)
    for row in training:
        train_time[row["method"]].append(float(row["training_s"]))
    for arm, color, name in [("same_epochs", "#0e627a", "Same epochs"), ("same_updates", "#b4602a", "Same updates")]:
        rows = sorted((r for r in window if r["method"].endswith(arm) and r["seed"] == "pooled"),
                      key=lambda r: int(r["train_windows"]))
        windows = [int(r["train_windows"]) for r in rows]
        for ax, key, title in zip(plots.flat, keys, titles):
            values = np.array([float(r[key]) if r[key] else np.nan for r in rows])
            ax.plot(windows, values, "o-", color=color, label=name)
            ax.set(xscale="log", xlabel="Training windows", title=title, xticks=windows,
                   xticklabels=[f"{x:,}" for x in windows])
            ax.xaxis.set_minor_formatter(NullFormatter())
            if key in {"ranking_accuracy", "high_fnr"}:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
            ax.grid(alpha=.22)
        ax = plots.flat[-1]
        xs = [np.mean(train_time[r["method"]]) for r in rows]
        ax.plot(xs, [float(r["action_regret_ns"]) for r in rows], "o-", color=color, label=name)
        if arm == "same_epochs":
            for x, r in zip(xs, rows):
                ax.annotate(r["method"].split("_")[0], (x, float(r["action_regret_ns"])),
                            xytext=(4, 4), textcoords="offset points", fontsize=8, color=color)
        ax.set(xscale="log", xlabel="Training seconds per seed", ylabel="Impulse regret (N·s)",
               title="Validation benefit vs compute")
        ax.margins(x=.10)
        ax.grid(alpha=.22)
    plots.flat[0].legend(frameon=False)
    fig.savefig(output/"window_scaling.png", dpi=170)
    plt.close(fig)


def updates_plot(output, epoch, window, training):
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.8), layout="constrained")
    updates = {(r["method"], r["seed"]): int(r["updates"]) for r in training}
    for label, chosen in (("S1", epoch), ("Same epochs", [r for r in window if r["method"].endswith("same_epochs")]),
                          ("Same updates", [r for r in window if r["method"].endswith("same_updates")])):
        points = [r for r in chosen if r["seed"] == "pooled"]
        for ax, key, title in zip(axes, ("action_wheel_rmse", "action_regret_ns"),
                                  ("Hold-relative wheel RMSE (rad/s)", "Impulse regret (N·s)")):
            x = [np.median([v for (name, _), v in updates.items() if name == r["method"]]) for r in points]
            y = [float(r[key]) for r in points]
            ax.plot(x, y, "o-", label=label)
            for a, b, r in zip(x, y, points):
                if label != "Same updates" and not r["method"].startswith("B0"):
                    ax.annotate(r["method"].split("_")[0], (a, b), xytext=(4, 4),
                                textcoords="offset points", fontsize=8)
            ax.set(xscale="symlog", xlabel="Additional optimizer updates", ylabel=title)
            ax.grid(alpha=.22)
    axes[0].legend(frameon=False)
    fig.savefig(output/"updates_vs_error.png", dpi=170)
    plt.close(fig)


def followup_plots(folder):
    roads = [r for r in read(folder, "road_scaling") if r["seed"] == "pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
    numbers = [int(r["roads"]) for r in roads]
    metrics = (("action_wheel_rmse", "Action response RMSE (rad/s)"),
               ("force_rmse", "Tire force RMSE (N)"), ("action_regret_ns", "Impulse regret (N·s)"))
    for ax, (key, title) in zip(axes, metrics):
        ax.plot(numbers, [float(r[key]) for r in roads], "o-", color="#0e627a")
        ax.set(xscale="log", xticks=numbers, xticklabels=[f"{n:,}" for n in numbers],
               xlabel="Training roads", ylabel=title)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.22)
    fig.savefig(folder/"road_scaling.png", dpi=170)
    plt.close(fig)

    pairs = [r for r in read(folder, "paired_statistics") if r["domain"] == "closed" and
             r["metric"] == "distance_m" and r["baseline"] == "A0" and not (r["kind"] or r["family"])]
    closed = [r for r in read(folder, "closed_loop_summary") if r["surface_type"] and not (r["kind"] or r["family"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    markers = dict(zip(("A0", "S1", "S2", "C3", "NLS-MPC", "Oracle-MPC"), ("o", "s", "D", "^", "P", "X")))
    for kind, offset, color in (("constant", -.12, "#0e627a"), ("dynamic", .12, "#b4602a")):
        subset = [r for r in pairs if r["surface_type"] == kind]
        means = np.array([float(r["mean_delta"]) for r in subset])
        errors = [means-[float(r["lo95"]) for r in subset], [float(r["hi95"]) for r in subset]-means]
        axes[0].errorbar(means, np.arange(len(subset))+offset, xerr=errors, fmt="o", color=color, label=kind.title())
        axes[0].set(yticks=np.arange(len(subset)), yticklabels=[r["method"] for r in subset])
        for r in closed:
            if r["surface_type"] == kind:
                x, y = float(r["distance_m"]), 1000*float(r["lock_s"])
                axes[1].scatter(x, y, color=color, marker=markers[r["method"]], s=36)
    axes[0].axvline(0, color=".5", linewidth=1)
    axes[0].legend(frameon=False)
    axes[0].set(xlabel="Paired distance difference vs A0 (m)", title="Road bootstrap 95% intervals")
    axes[1].set(xlabel="Mean braking distance (m)", ylabel="Mean lock time (ms)", title="Distance and lock risk")
    axes[1].margins(x=.12)
    handles = [plt.Line2D([], [], linestyle="", color=".25", marker=marker, label=name) for name, marker in markers.items()]
    axes[1].legend(handles=handles, loc="center", bbox_to_anchor=(.60, .45), ncols=2, frameon=False)
    for ax in axes:
        ax.grid(alpha=.22)
    fig.savefig(folder/"final_braking.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/training_scaling"))
    output = parser.parse_args().output
    epoch, window, training, curves = [read(output, name) for name in
                                      ("epoch_scaling", "window_scaling", "training_summary", "learning_curves")]
    learning_plot(output, curves)
    scale_plot(output, window, training)
    updates_plot(output, epoch, window, training)
    if (output/"followup"/"s4_status.json").exists():
        followup_plots(output/"followup")


if __name__ == "__main__":
    main()
