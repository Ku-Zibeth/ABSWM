"""Generate diagnostic figures from persisted 3.1 result tables."""
from __future__ import annotations

import ast
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .active_metrics import grouped
from .adaptive_analysis import read_rows


def make_figures(output):
    root = output/"figures"
    root.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": .2, "font.size": 9})
    blue, orange, neutral = "#28658c", "#b65e42", "#6d7d8a"

    def read(name):
        path = output/(name+".csv")
        return read_rows(path) if path.exists() else []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(root/(name+".png"), dpi=145, bbox_inches="tight")
        plt.close(fig)

    def amount(rows, key):
        return np.asarray([float(r[key]) if r.get(key) else np.nan for r in rows])

    def line(ax, rows, x, y, color, label=None):
        ax.plot(amount(rows, x), amount(rows, y), color=color, lw=1., label=label)

    audit = read("p0_protection_counterfactual")
    if audit:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.2, 3.3))
        horizons = sorted({int(r["horizon_ms"]) for r in audit})
        groups = [[float(r["actual_utility_delta"]) for r in audit if int(r["horizon_ms"]) == k] for k in horizons]
        ax.boxplot(groups, positions=np.arange(len(groups)), showfliers=False)
        ax.set(xticks=np.arange(len(groups)), xticklabels=horizons,
               xlabel="Horizon (ms)", ylabel="Truth utility: protected minus requested")
        reasons = Counter(r["reason"] for r in audit if r["horizon_ms"] == str(horizons[0]))
        bx.barh(list(reasons), list(reasons.values()), color=blue)
        bx.set_xlabel("Protection events")
        save(fig, "protection_attribution")

    paths = sorted((output/"traces").glob("*.csv"))

    def traces(names, scenario=None):
        found = {}
        for path in paths:
            if any(f"_{m}_" in path.stem for m in names):
                for name in names:
                    if f"_{name}_" in path.stem:
                        scene = path.stem.split("_")[0]
                        if scenario is None or scene == scenario:
                            found.setdefault((scene, path.stem.rsplit("_", 2)[1]), {}).setdefault(name, read_rows(path))
        return found

    comparison = traces(("old_peak", "B4"))
    complete = next((r for r in comparison.values() if len(r) == 2), None)
    if complete:
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.4))
        for label, pack in complete.items():
            color = blue if label == "B4" else orange
            for ax, key, title in zip(axes.flat, ("v", "slip", "force_N", "applied"),
                                      ("Speed (m/s)", "Slip", "Force (N)", "Torque (Nm)")):
                line(ax, pack, "time_s", key, color, label)
                ax.set_ylabel(title)
        axes[0, 0].legend()
        fig.supxlabel("Time (s)")
        save(fig, "traditional_peak_seeking")

    probability = [r for r in read("probability_calibration") if r["nominal"] == "0.9" and r["variable"] == "force_n"
                   and r["method"] in {"prototype", "continuous", "ensemble"} and r["kind"] == "constant"]
    if probability:
        fig, ax = plt.subplots(figsize=(6.1, 3.9))
        for method, color in zip(("prototype", "continuous", "ensemble"), (blue, orange, neutral)):
            rows = [r for r in probability if r["method"] == method]
            ax.scatter(amount(rows, "width"), amount(rows, "coverage"), label=method, color=color, alpha=.65)
        ax.axhline(.9, color="0.55", lw=.8)
        ax.set(xlabel="Nominal 90% force interval width (N)", ylabel="Observed coverage")
        ax.legend()
        save(fig, "predictive_distribution_comparison")

    beliefs = read("belief_revision") or read("p3_belief_revision")
    belief_groups = grouped([r for r in beliefs if r["method"] == "B13"], ("road_id", "initial_speed", "seed"))
    if belief_groups:
        rows = max(belief_groups.values(), key=lambda group: (sum(bool(r["revision"]) for r in group), len(group)))
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.2), sharex=True)
        for ax, series in zip(axes, (("observed_mu", "predictive_mu_mean"), ("innovation", "cusum"), ("entropy_after", "ess"))):
            for field, color in zip(series, (blue, orange)):
                line(ax, rows, "time_s", field, color, field.replace("_", " "))
            for step in rows:
                if step["revision"]:
                    ax.axvline(float(step["time_s"]), color="0.65", alpha=.5, lw=.7)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time (s)")
        save(fig, "innovation_and_belief_revision")

    refinement = read("latent_refinement") or read("p3_latent_refinement")
    if refinement:
        fig, ax = plt.subplots(figsize=(5.9, 3.7))
        before, after = amount(refinement, "refine_before"), amount(refinement, "refine_after")
        finite = np.isfinite(before) & np.isfinite(after)
        if finite.any():
            ax.scatter(before[finite], after[finite], color=blue, alpha=.4, s=9)
            max_error = np.max(before[finite])
            ax.plot([0, max_error], [0, max_error], color=neutral, lw=.8)
        ax.set(xlabel="Recent force-proxy residual before correction", ylabel="Residual after correction")
        save(fig, "information_refinement")

    durations = read("exploration_duration")
    if durations:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.3))
        for name, color in (("B10", neutral), ("B13", blue), ("B14", orange)):
            groups = grouped([r for r in durations if r["method"] == name], ("commit_steps",))
            steps = sorted(groups, key=lambda k: int(k[0]))
            x = [5*int(k[0]) for k in steps]
            ax.plot(x, [np.mean(amount(groups[k], "exploration_count")) for k in steps], ".-", label=name, color=color)
            bx.plot(x, [np.mean(amount(groups[k], "distance_m")) for k in steps], ".-", label=name, color=color)
        ax.set(xlabel="Commitment (ms)", ylabel="Executed exploration segments")
        bx.set(xlabel="Commitment (ms)", ylabel="Observed distance (m)")
        ax.legend()
        save(fig, "commitment_duration")

    branches = read("observation_branches")
    if branches:
        largest = max(branches, key=lambda r: float(r["realization_value"]))
        keys = ("road_id", "seed", "anchor", "length", "first")
        rows = [r for r in branches if all(r[k] == largest[k] for k in keys)]
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.2, 3.4))
        points = ax.scatter(amount(rows, "velocity"), amount(rows, "wheel"), s=30+350*amount(rows, "mass"),
                            c=amount(rows, "learned_next"), cmap="viridis")
        fig.colorbar(points, ax=ax, label="Continuation action index")
        ax.set(xlabel="Observable speed (m/s)", ylabel="Observable wheel speed (rad/s)")
        stack = np.asarray([ast.literal_eval(r["posterior"]) for r in rows])
        bottom = np.zeros(len(rows))
        for values in stack.T:
            bx.bar(range(len(rows)), values, bottom=bottom)
            bottom += values
        bx.set(xlabel="Observable branch", ylabel="Posterior mass")
        save(fig, "observation_branches")

    realized = read("realized_voi")
    factorial = read("exploration_factorial")
    if realized and factorial:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.6))
        for mode, color in (("normal", neutral), ("active", blue)):
            rows = [r for r in realized if r["realized_first"] == mode]
            ax.scatter(amount(rows, "predicted_voi"), amount(rows, "actual_interval_gain_100_ns"), color=color, label=mode)
        ax.set(xlabel="Expected normalized control value", ylabel="Realized 100 ms impulse gain (N·s)")
        ax.legend()
        labels, data = [], []
        for (first, informed), rows in grouped(factorial, ("first_action", "real_evidence")).items():
            labels.append(first+"\n"+("updated" if informed == "True" else "frozen"))
            data.append(amount(rows, "after_200_impulse_ns"))
        bx.boxplot(data, tick_labels=labels, showfliers=False)
        bx.set_ylabel("200 ms continuation impulse (N·s)")
        save(fig, "feedback_value_and_factorial")

    dynamics = read("dynamic_road_by_case")
    for kind in ("high_low", "low_high", "patch", "shape"):
        candidates = [r for r in dynamics if r["kind"] == kind and r["method"] == "B13"]
        if not candidates:
            continue
        series = traces(("B10", "B13", "B14"), candidates[0]["road_id"])
        group = max(series.values(), key=len, default={})
        if not group:
            continue
        fig, axes = plt.subplots(3, 2, figsize=(9., 7.1), sharex=True)
        for ax, field, label in zip(axes.flat, ("v", "slip", "force_N", "applied", "entropy_after", "innovation"),
                                     ("Speed (m/s)", "Slip", "Force (N)", "Torque (Nm)", "Entropy (nats)", "Innovation")):
            for method, color in (("B10", neutral), ("B13", blue), ("B14", orange)):
                if method in group:
                    trace = group[method]
                    line(ax, trace, "time_s", field, color, method)
                    if method == "B13":
                        for condition, color in (("safety", "#b72d43"), ("actual_exploration", "#00875c")):
                            marked = [r for r in trace if (r[condition] == "True" if condition == "actual_exploration" else bool(r[condition]))]
                            ax.scatter(amount(marked, "time_s"), amount(marked, field), color=color, s=4)
            ax.set_ylabel(label)
        axes[0, 0].legend()
        fig.supxlabel("Time (s)")
        save(fig, "trajectory_"+kind)

    recovery = [r for r in read("force_recovery") if r["method"] in {"B7", "B10", "B13", "B14", "fixed_bank"}]
    if recovery:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.5))
        labels, delays, censored = [], [], []
        for (name,), rows in grouped(recovery, ("method",)).items():
            labels.append(name)
            delays.append([float(r["force_recovery_s"]) if r.get("force_recovery_s") else min(.5, float(r["followup_s"]))
                           for r in rows])
            censored.append(np.mean([r["force_recovery_censored"] == "True" for r in rows]))
        ax.boxplot(delays, tick_labels=labels, showfliers=False)
        ax.set_ylabel("Recovery or censored follow-up (s)")
        bx.bar(labels, censored, color=orange)
        bx.set_ylabel("Censored fraction")
        save(fig, "dynamic_force_recovery")

    rankings = read("action_regret")
    if rankings:
        groups = grouped([r for r in rankings if r["kind"] == "constant"], ("method",))
        fig, ax = plt.subplots(figsize=(8.5, 3.7))
        ax.bar([n[0] for n in groups], [np.mean(amount(r, "pure_regret_ns")) for r in groups.values()], color=blue)
        ax.tick_params(axis="x", rotation=58)
        ax.set_ylabel("Mean braking impulse regret (N·s)")
        save(fig, "action_regret_comparison")

    calibration = [r for r in read("risk_calibration") if r["event"] == "high"]
    if calibration:
        fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.5), sharex=True, sharey=True)
        for ax, family in zip(axs, ("burckhardt", "magic", "rational")):
            ax.plot([0, 1], [0, 1], color=neutral, lw=.7)
            for method, color in (("adaptive_raw", neutral), ("adaptive_calibrated", blue), ("e4_raw", orange)):
                rows = sorted([r for r in calibration if r["suite"] == family and r["method"] == method],
                              key=lambda r: float(r["p_mean"]))
                if rows:
                    ax.plot(amount(rows, "p_mean"), amount(rows, "observed_rate"), ".-", color=color, label=method)
            ax.set_title(family)
        if axs[0].get_legend_handles_labels()[0]:
            axs[0].legend(fontsize=7)
        fig.supxlabel("Predicted high-slip probability")
        fig.supylabel("Observed event fraction")
        save(fig, "high_slip_calibration")

    closed = read("integrated_summary")
    if closed:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 3.9))
        for panel, kind in ((ax, "constant"), (bx, "high_low")):
            subset = [r for r in closed if r["kind"] == kind and r["method"] in
                      {"B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B12", "B13", "B14"}]
            for r in subset:
                x, y = float(r["high_slip_s"]), float(r["distance_m"])
                panel.scatter(x, y, color=blue, s=15)
                panel.annotate(r["method"], (x, y), xytext=(3, 2), textcoords="offset points", fontsize=7)
            panel.set(xlabel="Mean high-slip exposure (s)", ylabel="Mean observed distance (m)", title=kind.replace("_", " "))
        save(fig, "distance_risk_comparison")

    ablation = [r for r in read("paired_statistics") if r["metric"] == "distance_m"
                and r["method"] in {"B10", "B12", "B13", "forget", "regenerate", "refine_selected"}]
    if ablation:
        fig, ax = plt.subplots(figsize=(7.5, max(3.5, .28*len(ablation))))
        mean, lo, hi = (amount(ablation, k) for k in ("mean_delta", "lo95", "hi95"))
        checks = [f"{r['method']} − {r['baseline']} ({r['partition']})" for r in ablation]
        ax.errorbar(mean, checks, xerr=np.maximum(0, [mean-lo, hi-mean]), fmt="o", capsize=3, color=blue)
        ax.axvline(0., color="0.5", lw=.7)
        ax.set_xlabel("Distance difference (m), road-cluster 95% interval")
        save(fig, "control_ablations")
