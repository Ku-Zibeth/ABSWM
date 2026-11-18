from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import fingerprint, write_json
from .controllers import LeastSquaresPeak, NeuralPeak, simulate_closed, tune_controller
from .data import load_split, make_history, windows
from .models import load_model, seed_everything
from .physics import TYPICAL_ROADS, advance, friction, slip
from .training import KINDS


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty results: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curve_metrics(predicted, target, grid, coverage):
    observed = (grid >= coverage[0]) & (grid <= coverage[1])
    unseen = ~observed
    return {"curve_rmse": float(np.sqrt(np.mean((predicted-target)**2))),
            "observed_curve_rmse": float(np.sqrt(np.mean((predicted[observed]-target[observed])**2))) if observed.any() else None,
            "unobserved_curve_rmse": float(np.sqrt(np.mean((predicted[unseen]-target[unseen])**2))) if unseen.any() else None}


def state_metrics(predicted, target, c):
    error = predicted-target
    return {"one_step_v_rmse": float(np.sqrt(np.mean(error[:, 0, 0]**2))),
            "one_step_omega_rmse": float(np.sqrt(np.mean(error[:, 0, 1]**2))),
            "rollout_v_rmse": float(np.sqrt(np.mean(error[..., 0]**2))),
            "rollout_omega_rmse": float(np.sqrt(np.mean(error[..., 1]**2))),
            "rollout_slip_rmse": float(np.sqrt(np.mean((slip(predicted, c["radius"])-slip(target, c["radius"]))**2)))}


def parametric_rollout(history, actions, theta, c):
    x = history[-1, :2].astype(float)
    trajectory = []
    for u in actions:
        x, _ = advance(x, u, theta, c)
        trajectory.append(x)
    return np.asarray(trajectory)


def load_models(c, output):
    models = {}
    for kind in KINDS:
        for seed in c["seeds"]:
            model, checkpoint = load_model(output/"models"/f"{kind}_{seed}.pt", c["device"])
            if checkpoint["config_hash"] != fingerprint(c) or checkpoint["kind"] != kind or checkpoint["seed"] != seed:
                raise ValueError("Checkpoint configuration or identity mismatch")
            if checkpoint.get("completed_epochs") != c["epochs"]:
                raise ValueError("Training is incomplete; run train before evaluate")
            models[(kind, seed)] = model
    return models


def evaluate_open(c, output, manifest, models):
    data = load_split(output/"data", "test")
    samples = windows(data, c, c["test_windows"], c["data_seed"]+30)
    grid = np.linspace(0, 1, 1001)
    theta = np.asarray([manifest["roads"]["test"][int(data["road_index"][i])]["theta"]
                        for i, _ in samples["locators"]])
    true_curves = friction(grid[None], theta[:, None])
    rows = []
    for method in ("fixed", "nls", *KINDS):
        for seed in c["seeds"] if method in KINDS else [-1]:
            model = models.get((method, seed))
            predictions, curves, peaks = [], [], []
            if model is not None:
                with torch.inference_mode():
                    for start in range(0, len(theta), c["batch_size"]):
                        h = torch.as_tensor(samples["history"][start:start+c["batch_size"]], device=c["device"])
                        a = torch.as_tensor(samples["actions"][start:start+c["batch_size"]], device=c["device"])
                        peaks.extend(model.peak(h).cpu().numpy())
                        if method != "direct":
                            predictions.extend(model.rollout(h, a).cpu().numpy())
                            z = model.encode(h)
                            gt = torch.as_tensor(grid, dtype=torch.float32, device=c["device"])
                            values = model.friction(gt.expand(len(z), -1).reshape(-1),
                                      z[:, None].expand(-1, len(grid), -1).reshape(-1, z.shape[-1]))
                            curves.extend(values.reshape(len(z), -1).cpu().numpy())
            else:
                for h, a in zip(samples["history"], samples["actions"]):
                    estimator = LeastSquaresPeak(c)
                    peak = estimator(h) if method == "nls" else c["fixed_reference"]
                    fitted = estimator.theta if estimator.theta is not None else TYPICAL_ROADS["wet"]
                    peaks.append(peak)
                    curves.append(friction(grid, fitted))
                    predictions.append(parametric_rollout(h, a, fitted, c))
            metrics = state_metrics(np.asarray(predictions), samples["states"][:, 1:], c) if predictions else {}
            peak_errors = np.abs(np.asarray(peaks)-samples["peak"])
            clipped = np.clip(peaks, c["reference_min"], c["reference_max"])
            row = {"method": method, "seed": seed, "windows": len(theta),
                   "peak_mae": float(np.mean(peak_errors)),
                   "clipped_peak_mae": float(np.mean(np.abs(clipped-samples["peak"]))),
                   "out_of_reference_range_fraction": float(np.mean(np.asarray(peaks) != clipped)),
                   **{k: metrics.get(k) for k in ("one_step_v_rmse", "one_step_omega_rmse",
                     "rollout_v_rmse", "rollout_omega_rmse", "rollout_slip_rmse")}}
            curve_rows = [curve_metrics(p, t, grid, cov) for p, t, cov in zip(curves, true_curves, samples["coverage"])]
            for key in ("curve_rmse", "observed_curve_rmse", "unobserved_curve_rmse"):
                values = [r[key] for r in curve_rows if r[key] is not None]
                row[key] = float(np.mean(values)) if values else None
            low = samples["coverage"][:, 1] < 0.05
            row["low_excitation_windows"] = int(low.sum())
            row["low_excitation_peak_mae"] = float(np.mean(peak_errors[low])) if low.any() else None
            rows.append(row)
    write_csv(output/"open_loop.csv", rows)
    np.savez_compressed(output/"open_loop_windows.npz", locators=samples["locators"],
                        coverage=samples["coverage"], peak=samples["peak"])
    return rows, data


def history_study(c, output, manifest, models, data):
    rows = []
    for road_index, road in enumerate(manifest["roads"]["test"]):
        candidates = np.flatnonzero((data["road_index"] == road_index)
                                    & (data["initial_speed"] == max(c["initial_speeds"]))
                                    & (data["kind"] == c["excitation_kinds"][0]))
        i = int(candidates[0])
        length = int(data["lengths"][i])
        observed_slip = slip(data["states"][i, :length], c["radius"])
        for fraction in (0.25, 0.5, 0.9, 1.2):
            hits = np.flatnonzero(observed_slip >= fraction*road["peak"])
            # For pre-peak levels use the sample immediately before crossing.
            t = max(0, int(hits[0])-(fraction < 1)) if len(hits) else length-1
            lengths = sorted({max(2, round(c["history_steps"]*f)) for f in (0.2, 0.5, 1.0)})
            for history_length in lengths:
                h = make_history(data["states"][i], data["actions"][i], t, history_length, c)
                for method in ("fixed", "nls", *KINDS):
                    for seed in c["seeds"] if method in KINDS else [-1]:
                        if method == "fixed":
                            estimate = c["fixed_reference"]
                        elif method == "nls":
                            estimate = LeastSquaresPeak(c)(h)
                        else:
                            estimate = NeuralPeak(models[(method, seed)], c["device"])(h)
                        rows.append({"road_id": road["id"], "method": method, "seed": seed,
                                     "history_steps": history_length, "real_observations": min(t+1, history_length),
                                     "coverage_target_fraction": fraction, "reached": bool(len(hits)),
                                     "time_s": t*c["control_dt"], "history_lambda_max": float(h[:, 2].max()),
                                     "true_peak": road["peak"], "estimate": estimate,
                                     "absolute_error": abs(estimate-road["peak"])})
    write_csv(output/"history_study.csv", rows)
    return rows


def action_study(c, output, manifest, models, data):
    """At the same observed state, compare three counterfactual action sequences."""
    rows, diagnostic = [], None
    for road_index, road in enumerate(manifest["roads"]["test"]):
        i = int(np.flatnonzero((data["road_index"] == road_index)
                               & (data["initial_speed"] == max(c["initial_speeds"])))[0])
        observed = slip(data["states"][i, :data["lengths"][i]], c["radius"])
        hits = np.flatnonzero(observed >= 0.5*road["peak"])
        t = int(hits[0]) if len(hits) else min(c["history_steps"], len(observed)-1)
        h = make_history(data["states"][i], data["actions"][i], t, c["history_steps"], c)
        u0 = float(h[-1, 3])
        for name, target in (("release", 0.0), ("hold", u0), ("increase", min(c["torque_max"], u0+600))):
            limit = c["torque_rate"]*c["control_dt"]*np.arange(1, c["horizon_steps"]+1)
            actions = (u0+np.clip(target-u0, -limit, limit)).astype(np.float32)
            truth = parametric_rollout(h, actions, road["theta"], c)
            if diagnostic is None:
                diagnostic = {"history": h, "branches": {}}
            for kind in ("world", "current"):
                for seed in c["seeds"]:
                    with torch.inference_mode():
                        predicted = models[(kind, seed)].rollout(
                            torch.as_tensor(h[None], device=c["device"]),
                            torch.as_tensor(actions[None], device=c["device"])).cpu().numpy()[0]
                    rows.append({"road_id": road["id"], "method": kind, "seed": seed,
                                 "branch": name, **state_metrics(predicted[None], truth[None], c)})
                    if road_index == 0 and kind == "world" and seed == c["seeds"][0]:
                        diagnostic["branches"][name] = {"truth": truth, "predicted": predicted, "actions": actions}
    write_csv(output/"action_study.csv", rows)
    return rows, diagnostic


def evaluate_closed(c, output, manifest, models, gains):
    rows, example = [], {}
    trace_dir = output/"traces"
    trace_dir.mkdir(exist_ok=True)
    for road_index, road in enumerate(manifest["roads"]["test"]):
        for speed in c["initial_speeds"]:
            for method in ("fixed", "oracle", "nls", *KINDS):
                for seed in c["seeds"] if method in KINDS else [-1]:
                    selector = (LeastSquaresPeak(c) if method == "nls" else
                                NeuralPeak(models[(method, seed)], c["device"]) if method in KINDS else None)
                    result, trace = simulate_closed(road["theta"], speed, c, gains, selector, oracle=method == "oracle")
                    rows.append({"road_id": road["id"], "family": road["family"], "initial_speed": speed,
                                 "method": method, "seed": seed, **result})
                    # Keep every trace for audit; illustrative plot uses a single paired scenario.
                    np.savez_compressed(trace_dir/f"{road['id']}_{speed:g}_{method}_{seed}.npz", **trace)
                    if road_index == 0 and speed == max(c["initial_speeds"]) and seed in (-1, c["seeds"][0]):
                        example[method] = trace
        print(f"closed loop: road {road_index+1}/{len(manifest['roads']['test'])}", flush=True)
    write_csv(output/"closed_loop.csv", rows)
    return rows, example


def paired_comparisons(c, rows):
    lookup = {(r["road_id"], r["initial_speed"], r["method"], r["seed"]): r for r in rows}
    summaries = []
    road_ids = sorted({r["road_id"] for r in rows})
    for baseline in ("fixed", "nls", "direct", "oracle"):
        road_deltas = []
        for identifier in road_ids:
            deltas = []
            for speed in c["initial_speeds"]:
                for seed in c["seeds"]:
                    world = lookup[(identifier, speed, "world", seed)]
                    other = lookup[(identifier, speed, baseline, seed if baseline == "direct" else -1)]
                    deltas.append(world["distance_m"]-other["distance_m"])
            road_deltas.append(np.mean(deltas))
        rng = np.random.default_rng(c["data_seed"]+40)
        boots = np.mean(rng.choice(road_deltas, (2000, len(road_deltas)), replace=True), axis=1)
        summaries.append({"comparison": f"world minus {baseline}", "road_clusters": len(road_deltas),
                          "mean_distance_delta_m": float(np.mean(road_deltas)),
                          "bootstrap_95_low_m": float(np.percentile(boots, 2.5)),
                          "bootstrap_95_high_m": float(np.percentile(boots, 97.5)),
                          "road_deltas_m": [float(x) for x in road_deltas]})
    return summaries


def plots(c, output, open_rows, closed_rows, history_rows, example, diagnostic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    folder = output/"figures"
    folder.mkdir(exist_ok=True)
    methods = ("fixed", "oracle", "nls", *KINDS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for method, trace in example.items():
        axes[0, 0].plot(trace["time"], trace["states"][:, 0], label=method)
        axes[0, 1].plot(trace["time"], trace["slip"], label=method)
        axes[1, 0].plot(trace["time"][:-1], trace["actions"], label=method)
    axes[0, 0].set(ylabel="Vehicle speed [m/s]", title="One paired test scenario")
    axes[0, 1].set(ylabel="Braking slip", ylim=(0, 1))
    axes[1, 0].set(ylabel="Brake torque [Nm]", xlabel="Time [s]")
    values = [np.mean([r["distance_m"] for r in closed_rows if r["method"] == m]) for m in methods]
    axes[1, 1].bar(methods, values, color="steelblue")
    axes[1, 1].set(ylabel="Mean distance to 2 m/s [m]", title="All test roads, speeds and seeds")
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[0, 0].legend(ncol=2)
    fig.savefig(folder/"closed_loop.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    fractions = (0.25, 0.5, 0.9, 1.2)
    for method in ("fixed", "nls", *KINDS):
        values = [np.mean([r["absolute_error"] for r in history_rows
                           if r["method"] == method and r["coverage_target_fraction"] == f
                           and r["history_steps"] == c["history_steps"]]) for f in fractions]
        ax.plot(fractions, values, "o-", label=method)
    ax.set(xlabel="Prefix coverage target / true peak slip (evaluation only)",
           ylabel="Mean absolute peak-slip error", title="Finite-history identification")
    ax.legend()
    fig.savefig(folder/"history_study.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    lengths = sorted({r["history_steps"] for r in history_rows})
    for method in ("fixed", "nls", *KINDS):
        values = [np.mean([r["absolute_error"] for r in history_rows
                           if r["method"] == method and r["history_steps"] == length
                           and r["coverage_target_fraction"] == 0.9]) for length in lengths]
        ax.plot(np.asarray(lengths)*c["control_dt"], values, "o-", label=method)
    ax.set(xlabel="History duration [s]", ylabel="Mean absolute peak-slip error",
           title="History-length sensitivity at the same pre-peak observation")
    ax.legend()
    fig.savefig(folder/"history_length.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for kind in KINDS:
        for seed in c["seeds"]:
            path = output/"models"/f"{kind}_{seed}_history.csv"
            with path.open(encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            ax = axes[1 if kind == "direct" else 0]
            ax.plot([int(r["epoch"]) for r in history], [float(r["val_loss"]) for r in history], label=f"{kind}/{seed}")
    for ax in axes:
        ax.set(xlabel="Epoch", ylabel="Validation loss")
        ax.legend(fontsize=7)
    axes[0].set_title("World and current-only: same loss")
    axes[1].set_title("Direct peak: different loss; separate scale")
    fig.savefig(folder/"training.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    time_axis = c["control_dt"]*np.arange(1, c["horizon_steps"]+1)
    for name, branch in diagnostic["branches"].items():
        line, = ax.plot(time_axis, slip(branch["truth"], c["radius"]), label=f"{name}: simulator")
        ax.plot(time_axis, slip(branch["predicted"], c["radius"]), "--", color=line.get_color(), label=f"{name}: model")
    ax.set(xlabel="Prediction time [s]", ylabel="Braking slip", title="Same history, different candidate torques")
    ax.legend(fontsize=8)
    fig.savefig(folder/"action_branches.png", dpi=160)
    plt.close(fig)


def plot_curves(c, output, manifest, models, data):
    import matplotlib.pyplot as plt

    road = manifest["roads"]["test"][0]
    i = int(np.flatnonzero((data["road_index"] == 0)
                           & (data["initial_speed"] == max(c["initial_speeds"])))[0])
    observed = slip(data["states"][i, :data["lengths"][i]], c["radius"])
    grid = np.linspace(0, 1, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, fraction in zip(axes, (0.5, 1.2)):
        hits = np.flatnonzero(observed >= fraction*road["peak"])
        t = max(0, int(hits[0])-(fraction < 1)) if len(hits) else len(observed)-1
        h = make_history(data["states"][i], data["actions"][i], t, c["history_steps"], c)
        ax.plot(grid, friction(grid, road["theta"]), "k", linewidth=2, label="Simulator truth")
        estimator = LeastSquaresPeak(c)
        estimator(h)
        if estimator.theta is not None:
            ax.plot(grid, friction(grid, estimator.theta), label="NLS")
        for kind in ("world", "current"):
            model = models[(kind, c["seeds"][0])]
            with torch.inference_mode():
                context = model.encode(torch.as_tensor(h[None], device=c["device"]))
                mu = model.friction(torch.as_tensor(grid, dtype=torch.float32, device=c["device"]),
                                    context.expand(len(grid), -1)).cpu().numpy()
            ax.plot(grid, mu, label=f"{kind}, seed {c['seeds'][0]}")
        ax.axvspan(float(h[:, 2].min()), float(h[:, 2].max()), color="grey", alpha=0.15,
                   label="Observed slip interval")
        ax.axvline(road["peak"], color="k", linestyle=":", linewidth=1)
        ax.set(xlabel="Braking slip", ylabel="Friction coefficient", xlim=(0, 1), ylim=(0, 2),
               title=f"History at t={t*c['control_dt']:.3f}s; target coverage {fraction:g} x peak")
        ax.legend(fontsize=8)
    fig.savefig(output/"figures"/"friction_curves.png", dpi=160)
    plt.close(fig)


def make_report(c, output, open_rows, closed_rows, paired):
    lines = ["# ABSWM experiment report", "", "## Material Passport", "",
             "- Type: executable simulation research prototype", "- Verification status: EXECUTED (simulation only)",
             f"- Profile: {c['name']}; roads train/val/test: {c['roads']}; seeds: {c['seeds']}",
             "- Inputs: resolved config.json, data/manifest.json; output provenance: environment.json", "",
             "## Interpretation boundary", "",
             "This run tests a quarter-car on fixed, uniform Burckhardt roads. Test parameters are unseen",
             "combinations from the same family as training. It does not demonstrate generalization to",
             "another tire-model family, changing roads, measurement noise, ESC, or real vehicles.", "",
             "The world model predicts action-conditioned futures; its ABS controller currently uses only",
             "the estimated friction-curve peak. Closed-loop distance improvements cannot by themselves",
             "establish a benefit from action planning. The direct-peak baseline receives peak labels;",
             "world/current models receive only observed-trajectory force and future-state labels.", "",
             "## Closed-loop results", "",
             "Distance is measured to 2 m/s. Oracle uses the same PI and actuator limits; it is a reference,",
             "not a guaranteed global optimum. Rows average over paired roads/speeds and available seeds.", "",
             "| Method | Distance [m] | High-slip time [s] | Peak MAE | Selector p95 [ms]* | Incomplete runs |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in ("fixed", "oracle", "nls", *KINDS):
        group = [r for r in closed_rows if r["method"] == method]
        vals = [np.mean([r[k] for r in group]) for k in ("distance_m", "high_slip_s", "peak_mae", "selector_p95_ms")]
        lines.append(f"| {method} | {vals[0]:.3f} | {vals[1]:.4f} | {vals[2]:.5f} | {vals[3]:.3f} | {sum(not r['stopped'] for r in group)} |")
    lines += ["", "*Mean of within-run selector p95 values; includes encoding and curve search/fit,",
              "excludes plant integration. It is a CPU wall-clock measurement, not a real-time guarantee.", "",
              "## Open-loop results", "",
              "Fixed uses a public wet-road curve for its rollout; its control reference remains 0.12.", "",
              "| Method | Peak MAE | Rollout v RMSE [m/s] | Rollout omega RMSE [rad/s] | Curve RMSE |",
              "|---|---:|---:|---:|---:|"]
    for method in ("fixed", "nls", *KINDS):
        group = [r for r in open_rows if r["method"] == method]
        formatted = []
        for key in ("peak_mae", "rollout_v_rmse", "rollout_omega_rmse", "curve_rmse"):
            values = [r[key] for r in group if r[key] is not None]
            formatted.append(f"{np.mean(values):.5f}" if values else "—")
        lines.append(f"| {method} | " + " | ".join(formatted) + " |")
    lines += ["", "## Paired distance differences", "",
              "Negative values favor world. Bootstrap resamples roads as clusters after averaging speeds",
              "and seeds within each road. Small test sets yield unstable intervals; these are descriptive.", "",
              "| Comparison | Mean difference [m] | Road-cluster bootstrap 95% interval [m] |",
              "|---|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['comparison']} | {row['mean_distance_delta_m']:.3f} | [{row['bootstrap_95_low_m']:.3f}, {row['bootstrap_95_high_m']:.3f}] |")
    lines += ["", "## Diagnostics and limitations", "",
              "- history_study.csv: prefix identification before/near/after the true peak, with achieved coverage recorded.",
              "- action_study.csv: release/hold/increase torque forecasts from the same measured history.",
              "- open_loop.csv: separate observed/unobserved curve errors, low-excitation errors and clipped reference errors.",
              "- Learned curve maxima outside the allowed reference interval are reported before clipping.",
              "- Training force labels come only from the simulated trajectory, not from full-curve labels.",
              "- The predictor uses projected 1 ms Euler; the plant uses projected 1 ms RK4. Numerical error is part of the reported prediction error.",
              "- NLS uses causal finite differences and public typical-road starting points; learning models additionally have offline training priors.",
              "- Closed-loop NLS retains its last parameter estimate; world/direct encode the recent finite window. Open-loop and prefix NLS are reset for each window.",
              "- High slip means lambda > 0.8; values describe this simulation, not certified safety.",
              "- Demo training is deliberately short. Do not treat its ranking as a publication result.", "",
              "## Figures", "", "![Closed loop](figures/closed_loop.png)", "",
              "![History](figures/history_study.png)", "", "![Actions](figures/action_branches.png)", "",
              "![History length](figures/history_length.png)", "",
              "![Friction curves](figures/friction_curves.png)", "",
              "![Training](figures/training.png)", ""]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_all(c, output):
    manifest = json.loads((output/"data"/"manifest.json").read_text(encoding="utf-8"))
    seed_everything(c["seeds"][0], c)
    models = load_models(c, output)
    gains = tune_controller(c, manifest["roads"]["val"], output)
    opened, data = evaluate_open(c, output, manifest, models)
    history = history_study(c, output, manifest, models, data)
    _, diagnostic = action_study(c, output, manifest, models, data)
    closed, example = evaluate_closed(c, output, manifest, models, gains)
    paired = paired_comparisons(c, closed)
    write_json(output/"paired_comparisons.json", paired)
    plots(c, output, opened, closed, history, example, diagnostic)
    plot_curves(c, output, manifest, models, data)
    make_report(c, output, opened, closed, paired)
    print(f"report: {output/'report.md'}", flush=True)
