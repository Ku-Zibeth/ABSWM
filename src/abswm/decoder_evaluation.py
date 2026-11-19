"""Matched windows, counterfactual actions and seed comparisons for E2–E4."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .config import write_json
from .controllers import LeastSquaresPeak
from .data import load_split, make_history
from .decoder_stress import make_stress_cases, tail_friction
from .decoder_study import LinearProbe, load_windows, read_json
from .diagnostic_experiments import paired_ramp, prefix_before_cap
from .diagnostics import (chunks, coverage_masks, encode, neural_curve, neural_rollout,
                          parametric_forecast, probe_targets, ridge_predictions, rmse, tensor)
from .evaluation import write_csv
from .physics import TYPICAL_ROADS, friction, slip


PEAK_ONLY = {"direct", "linear_probe", "e2"}
COMPONENTS = ("peak_mae", "peak_mu_mae", "curve_rmse", "inside_rmse", "outside_rmse", "above_rmse",
              "point_mu_rmse", "point_fx_rmse_n", "rollout_v_rmse", "rollout_w_rmse", "rollout_slip_rmse",
              "end_v_rmse", "end_w_rmse", "end_slip_rmse")


@torch.inference_mode()
def predictions(c, model, method, history, grid, actions=None):
    if method == "nls":
        estimates, params = [], []
        for h in history:
            estimator = LeastSquaresPeak(c)
            estimates.append(estimator(h))
            params.append(estimator.theta if estimator.theta is not None else TYPICAL_ROADS["wet"])
        theta = np.array(params)
        curves = friction(grid[None], theta[:, None])
        point = friction(history[:, -1, 2], theta)
        rollout = None if actions is None else parametric_forecast(history[:, -1, :2], actions, theta, c)
        return np.array(estimates), curves, point, rollout
    if method in PEAK_ONLY:
        peak = [model.peak(tensor(h, c)).cpu().numpy() for h in chunks(history, c)]
        return np.concatenate(peak), None, None, None
    z = encode(model, history, c)
    curves = neural_curve(model, np.broadcast_to(grid, (len(z), len(grid))), z, c)
    if method in ("world", "current"):
        peak = np.concatenate([model.peak(tensor(h, c), grid_size=len(grid)).cpu().numpy()
                               for h in chunks(history, c)])
    else:
        peak = grid[curves.argmax(axis=1)]
    point = neural_curve(model, history[:, -1:, 2], z, c)[:, 0]
    rollout = None if actions is None else neural_rollout(model, history, actions, z, c)
    return peak, curves, point, rollout


def grouped_metrics(c, point, rollout, sample, mask):
    if rollout is None:
        return {k: None for k in COMPONENTS[6:]}
    state = rollout[mask]-sample["states"][mask, 1:]
    slip_error = slip(rollout[mask], c["radius"])-slip(sample["states"][mask, 1:], c["radius"])
    point_error = point[mask]-sample["mu"][mask]
    return {"point_mu_rmse": rmse(point_error), "point_fx_rmse_n": rmse(c["mass"]*c["gravity"]*point_error),
            "rollout_v_rmse": rmse(state[..., 0]), "rollout_w_rmse": rmse(state[..., 1]),
            "rollout_slip_rmse": rmse(slip_error), "end_v_rmse": rmse(state[:, -1, 0]),
            "end_w_rmse": rmse(state[:, -1, 1]), "end_slip_rmse": rmse(slip_error[:, -1])}


def evaluate_windows(c, d, models, sample, labels, grid):
    theta = np.array([r["theta"] for r in labels])
    family = np.array([r["family"] for r in labels])
    roads = np.array([r["id"] for r in labels])
    curves = friction(grid[None], theta[:, None])
    true_peak_mu = friction(sample["peak"], theta)
    masks = coverage_masks(np.broadcast_to(grid, curves.shape), sample["coverage"])
    groups = {("all", "all"): np.ones(len(roads), dtype=bool)}
    groups.update({("family", f): family == f for f in np.unique(family)})
    observed = sample["coverage"][:, 1]
    edges = d["coverage_edges"]
    for lo, hi in zip(edges, edges[1:]):
        label = f"[{lo:g},{hi:g}" + ("]" if hi == edges[-1] else ")")
        groups["max_slip", label] = (observed >= lo) & ((observed <= hi) if hi == edges[-1] else (observed < hi))
    by_group, by_horizon, by_window = [], [], []
    for (method, seed), model in models.items():
        peak, predicted_curve, point, rollout = predictions(c, model, method, sample["history"], grid, sample["actions"])
        keys = {"method": method, "seed": seed}
        abs_peak = np.abs(peak-sample["peak"])
        curve_error = None if predicted_curve is None else predicted_curve-curves
        peak_mu_error = None if predicted_curve is None else predicted_curve.max(axis=1)-true_peak_mu
        for (group, label), mask in groups.items():
            if not mask.any():
                continue
            row = {**keys, "group": group, "label": label, "windows": int(mask.sum()),
                   "peak_mae": float(abs_peak[mask].mean()), "peak_bias": float((peak[mask]-sample["peak"][mask]).mean()),
                   "peak_mu_mae": float(abs(peak_mu_error[mask]).mean()) if peak_mu_error is not None else None}
            for region in ("all", "inside", "outside", "above"):
                name = "curve_rmse" if region == "all" else region+"_rmse"
                row[name] = rmse(curve_error[mask][masks[region][mask]]) if curve_error is not None else None
            by_group.append({**row, **grouped_metrics(c, point, rollout, sample, mask)})
        if rollout is not None:
            error = rollout-sample["states"][:, 1:]
            slip_error = slip(rollout, c["radius"])-slip(sample["states"][:, 1:], c["radius"])
            for step in d["forecast_steps"]:
                k = step-1
                by_horizon.append({**keys, "step": step, "time_ms": step*c["control_dt"]*1000,
                                   "v_rmse": rmse(error[:, k, 0]), "w_rmse": rmse(error[:, k, 1]),
                                   "slip_rmse": rmse(slip_error[:, k])})
        for i, (trace, t) in enumerate(sample["locators"]):
            by_window.append({**keys, "window": i, "road_id": roads[i], "family": family[i], "trace": trace, "t": t,
                              "max_slip": observed[i], "true_peak": sample["peak"][i], "predicted_peak": peak[i],
                              "peak_abs_error": abs_peak[i], "peak_mu_abs_error": abs(peak_mu_error[i])
                              if curve_error is not None else None,
                              "curve_rmse": rmse(curve_error[i]) if curve_error is not None else None,
                              "above_rmse": rmse(curve_error[i, masks["above"][i]]) if curve_error is not None else None,
                              "endpoint_w_abs_error": abs(error[i, -1, 1]) if rollout is not None else None})
    return by_group, by_horizon, by_window


def mean_and_spread(rows, group_keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    output = []
    for key, group in groups.items():
        result = {k: v for k, v in zip(group_keys, key)}
        result["seeds"] = len({r["seed"] for r in group})
        for metric in metrics:
            values = [float(r[metric]) for r in group if r[metric] is not None]
            result[metric+"_mean"] = float(np.mean(values)) if values else None
            result[metric+"_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        output.append(result)
    return output


def summarize_cases(rows, groups):
    metrics = ("predicted_peak", "peak_abs_error", "peak_mu_abs_error", "all_rmse", "inside_rmse", "outside_rmse",
               "above_rmse", "tail_rmse_ge_0_4", "tail_mu_true", "tail_mu_predicted")
    split = {}
    keys = ("method", "seed", *groups)
    for row in rows:
        split.setdefault(tuple(row[k] for k in keys), []).append(row)
    per_seed = []
    for key, group in split.items():
        summary = dict(zip(keys, key))
        summary["cases"] = len(group)
        for metric in metrics:
            values = [r[metric] for r in group if r[metric] is not None]
            summary[metric] = float(np.mean(values)) if values else None
        per_seed.append(summary)
    return mean_and_spread(per_seed, ("method", *groups, "cases"), metrics)


def paired_differences(d, rows):
    lookup = {(r["method"], r["seed"], r["window"]): r for r in rows}
    windows = sorted({r["window"] for r in rows})
    seeds = sorted({r["seed"] for r in rows if r["seed"] >= 0})
    rng = np.random.default_rng(d["analysis_seed"])
    pairs = [("e2", "linear_probe"), ("e3_original", "world"), ("e3", "e3_original"),
             ("e3", "nls"), ("e3_refine", "e3"), ("e4_a", "e3_matched"), ("e4_b", "e4_a"),
             ("e4_c", "e4_b"), ("e4_d", "e4_c"), ("e4_d_weighted", "e4_d")]
    out = []
    for left, right in pairs:
        for metric in ("peak_abs_error", "curve_rmse", "above_rmse", "endpoint_w_abs_error"):
            per_road = {}
            for seed, w in itertools.product(seeds, windows):
                a = lookup[left, seed, w]
                b = lookup[right, -1 if right == "nls" else seed, w]
                if a[metric] is not None and b[metric] is not None:
                    per_road.setdefault(a["road_id"], []).append(float(a[metric])-float(b[metric]))
            if per_road:
                values = np.array([np.mean(group) for group in per_road.values()])
                bootstrap = rng.choice(values, (d["bootstrap_samples"], len(values))).mean(axis=1)
                low, high = np.percentile(bootstrap, [2.5, 97.5])
                out.append({"method": left, "baseline": right, "metric": metric, "road_clusters": len(values),
                            "mean_delta": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)})
    return out


def pair_cases(c, d, roads):
    data, labels = [], []
    for name, road_id in d["pair_roads"].items():
        r = roads[road_id]
        for speed, rate in itertools.product(c["initial_speeds"], d["pair_torque_rates"]):
            states, actions = paired_ramp(r["theta"], speed, rate, c, max(d["pair_caps"]))
            observed = slip(states, c["radius"])
            for cap in d["pair_caps"]:
                t, reached = prefix_before_cap(observed, cap)
                data.append(make_history(states, actions, t, c["history_steps"], c))
                labels.append({"curve": name, "slip_cap": cap, "speed": speed, "torque_rate": rate,
                               "crossed_cap": reached, **r})
    return np.array(data), labels


def curve_cases(c, models, grid, history, labels, *, shifted=False):
    theta = np.array([r["theta"] for r in labels])
    actual = (tail_friction(grid[None], theta[:, None], np.array([r["shift"] for r in labels])[:, None])
              if shifted else friction(grid[None], theta[:, None]))
    peak_mu = actual.max(axis=1)
    observed = history[..., 2]
    coverage = np.column_stack((observed.min(axis=1), observed.max(axis=1)))
    masks = coverage_masks(np.broadcast_to(grid, actual.shape), coverage)
    tail = grid >= 0.4
    retained, rows = {}, []
    for (method, seed), model in models.items():
        peak, curve, _, _ = predictions(c, model, method, history, grid)
        if not shifted and curve is not None:
            retained[method, seed] = curve
        error = None if curve is None else curve-actual
        for i, label in enumerate(labels):
            rows.append({**{k: v for k, v in label.items() if k != "theta"}, "method": method, "seed": seed,
                         "observed_min": coverage[i, 0], "observed_max": coverage[i, 1],
                         "predicted_peak": peak[i], "peak_abs_error": abs(peak[i]-label["peak"]),
                         "peak_mu_abs_error": abs(curve[i].max()-peak_mu[i]) if curve is not None else None,
                         **{region+"_rmse": rmse(error[i, mask[i]]) if curve is not None else None
                            for region, mask in masks.items()},
                         "tail_rmse_ge_0_4": rmse(error[i, tail]) if curve is not None else None,
                         "tail_mu_true": float(actual[i, tail].mean()),
                         "tail_mu_predicted": float(curve[i, tail].mean()) if curve is not None else None})
    return rows, (actual, coverage, retained)


def branch_cases(c, base, roads):
    data = load_split(base/"data", "test")
    histories, actions, truth = [], [], []
    for i, r in enumerate(roads):
        candidates = (data["road_index"] == i) & (data["initial_speed"] == max(c["initial_speeds"]))
        trace = int(np.flatnonzero(candidates)[0])
        lam = slip(data["states"][trace, :data["lengths"][trace]], c["radius"])
        t, _ = prefix_before_cap(lam, 0.05)
        h = make_history(data["states"][trace], data["actions"][trace], t, c["history_steps"], c)
        u0 = float(h[-1, 3])
        limit = c["torque_rate"]*c["control_dt"]*np.arange(1, c["horizon_steps"]+1)
        for requested in (0.0, u0, min(c["torque_max"], u0+600)):
            histories.append(h)
            actions.append(u0+np.clip(requested-u0, -limit, limit))
            truth.append(r["theta"])
    h, u = np.asarray(histories), np.asarray(actions, dtype=np.float32)
    return h, u, parametric_forecast(h[:, -1, :2], u, np.asarray(truth), c)


def action_comparison(c, d, models, cases, grid):
    history, actions, truth = cases
    lam = slip(truth, c["radius"])
    rows = []
    for (method, seed), model in models.items():
        if method in PEAK_ONLY:
            continue
        _, _, _, rollout = predictions(c, model, method, history, grid, actions)
        delta = rollout-truth
        slip_error = slip(rollout, c["radius"])-lam
        for i, branch in enumerate(("release", "hold", "increase")):
            for step in d["forecast_steps"]:
                k = step-1
                rows.append({"method": method, "seed": seed, "branch": branch, "step": step,
                             "time_ms": 1000*step*c["control_dt"], "v_rmse": rmse(delta[i::3, k, 0]),
                             "w_rmse": rmse(delta[i::3, k, 1]), "slip_rmse": rmse(slip_error[i::3, k])})
    return rows


@torch.inference_mode()
def latent_probes(c, d, models, train, test, train_labels, test_labels):
    def targets(sample, labels):
        cfg = {"probe_slips": [0.02, 0.1, 0.2, 0.6]}
        return probe_targets(c, cfg, sample, np.array([r["family"] for r in labels]),
                             np.array([r["theta"] for r in labels]))
    names, y, train_class = targets(train, train_labels)
    _, truth, test_class = targets(test, test_labels)
    out = []
    for (method, seed), model in models.items():
        if not (method == "world" or method.startswith(("e3", "e4"))):
            continue
        a = encode(model, train["history"], c).cpu().numpy().astype(float)
        b = encode(model, test["history"], c).cpu().numpy().astype(float)
        pred = ridge_predictions(a, b, y, d["ridge"])
        for i, name in enumerate(names):
            error = pred[:, i]-truth[:, i]
            variance = ((truth[:, i]-truth[:, i].mean())**2).sum()
            out.append({"method": method, "seed": seed, "target": name, "rmse": rmse(error),
                        "mae": float(np.abs(error).mean()), "r2": float(1-(error**2).sum()/variance) if variance else None,
                        "balanced_accuracy": None})
        actual = test_class.argmax(axis=1)
        correct = ridge_predictions(a, b, train_class, d["ridge"]).argmax(axis=1) == actual
        out.append({"method": method, "seed": seed, "target": "road_family", "rmse": None, "mae": None, "r2": None,
                    "balanced_accuracy": float(np.mean([correct[actual == label].mean() for label in np.unique(actual)]))})
    return out


@torch.inference_mode()
def benchmark(c, d, models, sample):
    output = []
    n = len(sample["history"])
    ids = np.random.default_rng(d["analysis_seed"]).choice(n, d["timing_repetitions"], replace=d["timing_repetitions"] > n)
    h, u = tensor(sample["history"], c), tensor(sample["actions"], c)
    for (method, seed), model in models.items():
        def peak(i):
            if method == "nls":
                return LeastSquaresPeak(c)(sample["history"][i])
            if method in PEAK_ONLY:
                return model.peak(h[i:i+1]).cpu()
            return model.peak(h[i:i+1], grid_size=d["test_grid_points"]).cpu()

        def rollout(i):
            if method == "nls":
                nls = LeastSquaresPeak(c)
                nls(sample["history"][i])
                theta = nls.theta if nls.theta is not None else TYPICAL_ROADS["wet"]
                return parametric_forecast(sample["history"][i:i+1, -1, :2], sample["actions"][i:i+1], theta, c)
            return model.rollout(h[i:i+1], u[i:i+1]).cpu()

        def time_call(call):
            for i in np.resize(ids, d["timing_warmup"]):
                call(i)
            durations = []
            for i in ids:
                start = perf_counter()
                call(i)
                durations.append(1000*(perf_counter()-start))
            return list(np.percentile(durations, [50, 95]))

        pp = time_call(peak)
        rp = [None, None] if method in PEAK_ONLY else time_call(rollout)
        if method == "linear_probe":
            total = sum(p.numel() for p in model.model.encoder.parameters())+len(model.coefficients)+1
            trainable = len(model.coefficients)+1
        elif model is None:
            total, trainable = 3, 0
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        output.append({"method": method, "seed": seed, "parameters": total, "trainable_parameters": trainable,
                       "peak_p50_ms": pp[0], "peak_p95_ms": pp[1], "rollout_p50_ms": rp[0], "rollout_p95_ms": rp[1]})
    return output


def plots(output, c, d, summary, grouped, horizon, cases, pair_labels, grid, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output/"figures"
    folder.mkdir(exist_ok=True)
    styles = {"current": "Current", "world": "World", "linear_probe": "Linear probe", "direct": "Direct",
              "nls": "NLS", "e2": "E2", "e3": "E3", "e3_original": "E3 original head",
              "e3_refine": "E3 continued", "e3_matched": "E3 matched LR", "e4_a": "E4 A", "e4_b": "E4 B", "e4_c": "E4 C",
              "e4_d": "E4 D", "e4_d_weighted": "E4 D weighted"}
    selected = ["nls", "world", "e3", chosen]
    truth, coverage, estimates = cases
    fig, axes = plt.subplots(2, len(d["pair_caps"]), figsize=(13, 6), layout="constrained", sharex=True, sharey=True)
    for i, j in np.ndindex(axes.shape):
        name, cap = list(d["pair_roads"])[i], d["pair_caps"][j]
        idx = next(k for k, r in enumerate(pair_labels) if r["curve"] == name and r["slip_cap"] == cap
                   and r["speed"] == max(c["initial_speeds"]) and r["torque_rate"] == d["pair_torque_rates"][0])
        ax = axes[i, j]
        ax.axvspan(*coverage[idx], color="0.90")
        ax.plot(grid, truth[idx], color="black", linewidth=2, label="True")
        for method in selected:
            values = [value[idx] for (key, _), value in estimates.items() if key == method]
            line, = ax.plot(grid, np.mean(values, axis=0), label=styles[method], linewidth=1.4)
            if len(values) > 1 and method in ("e3", chosen):
                ax.fill_between(grid, np.min(values, axis=0), np.max(values, axis=0), color=line.get_color(), alpha=0.12)
        ax.set(title=f"{name} · cap {cap:.0%}", xlabel="Slip" if i else "",
               ylabel="Friction coefficient" if j == 0 else "", xlim=(0, 1))
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.savefig(folder/"pair_curves.png", dpi=175)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    methods = list(styles)
    overall = {r["method"]:r for r in summary}
    axs[0].barh(list(range(len(methods))), [100*overall[m]["peak_mae_mean"] for m in methods],
                xerr=[100*(overall[m]["peak_mae_seed_sd"] or 0) for m in methods], color="#366976")
    axs[0].set(yticks=range(len(methods)), yticklabels=[styles[m] for m in methods], xlabel="Peak MAE [percentage points]")
    axs[0].invert_yaxis()
    coverage = [r for r in grouped if r["group"] == "max_slip" and r["method"] == "e3"]
    labels = [r["label"] for r in coverage]
    for method in ("nls", "direct", "linear_probe", "e2", "e3", chosen):
        values = {r["label"]: 100*r["peak_mae_mean"] for r in grouped if r["group"] == "max_slip" and r["method"] == method}
        axs[1].plot(labels, [values[label] for label in labels], "o-", label=styles[method])
    axs[1].set(xlabel="Maximum slip in observed history", ylabel="Peak MAE [percentage points]")
    axs[1].tick_params(axis="x", rotation=35)
    axs[1].legend(fontsize=8)
    fig.savefig(folder/"peak_errors.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, (field, label) in zip(axes, (("v_rmse", "Speed RMSE [m/s]"), ("w_rmse", "Wheel speed RMSE [rad/s]"),
                                        ("slip_rmse", "Slip RMSE"))):
        for method in ("world", "current", "nls", "e3", "e4_a", "e4_b", "e4_c", "e4_d", "e4_d_weighted"):
            rows = [r for r in horizon if r["method"] == method]
            ax.plot([r["time_ms"] for r in rows], [r[field+"_mean"] for r in rows], ".-", label=styles[method])
        ax.set(xlabel="Forecast horizon [ms]", ylabel=label)
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.savefig(folder/"dynamics.png", dpi=175)
    plt.close(fig)


def evaluate_study(c, d, manifest, base, output, models, train):
    start = perf_counter()
    train_sample, train_indices = load_windows(c, base, "train")
    np.testing.assert_array_equal(train_sample["locators"], train["locators"])
    linear_fits = []
    for seed in c["seeds"]:
        start_fit = perf_counter()
        probe = LinearProbe(models["world", seed], train["history"], train["peak"], c, d["ridge"])
        fit_seconds = perf_counter()-start_fit
        models["linear_probe", seed] = probe
        linear_fits.append({"method": "linear_probe", "seed": seed, "fit_seconds": fit_seconds,
                            "train_windows": len(train["peak"]), "ridge": d["ridge"]})
        np.savez_compressed(output/"models"/f"linear_probe_{seed}.npz", center=probe.center, scale=probe.scale,
                            coefficients=probe.coefficients, intercept=probe.intercept)
    write_csv(output/"linear_probe_fit.csv", linear_fits)
    models["nls", -1] = None
    test, test_indices = load_windows(c, base, "test")
    roads = manifest["roads"]
    train_labels = [roads["train"][i] for i in train_indices]
    test_labels = [roads["test"][i] for i in test_indices]
    grid = np.linspace(0, 1, d["test_grid_points"])
    print("evaluate matched test windows", flush=True)
    by_group, by_horizon, by_window = evaluate_windows(c, d, models, test, test_labels, grid)
    write_csv(output/"test_metrics.csv", by_group)
    write_csv(output/"window_errors.csv", by_window)
    write_csv(output/"horizon_by_seed.csv", by_horizon)
    write_csv(output/"paired_differences.csv", paired_differences(d, by_window))
    grouped = mean_and_spread(by_group, ("method", "group", "label", "windows"), COMPONENTS)
    summary = [r for r in grouped if r["group"] == "all"]
    write_csv(output/"comparison.csv", summary)
    write_csv(output/"stratified_comparison.csv", grouped)
    horizon = mean_and_spread(by_horizon, ("method", "step", "time_ms"), ("v_rmse", "w_rmse", "slip_rmse"))
    write_csv(output/"horizon_comparison.csv", horizon)

    print("evaluate the matched prefixes and the shifted friction tails", flush=True)
    history, pair_labels = pair_cases(c, d, {r["id"]:r for r in roads["test"]})
    pair_rows, curves = curve_cases(c, models, grid, history, pair_labels)
    write_csv(output/"pair_estimates.csv", pair_rows)
    write_csv(output/"pair_comparison.csv", summarize_cases(pair_rows, ("curve", "slip_cap")))
    np.savez_compressed(output/"pair_curves.npz", slip_grid=grid, truth=curves[0], coverage=curves[1],
                        method=np.array([m for m, _ in curves[2]]), seed=np.array([s for _, s in curves[2]]),
                        predictions=np.stack(list(curves[2].values())))
    stress_history, stress_labels = make_stress_cases(c, roads["test"])
    stress, _ = curve_cases(c, models, grid, stress_history, stress_labels, shifted=True)
    write_csv(output/"tail_stress.csv", stress)
    write_csv(output/"tail_stress_comparison.csv", summarize_cases(stress, ("shift", "slip_cap")))
    matched = {}
    for h, row in zip(stress_history, stress_labels):
        if row["slip_cap"] == 0.20:
            matched.setdefault((row["road_id"], row["speed"]), []).append(h)
    identical = sum(len(group) == 3 and all(np.array_equal(group[0], h) for h in group[1:]) for group in matched.values())
    write_json(output/"tail_stress_protocol.json", {"cases": len(stress_labels), "ramp_nm_s": 1800, "caps": [0.20, 0.65],
                                                       "matched_early_histories": identical, "early_history_groups": len(matched)})

    print("evaluate candidate torques, latent probes and single-window timing", flush=True)
    branches = action_comparison(c, d, models, branch_cases(c, base, roads["test"]), grid)
    write_csv(output/"action_by_seed.csv", branches)
    write_csv(output/"action_comparison.csv", mean_and_spread(branches, ("method", "branch", "step", "time_ms"),
                                                               ("v_rmse", "w_rmse", "slip_rmse")))
    write_csv(output/"latent_probes.csv", latent_probes(c, d, models, train, test, train_labels, test_labels))
    write_csv(output/"inference_timing.csv", benchmark(c, d, models, test))
    training = read_json(output/"training_summary.json")
    names = sorted({r["method"] for r in training if r["method"].startswith("e4_")})
    scores = {name: float(np.mean([r["val_score"] for r in training if r["method"] == name])) for name in names}
    chosen = min(scores, key=scores.get)
    plots(output, c, d, summary, grouped, horizon, curves, pair_labels, grid, chosen)
    hashes = {}
    for path in (base/"data"/"test.npz", base/"open_loop_windows.npz", base/"run_timing.json"):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__), Path(__file__).with_name("decoder_stress.py"))}
    write_json(output/"evaluation.json", {"base_test_sha256": hashes, "evaluation_source_sha256": source,
                                           "test_windows": len(test_indices), "road_clusters": len(set(test_indices)),
                                           "validation_chosen_e4": chosen, "e4_validation_scores": scores,
                                           "evaluation_seconds": perf_counter()-start})
    print(f"Decoder study results: {output}", flush=True)
