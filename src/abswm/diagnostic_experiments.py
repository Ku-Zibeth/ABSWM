"""Matched excitation and PI interventions for phase-one research."""

from __future__ import annotations

import csv
import itertools

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from .controllers import LeastSquaresPeak, simulate_closed
from .data import make_history
from .diagnostics import coverage_masks, encode, neural_curve, rmse, tensor
from .physics import TYPICAL_ROADS, advance_scalar, friction, slip


def prefix_before_cap(observed_slip, cap):
    observed_slip = np.asarray(observed_slip)
    crossings = np.flatnonzero(observed_slip > cap)
    if not len(observed_slip) or (crossings.size and crossings[0] == 0):
        raise ValueError("A capped prefix requires an initial observation within the cap")
    return (int(crossings[0])-1, True) if crossings.size else (len(observed_slip)-1, False)


def paired_ramp(theta, speed, ramp_rate, c, max_cap):
    """Apply a common physical torque schedule until the highest cap is crossed."""
    dt = c["control_dt"]
    v, w = speed, speed/c["radius"]
    states, actions = [(v, w)], []
    for step in range(round(c["data_time"]/dt)):
        u = min(c["torque_max"], min(ramp_rate, c["torque_rate"])*(step+1)*dt)
        v, w, _ = advance_scalar(v, w, u, theta, c)
        states.append((v, w))
        actions.append(u)
        if 1-c["radius"]*w/max(v, 1e-8) > max_cap or v <= c["stop_speed"]:
            break
    return np.array(states), np.array(actions)


def midpoint_observations(history, c):
    """Same observable samples and mask as the production NLS."""
    h = np.asarray(history, dtype=float)
    lam = (h[1:, 2]+h[:-1, 2])/2
    mu = -np.diff(h[:, 0])/(c["control_dt"]*c["gravity"])
    selected = (lam > 0.002) & (mu >= 0) & (mu <= 2.0)
    return lam[selected], mu[selected]


def local_condition(lam, theta):
    """Evaluator-only condition number of d mu / d log(theta)."""
    if len(lam) < 3:
        return None
    a, b, d = theta
    jacobian = np.column_stack((a*(-np.expm1(-b*lam)), a*b*lam*np.exp(-b*lam), -d*lam))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return float(singular[0]/singular[-1]) if singular[-1] > np.finfo(float).eps*singular[0] else None


def flat_pchip(lam, mu, grid):
    """Local nonparametric baseline with constant continuation above the samples."""
    knots, inverse, counts = np.unique(np.r_[0.0, lam], return_inverse=True, return_counts=True)
    if len(knots) < 2:
        return None
    averages = np.bincount(inverse, weights=np.r_[0.0, mu])/counts
    return PchipInterpolator(knots, averages)(np.minimum(grid, knots[-1]))


@torch.inference_mode()
def pair_study(c, d, roads, models, grid):
    rows, curves = [], []
    high = max(d["pair_caps"])
    for name, road_id in d["pair_roads"].items():
        road = roads[road_id]
        theta = road["theta"]
        true_curve = friction(grid, theta)
        for speed, rate in itertools.product(c["initial_speeds"], d["pair_torque_rates"]):
            states, actions = paired_ramp(theta, speed, rate, c, high)
            observed = slip(states, c["radius"])
            for cap in d["pair_caps"]:
                t, crossed = prefix_before_cap(observed, cap)
                history = make_history(states, actions, t, c["history_steps"], c)
                lam, mu = midpoint_observations(history, c)
                midpoint_error = mu-friction(lam, theta)
                coverage = np.array([[history[:, 2].min(), history[:, 2].max()]])
                masks = coverage_masks(grid[None], coverage)
                base = {"curve": name, "road_id": road_id, "initial_speed_m_s": speed, "torque_ramp_nm_s": rate,
                        "slip_cap": cap, "crossed_cap": crossed, "prefix_time_s": t*c["control_dt"],
                        "real_observations": min(t+1, c["history_steps"]), "midpoint_samples": len(lam),
                        "observed_min": float(coverage[0, 0]), "observed_max": float(coverage[0, 1]),
                        "midpoint_mu_rmse": rmse(midpoint_error),
                        "condition_log_theta": local_condition(lam, theta), "true_peak": road["peak"]}
                estimator = LeastSquaresPeak(c)
                peak = estimator(history)
                nls_theta = estimator.theta if estimator.theta is not None else TYPICAL_ROADS["wet"]
                np_curve = flat_pchip(lam, mu, grid)
                estimates = [("nls", -1, peak, friction(grid, nls_theta), estimator.theta is None)]
                estimates.append(("pchip_flat", -1, float(grid[np_curve.argmax()]) if np_curve is not None else None,
                                  np_curve, np_curve is None))
                for method in ("direct", "world", "current"):
                    for seed in c["seeds"]:
                        model = models[method, seed]
                        curve = None
                        if method == "direct":
                            peak = float(model.peak(tensor(history[None], c))[0])
                        else:
                            context = encode(model, history[None], c)
                            curve = neural_curve(model, grid[None], context, c)[0]
                            peak = float(model.peak(tensor(history[None], c), grid_size=len(grid))[0])
                        estimates.append((method, seed, peak, curve, False))
                for method, seed, peak, curve, used_fallback in estimates:
                    error = curve-true_curve if curve is not None else None
                    rows.append({**base, "method": method, "seed": seed, "fallback": used_fallback, "peak": peak,
                                 "peak_abs_error": abs(peak-road["peak"]) if peak is not None else None,
                                 **{f"{region}_mu_rmse": rmse(error[mask[0]]) if error is not None else None
                                    for region, mask in masks.items()}})
                    if speed == max(c["initial_speeds"]) and rate == d["pair_torque_rates"][0] and curve is not None:
                        curves.append({"road": name, "cap": cap, "method": method, "seed": seed,
                                       "mu": curve, "truth": true_curve, "coverage": coverage[0]})
    return rows, curves


def pair_uncertainty(rows):
    groups = {}
    for row in rows:
        keys = ("curve", "initial_speed_m_s", "torque_ramp_nm_s", "slip_cap", "method")
        if row["seed"] >= 0:
            groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for (name, speed, rate, cap, method), group in groups.items():
        peaks = [r["peak"] for r in group]
        truth = group[0]["true_peak"]
        output.append({"curve": name, "initial_speed_m_s": speed, "torque_ramp_nm_s": rate, "slip_cap": cap,
                       "method": method, "seed_count": len(peaks), "seed_mean_peak": float(np.mean(peaks)),
                       "seed_min_peak": min(peaks), "seed_max_peak": max(peaks),
                       "seed_peak_range": max(peaks)-min(peaks),
                       "truth_in_seed_range": min(peaks) <= truth <= max(peaks)})
    return output


def trace_stats(trace, road, c):
    time = trace["time"]
    dt = np.diff(time)
    lam = trace["slip"][:-1]
    true_peak = road["peak"]
    refs = trace["reference"]
    estimates = trace["estimate"]
    applied = trace["actions"]
    indices = np.arange(len(dt))
    active = indices >= c["history_steps"]
    all_steps = np.ones(len(dt), dtype=bool)
    peak_mu = friction(true_peak, road["theta"])
    mu = friction(lam, road["theta"])

    def weighted(values, mask):
        return float(np.average(values[mask], weights=dt[mask])) if np.any(mask) else None

    return {
        "reference_mae_after_warmup": weighted(np.abs(refs-true_peak), active),
        "reference_rate_gap_mae_after_warmup": weighted(np.abs(refs-np.clip(estimates, c["reference_min"],
                                                                            c["reference_max"])), active),
        "tracking_rmse_after_warmup": rmse_weighted(lam-refs, dt, active),
        "tracking_rmse_all": rmse_weighted(lam-refs, dt, all_steps),
        "achieved_slip_mae_after_warmup": weighted(np.abs(lam-true_peak), active),
        "force_utilization_after_warmup": weighted(mu/peak_mu, active),
        "raw_peak_outside_after_warmup": weighted((estimates < c["reference_min"]) |
                                                (estimates > c["reference_max"]), active),
        "estimate_at_one_after_warmup": weighted(estimates == 1, active),
        "torque_at_limit_fraction": weighted(np.isclose(applied, c["torque_max"], rtol=0, atol=1e-8), all_steps),
        "torque_slew_at_limit_fraction": weighted(np.isclose(np.abs(np.diff(np.r_[0, applied])),
                                                            c["torque_rate"]*c["control_dt"],
                                                            rtol=0, atol=1e-8), all_steps),
        "wheel_lock_time_s": float(dt[lam >= 0.99].sum()),
        "max_slip": float(trace["slip"].max()),
    }


def rmse_weighted(error, weights, mask):
    return float(np.sqrt(np.average(np.square(error[mask]), weights=weights[mask]))) if np.any(mask) else None


def read_closed(run):
    float_keys = ("initial_speed", "distance_m", "time_s", "high_slip_s", "tracking_rmse",
                  "peak_mae", "torque_total_variation", "selector_mean_ms", "selector_p95_ms")
    with (run/"closed_loop.csv").open(encoding="utf-8", newline="") as stream:
        data = list(csv.DictReader(stream))
    for row in data:
        for key in float_keys:
            row[key] = float(row[key]) if row[key] else None
        row["seed"] = int(row["seed"])
        row["selector_updates"] = int(row["selector_updates"])
        row["stopped"] = row["stopped"] == "True"
    return data


def read_trace(run, row):
    name = f"{row['road_id']}_{row['initial_speed']:g}_{row['method']}_{row['seed']}.npz"
    with np.load(run/"traces"/name, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def replay_world(trace, theta, speed, c, gains):
    first_update = ((c["history_steps"]+c["reference_period"]-1)//c["reference_period"])*c["reference_period"]
    estimates = iter(trace["estimate"][first_update::c["reference_period"]])
    return simulate_closed(theta, speed, c, gains, selector=lambda _: float(next(estimates)))


def control_study(c, d, run, roads, gains):
    rows, memories = [], []
    source = read_closed(run)
    fixed_probe_steps = {round(s/c["control_dt"]) for s in d["control_probe_seconds"]}
    replay_checked = set()
    for row in source:
        trace = read_trace(run, row)
        road = roads[row["road_id"]]
        rows.append({**row, **trace_stats(trace, road, c)})
        theta, peak = road["theta"], road["peak"]
        speed, method = row["initial_speed"], row["method"]
        if method == "world" and row["seed"] == c["seeds"][0] and speed not in replay_checked:
            result, replay = replay_world(trace, theta, speed, c, gains)
            np.testing.assert_allclose(replay["states"], trace["states"], rtol=0, atol=1e-9)
            np.testing.assert_allclose(replay["actions"], trace["actions"], rtol=0, atol=1e-9)
            np.testing.assert_allclose(replay["reference"], trace["reference"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(result["distance_m"], row["distance_m"], rtol=0, atol=1e-9)
            replay_checked.add(speed)
        if method != "fixed":
            continue
        delayed_result, delayed_trace = simulate_closed(theta, speed, c, gains, selector=lambda _: peak)
        keys = {k: row[k] for k in ("road_id", "family", "initial_speed", "seed")}
        rows.append({**keys, "method": "oracle_delayed", **delayed_result, **trace_stats(delayed_trace, road, c)})
        persistent = LeastSquaresPeak(c)
        for step in range(c["history_steps"], min(len(trace["actions"]), max(fixed_probe_steps)+1)):
            if step % c["reference_period"] != 0:
                continue
            h = make_history(trace["states"], trace["actions"], step, c["history_steps"], c)
            continuous = persistent(h)
            if step not in fixed_probe_steps:
                continue
            reset = LeastSquaresPeak(c)
            fresh = reset(h)
            memories.append({**keys, "time_s": step*c["control_dt"], "true_peak": peak,
                             "history_slip_min": float(h[:, 2].min()), "history_slip_max": float(h[:, 2].max()),
                             "persistent": continuous, "fresh": fresh, "persistent_has_theta": persistent.theta is not None,
                             "fresh_has_theta": reset.theta is not None})
    return rows, memories, sorted(replay_checked)


def distance_pairs(c, d, rows):
    lookup = {(r["road_id"], r["initial_speed"], r["method"], r["seed"]): r for r in rows}
    roads = sorted({r["road_id"] for r in rows})
    rng = np.random.default_rng(d["seed"])
    output = []
    for method, baseline in (("world", "nls"), ("world", "direct"), ("world", "fixed"),
                             ("world", "oracle"), ("world", "oracle_delayed"), ("oracle_delayed", "oracle")):
        per_road = []
        excluded = 0
        for road in roads:
            deltas = []
            for speed, seed in itertools.product(c["initial_speeds"], c["seeds"] if method == "world" else [-1]):
                a = lookup[road, speed, method, seed]
                b = lookup[road, speed, baseline, seed if baseline == "direct" else -1]
                if a["stopped"] and b["stopped"]:
                    deltas.append(a["distance_m"]-b["distance_m"])
                else:
                    excluded += 1
            if deltas:
                per_road.append(np.mean(deltas))
        boots = rng.choice(per_road, (d["bootstrap_samples"], len(per_road))).mean(axis=1) if per_road else []
        output.append({"method": method, "baseline": baseline, "roads": len(per_road), "incomplete_pairs": excluded,
                       "delta_m": float(np.mean(per_road)) if per_road else None,
                       "ci_low_m": float(np.percentile(boots, 2.5)) if len(boots) else None,
                       "ci_high_m": float(np.percentile(boots, 97.5)) if len(boots) else None})
    return output
