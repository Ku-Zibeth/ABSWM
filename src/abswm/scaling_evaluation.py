"""Evaluation using the registered E4 A MPC candidates and score."""
from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
import torch

from .active_fast import static_forecast
from .active_metrics import ranking_rows
from .active_rollout import GRID, Forecast, candidate_actions, true_forecast, utilities
from .tire_prior_data import tire_function


VALUE_KEYS = ("vehicle", "wheel", "slip", "force")


def _values(y):
    return np.stack((y.state[..., 0], y.state[..., 1], y.slip, y.force), axis=-1)


def frozen_labels(histories, roads, c, horizons):
    return {n: (actions, true_forecast(roads, histories[:, -1, :2], actions, c))
            for n in horizons
            for actions in (candidate_actions(histories[:, -1, 3], n, c),)}


def _forecast_row(forecast, truth):
    prediction, actual = _values(forecast)[0], _values(truth)[0]
    error = (prediction-actual)**2
    contrasts = (prediction[1:]-prediction[:1])-(actual[1:]-actual[:1])
    return error.mean(axis=0), (contrasts**2).mean(axis=0)


def _truth_branch(truth, i):
    return Forecast(*(getattr(truth, key)[i:i+1] for key in ("state", "slip", "mu", "interval_mu", "max_slip", "min_wheel")),
                    truth.dt, truth.load)


@torch.inference_mode()
def evaluate(model, histories, roads, labels, c, options, tie_margin, *, method, seed, timing=False):
    short, long = options["horizon"], c["horizon_steps"]
    initial = histories[:, -1, :2]
    previous = histories[:, -1, 3]
    grid = torch.as_tensor(GRID, dtype=torch.float32, device=c["device"])
    true_curves = tire_function(roads)(np.broadcast_to(GRID, (len(roads), len(GRID))))
    ygrid = []
    marks = []
    for batch in torch.as_tensor(histories, dtype=torch.float32, device=c["device"]).split(c["batch_size"]):
        start = perf_counter()
        ygrid.append(model.curve(model.encode(batch), grid).cpu().numpy())
        if timing:
            marks.append({"component": "encoder_decoder_batch", "batch": len(batch), "ms": (perf_counter()-start)*1000})
    curves = np.concatenate(ygrid)
    rows = []
    for i, (road, curve) in enumerate(zip(roads, curves)):
        mark = perf_counter()
        action_short, true_short = labels[short]
        prediction = static_forecast(initial[i], action_short[i], curve, c)
        forecast, truth = prediction, _truth_branch(true_short, i)
        score = utilities(forecast, action_short[i], previous[i], options)[0]
        answer = utilities(truth, action_short[i], previous[i], options)[0]
        winner = int(np.argmax(score))
        high, lock = truth.high[0], truth.lock[0]
        pred_high, pred_lock = forecast.high[0], forecast.lock[0]
        result = ranking_rows(truth, forecast.interval_mu[0].mean(axis=-1), winner,
                              answer, high, lock, tie_margin)
        row = {"method": method, "seed": seed, "road_id": road["id"], "family": road["family"],
               "surface": road["subfamily"], "window": i, "slip_max": float(histories[i, :, 2].max()),
               "regime": "weak" if histories[i, :, 2].max() < .05 else "strong" if histories[i, :, 2].max() >= .20 else "intermediate",
               **result, "high_positive": int(high.sum()), "high_negative": int((~high).sum()),
               "high_missed": int((high & ~pred_high).sum()), "high_false": int((~high & pred_high).sum()),
               "lock_positive": int(lock.sum()), "lock_missed": int((lock & ~pred_lock).sum()),
               "candidate_count": len(high), "risk_positive": int((high | lock).sum()),
               "unsafe_among_predicted_feasible": int(((high | lock) & ~(pred_high | pred_lock)).sum()),
               "predicted_feasible": int((~(pred_high | pred_lock)).sum()),
               "curve_mse": float(np.mean((curve-true_curves[i])**2)),
               "peak_slip_abs": float(abs(GRID[np.argmax(curve)]-GRID[np.argmax(true_curves[i])])),
               "peak_mu_abs": float(abs(curve.max()-true_curves[i].max())),
               "friction_regret": float((true_curves[i].max()-true_curves[i, np.argmax(curve)])/true_curves[i].max())}
        short_error, short_response = _forecast_row(forecast, truth)
        row["action_wheel_mse"] = float(short_response[-1, 1])
        row["mpc_force_mse"] = float(short_error[-1, 3])
        action_long, true_long = labels[long]
        full = forecast if long == short else static_forecast(initial[i], action_long[i], curve, c)
        error, response = _forecast_row(full, _truth_branch(true_long, i))
        for n in (1, 5, 10, 20):
            m = round(n*c["control_dt"]*1000)
            for j, key in enumerate(VALUE_KEYS):
                row[f"{key}_mse_{m}ms"] = float(error[n-1, j])
                row[f"response_{key}_mse_{m}ms"] = float(response[n-1, j])
        rows.append(row)
        if timing:
            marks.append({"component": "both_candidate_rollouts_and_scoring", "batch": 1, "ms": (perf_counter()-mark)*1000})
    perf = []
    for component in dict.fromkeys(m["component"] for m in marks):
        times = [m["ms"] for m in marks if m["component"] == component]
        sizes = [m["batch"] for m in marks if m["component"] == component]
        perf.append({"method": method, "seed": seed, "component": component, "calls": len(times),
                     "rows": sum(sizes), "p50_ms": float(np.median(times)),
                     "p95_ms": float(np.percentile(times, 95)), "max_ms": float(np.max(times))})
    return rows, perf


def summary(rows):
    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    def ratio(num, den):
        n = sum(r[den] for r in rows)
        return sum(r[num] for r in rows)/n if n else None

    result = {"windows": len(rows), "road_clusters": len({r["road_id"] for r in rows}),
              "action_wheel_rmse": float(np.sqrt(mean("action_wheel_mse"))),
              "force_rmse": float(np.sqrt(mean("mpc_force_mse"))),
              "action_regret_ns": mean("pure_regret_ns"), "utility_regret": mean("utility_regret"),
              "risk_constrained_regret": mean("risk_constrained_regret"),
              "risk_constrained_n": sum(r["risk_constrained_regret"] is not None for r in rows),
              "ranking_accuracy": ratio("correct_pairs", "pairs"), "top1": mean("top1"),
              "high_fnr": ratio("high_missed", "high_positive"), "high_fpr": ratio("high_false", "high_negative"),
              "lock_fnr": ratio("lock_missed", "lock_positive"), "violation_rate": mean("violation"),
              "high_positive": sum(r["high_positive"] for r in rows),
              "high_negative": sum(r["high_negative"] for r in rows),
              "high_missed": sum(r["high_missed"] for r in rows),
              "unsafe_candidate_rate": ratio("risk_positive", "candidate_count"),
              "unsafe_among_predicted_feasible": ratio("unsafe_among_predicted_feasible", "predicted_feasible"),
              "curve_rmse": float(np.sqrt(mean("curve_mse"))), "peak_slip_mae": mean("peak_slip_abs"),
              "peak_mu_mae": mean("peak_mu_abs"), "friction_regret": mean("friction_regret")}
    for key in rows[0]:
        if "_mse_" in key:
            result[key.replace("_mse_", "_rmse_")] = float(np.sqrt(mean(key)))
    return result


def road_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["road_id"]].append(r)
    return groups


def paired_summary(left, right, repeat, seed):
    """Bootstrap road-level differences, retaining the paired seed/windows in each cluster."""
    a, b = road_groups(left), road_groups(right)
    keys = sorted(a.keys() & b.keys())
    estimates = [(summary(a[k]), summary(b[k])) for k in keys]
    metrics = ("action_wheel_rmse", "force_rmse", "action_regret_ns", "ranking_accuracy",
               "high_fnr", "high_fpr", "violation_rate")
    rng = np.random.default_rng(seed)
    result = []
    for metric in metrics:
        delta = np.array([p[metric]-q[metric] for p, q in estimates if p[metric] is not None and q[metric] is not None])
        if len(delta):
            draw = rng.integers(0, len(delta), (repeat, len(delta)))
            lo, hi = np.quantile(delta[draw].mean(axis=1), [.025, .975])
            worst = delta.min() if metric == "ranking_accuracy" else delta.max()
            result.append({"metric": metric, "road_clusters": len(delta), "delta": float(delta.mean()),
                           "road_delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else None,
                           "road_delta_median": float(np.median(delta)), "worst_road_delta": float(worst),
                           "lo95": float(lo), "hi95": float(hi)})
    return result
