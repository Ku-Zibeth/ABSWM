"""Probability scores, paired control comparisons and pre-registered acceptance."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.special import logsumexp, ndtr

from .active_information import log_weights


LEVELS = (.50, .80, .90, .95)
VARIABLES = ("velocity_m_s", "wheel_rad_s", "slip", "force_n")


def values_for(forecast):
    return np.stack((forecast.state[..., 0], forecast.state[..., 1], forecast.slip, forecast.force), axis=-1)


def gaussian_abs(delta, sigma):
    standardized = delta/sigma
    return sigma*np.sqrt(2/np.pi)*np.exp(-.5*standardized**2)+delta*(2*ndtr(standardized)-1)


def mixture_scores(predicted, truth, weights, sigma):
    """Marginal proper scores and central intervals for a Gaussian mixture."""
    k = len(weights)
    sigma = np.asarray(sigma)
    residual = truth[None]-predicted
    log_component = -.5*(residual/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))
    w = weights.reshape((k,)+(1,)*(predicted.ndim-1))
    nll = -logsumexp(log_component+log_weights(w), axis=0)
    crps = (w*gaussian_abs(residual, sigma)).sum(axis=0)
    crps -= .5*(w[:, None]*w[None, :]*gaussian_abs(predicted[:, None]-predicted[None, :],
                                                    np.sqrt(2)*sigma)).sum(axis=(0, 1))
    q = (1-np.array(LEVELS))/2
    targets = np.concatenate((q, 1-q))
    lo = np.broadcast_to(predicted.min(axis=0)-8*sigma, (len(targets), *truth.shape)).copy()
    hi = np.broadcast_to(predicted.max(axis=0)+8*sigma, lo.shape).copy()
    targets = targets.reshape((-1,)+(1,)*truth.ndim)
    for _ in range(24):
        x = (lo+hi)/2
        cdf = (w[None]*ndtr((x[:, None]-predicted[None])/sigma)).sum(axis=1)
        too_small = cdf < targets
        lo = np.where(too_small, x, lo)
        hi = np.where(too_small, hi, x)
    z = (lo+hi)/2
    low, high = np.split(z, 2, axis=0)
    return nll, crps, (low <= truth) & (truth <= high), high-low


def ranking_rows(truth, predicted, chosen, oracle_score, high_true, lock_true, tie_margin):
    reward_truth = truth.interval_mu[0].mean(axis=-1)
    reward = predicted
    a, b = np.triu_indices(len(reward_truth), k=1)
    difference = reward_truth[a]-reward_truth[b]
    mask = np.abs(difference) > tie_margin
    matches = np.sign(reward[a]-reward[b]) == np.sign(difference)
    feasible = ~(high_true | lock_true)
    best = int(np.argmax(oracle_score))
    valid = bool(feasible[chosen])
    feasible_best = np.max(oracle_score[feasible]) if feasible.any() else None
    reward_best = reward_truth.max()
    selected_reward = reward_truth[chosen]
    return {"pairs": int(mask.sum()), "correct_pairs": int((matches & mask).sum()),
            "top1": chosen == best, "top1_reward_tie": bool(reward_best-selected_reward <= tie_margin),
            "chosen": chosen, "oracle": best, "pure_regret_ns": float(truth.impulse.max()-truth.impulse[0, chosen]),
            "utility_regret": float(oracle_score.max()-oracle_score[chosen]),
            "risk_constrained_regret": float(feasible_best-oracle_score[chosen]) if valid else None,
            "oracle_feasible": bool(feasible.any()), "violation": not valid}


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def mean_rows(rows, keys, values):
    result = []
    for key, group in grouped(rows, keys).items():
        summary = dict(zip(keys, key)) | {"n": len(group)}
        for name in values:
            vals = [row[name] for row in group if row.get(name) is not None and np.isfinite(row[name])]
            summary[name] = float(np.mean(vals)) if vals else None
        result.append(summary)
    return result


def risk_summary(rows):
    result = []
    keys = ("suite", "method", "event")
    for key, group in grouped(rows, keys).items():
        p = np.array([row["probability"] for row in group])
        y = np.array([row["observed"] for row in group], dtype=bool)
        predicted = p >= .5
        result.append(dict(zip(keys, key)) | {"branches": len(y), "positives": int(y.sum()),
                      "brier": float(np.mean((p-y)**2)), "fnr": float((~predicted & y).sum()/y.sum()) if y.any() else None,
                      "fpr": float((predicted & ~y).sum()/(~y).sum()) if (~y).any() else None,
                      "accuracy": float(np.mean(predicted == y))})
    return result


def risk_bins(rows):
    result = []
    for key, group in grouped(rows, ("suite", "method", "event")).items():
        p = np.array([row["probability"] for row in group])
        y = np.array([row["observed"] for row in group])
        index = np.minimum((p*10).astype(int), 9)
        for i in np.unique(index):
            mask = index == i
            result.append(dict(zip(("suite", "method", "event"), key)) |
                          {"bin": i/10, "n": int(mask.sum()), "p_mean": float(p[mask].mean()),
                           "observed_rate": float(y[mask].mean())})
    return result


def control_summary(rows, keys=("method",)):
    metrics = ("distance_m", "time_s", "decel_m_s2", "impulse_ns", "friction_utilization", "high_slip_s",
               "lock_s", "torque_variation_nm", "safety_count", "exploration_count", "modification_count",
               "end_speed")
    result = []
    for key, group in grouped(rows, keys).items():
        fail = [row for row in group if not row["stopped"]]
        data = dict(zip(keys, key)) | {"n": len(group), "roads": len({r["road_id"] for r in group}),
                                     "failed": len(fail), "failure_rate": len(fail)/len(group)}
        for name in metrics:
            x = np.array([r[name] for r in group])
            data[name] = float(x.mean())
            data[f"{name}_sd"] = float(x.std(ddof=1) if len(x) > 1 else 0.)
            data[f"{name}_median"] = float(np.median(x))
            data[f"{name}_min"] = float(x.min())
            data[f"{name}_max"] = float(x.max())
        success = [r["distance_m"] for r in group if r["stopped"]]
        data["distance_to_2_m_s_success_only"] = float(np.mean(success)) if success else None
        data["worst_road"] = max(group, key=lambda r: (not r["stopped"], r["distance_m"]))["road_id"]
        result.append(data)
    return result


def paired_control(rows, samples, seed, contrasts):
    keys = ("road_id", "initial_speed", "initial_slip", "initial_torque")
    rng = np.random.default_rng(seed)
    result = []
    for left, right in contrasts:
        a = grouped([r for r in rows if r["method"] == left], keys)
        b = grouped([r for r in rows if r["method"] == right], keys)
        common = sorted(a.keys() & b.keys())
        if not common:
            continue
        for metric in ("distance_m", "time_s", "high_slip_s", "lock_s", "stopped"):
            groups = defaultdict(list)
            for key in common:
                x, y = a[key], b[key]
                bx, by = {r["seed"]: r[metric] for r in x}, {r["seed"]: r[metric] for r in y}
                paired = sorted(bx.keys() & by.keys())
                if paired:
                    delta = np.mean([float(bx[i])-float(by[i]) for i in paired])
                else:
                    delta = np.mean(list(bx.values()))-np.mean(list(by.values()))
                groups[key[0]].append(delta)
            value = np.array([np.mean(v) for v in groups.values()])
            reweighted = value[rng.integers(len(value), size=(samples, len(value)))].mean(axis=1)
            lo, hi = np.quantile(reweighted, [.025, .975])
            result.append({"method": left, "baseline": right, "metric": metric, "road_clusters": len(value),
                           "initial_pairs": len(common), "mean_delta": float(value.mean()), "sd": float(value.std()),
                           "median": float(np.median(value)), "worst_delta": float(value.max()),
                           "lo95": float(lo), "hi95": float(hi)})
    return result
