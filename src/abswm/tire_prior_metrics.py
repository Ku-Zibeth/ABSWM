"""Window, action, and paired-road metrics for the physics prior study."""

from __future__ import annotations

import numpy as np

from .diagnostics import rmse
from .physics import slip
from .tire_diagnosis import bias_row, descriptors
from .tire_prior_data import ACTIONS, tire_function


def curve_data(c, d, suite, grid, values, exact=None, clipped=False, current=None):
    fn = tire_function(suite["roads"])
    true = fn(np.broadcast_to(grid, values.shape))
    near_zero = fn(np.full(len(values), 1e-6))/1e-6
    truth = descriptors(true, grid, stiffness=near_zero)
    predicted = descriptors(values, grid) if exact is None else descriptors(
        values, grid, height=exact[:, 0], peak=exact[:, 1], stiffness=exact[:, 2])
    scores = predicted.copy()
    if clipped:
        scores[:, 0] = scores[:, 0].clip(*d["height_range"])
        scores[:, 1] = scores[:, 1].clip(*d["peak_range"])
    hmax, hmin = suite["history"][..., 2].max(axis=1), suite["history"][..., 2].min(axis=1)
    current_slip = suite["history"][:, -1, 2]
    if current is None:
        current = interpolate(values, current_slip)
    endpoint = fn(scores[:, 1])
    regret = np.maximum(truth[:, 0]-endpoint, 0)/np.maximum(truth[:, 0], 1e-8)
    steps = values[:, 1:]-values[:, :-1]
    peak = predicted[:, 1, None]
    violates = np.any(((grid[1:] < peak-.015) & (steps < -.001)) |
                      ((grid[:-1] > peak+.015) & (steps > .001)), axis=1)
    outside = ((predicted[:, 0] > d["height_range"][1]) | (predicted[:, 0] < d["height_range"][0]) |
               (predicted[:, 1] > d["peak_range"][1]) | (predicted[:, 1] < d["peak_range"][0]))
    allowed = (values[:, 0].__abs__() < 1e-6) & (values.min(axis=1) >= -1e-6)
    errors = values-true
    consistency = np.abs(interpolate(values, scores[:, 1])-scores[:, 0])
    masks = {"low": grid < .05, "peak": true >= .95*truth[:, 0, None],
             "observed": (grid >= hmin[:, None]) & (grid <= hmax[:, None]),
             "unobserved": (grid < hmin[:, None]) | (grid > hmax[:, None]),
             "above": grid > hmax[:, None], "tail": grid > truth[:, 1, None]}
    return {"true": true, "predicted": values, "truth": truth, "physical": scores, "differences": scores-truth,
            "errors": errors, "masks": masks, "physical_violation": ~allowed | violates | outside,
            "shape_violation": violates, "outside_range": outside, "regret": regret,
            "force_error": (current-fn(current_slip))*c["mass"]*c["gravity"],
            "consistency": consistency, "max_slip": hmax}


def interpolate(curves, values):
    bins = curves.shape[1]-1
    fraction = np.clip(values, 0, 1)*bins
    left = fraction.astype(int).clip(max=bins-1)
    r = np.arange(len(curves))
    return curves[r, left]+(fraction-left)*(curves[r, left+1]-curves[r, left])


def suite_groups(suite, values):
    surface = np.array([r["surface"] for r in suite["roads"]])
    groups = {"all": np.ones(len(surface), dtype=bool), "hard": surface == "hard", "low": surface == "low"}
    if np.any(suite["caps"]):
        groups.update({f"target_{cap:g}": suite["caps"] == cap for cap in np.unique(suite["caps"])})
    for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .5), (.5, 1.001)):
        groups[f"observed_{lo:g}_{hi:g}"] = (values >= lo) & (values < hi)
    return {name: mask for name, mask in groups.items() if mask.any()}


def curve_rows(suite, info, keys):
    rows = []
    bias = []
    attributes = ("peak_mu", "peak_slip", "stiffness", "tail_ratio", "width95")
    for name, mask in suite_groups(suite, info["max_slip"]).items():
        e = info["differences"][mask]
        error = info["errors"][mask]
        r = info["regret"][mask]
        above_peak = e[:, 1] > 1e-6
        row = {**keys, "group": name, "n": int(mask.sum()), "curve_rmse": rmse(error),
               "point_force_rmse_n": rmse(info["force_error"][mask]), "regret": float(r.mean()),
               "within95": float((r <= .05).mean()),
               "postpeak_regret": float(r[above_peak].mean()) if above_peak.any() else None,
               "shape_violation": float(info["shape_violation"][mask].mean()),
               "outside_range": float(info["outside_range"][mask].mean()),
               "physical_violation": float(info["physical_violation"][mask].mean()),
               "peak_curve_inconsistency": float(info["consistency"][mask].mean())}
        for k, grid_mask in info["masks"].items():
            condition = np.broadcast_to(grid_mask, info["errors"].shape)[mask]
            row[f"{k}_rmse"] = rmse(error[condition])
        for j, feature in enumerate(attributes):
            stats = bias_row(e[:, j])
            bias.append({**keys, "group": name, "feature": feature, "n": int(mask.sum()), **stats})
            row[f"{feature}_mae"] = stats["mae"]
            row[f"{feature}_bias"] = stats["mean_bias"]
        rows.append(row)
    return rows, bias


def grid_forecast(curves, initial, actions, c):
    y = np.asarray(initial, dtype=float)
    dt = c["physics_dt"]
    z, mus = [], []
    for u in actions.T:
        for _ in range(round(c["control_dt"]/dt)):
            mu = interpolate(curves, slip(y, c["radius"]))
            dw = (c["radius"]*c["mass"]*c["gravity"]*mu-u)/c["inertia"]
            y = np.maximum(y+dt*np.column_stack((-c["gravity"]*mu, dw)), 0)
        z.append(y)
        mus.append(interpolate(curves, slip(y, c["radius"])))
    return np.stack(z, axis=1), np.stack(mus, axis=1)


def action_rows(c, d, suite, forecast, mu, truth, true_mu, keys):
    diff = forecast-truth
    lam, actual_slip = slip(forecast, c["radius"]), slip(truth, c["radius"])
    error = lam-actual_slip
    scale = c["mass"]*c["gravity"]
    out = []
    for name, mask in {"all": np.ones(len(lam), dtype=bool),
                       "hard": np.array([r["surface"] == "hard" for r in suite["roads"]]),
                       "low": np.array([r["surface"] == "low" for r in suite["roads"]])}.items():
        if not mask.any():
            continue
        for k in d["action_steps"]:
            at = k-1
            for j, branch in enumerate(ACTIONS):
                out.append({**keys, "group": name, "n": int(mask.sum()), "branch": branch, "step": k,
                            "v_rmse": rmse(diff[mask, j, at, 0]), "w_rmse": rmse(diff[mask, j, at, 1]),
                            "slip_rmse": rmse(error[mask, j, at]), "fx_rmse": rmse(scale*(mu-true_mu)[mask, j, at]),
                            "delta_v_rmse": rmse(diff[mask, j, at, 0]-diff[mask, 0, at, 0]),
                            "delta_w_rmse": rmse(diff[mask, j, at, 1]-diff[mask, 0, at, 1]),
                            "delta_slip_rmse": rmse(error[mask, j, at]-error[mask, 0, at]),
                            "delta_fx_rmse": rmse(scale*((mu-true_mu)[mask, j, at]-(mu-true_mu)[mask, 0, at]))})
    impulse_scale = c["control_dt"]*scale
    predicted_gain, true_gain = np.sum(mu, axis=-1)*impulse_scale, np.sum(true_mu, axis=-1)*impulse_scale
    expected = np.argmax(predicted_gain, axis=1)
    lost = true_gain.max(axis=1)-true_gain[np.arange(len(mu)), expected]
    rank = np.triu_indices(len(ACTIONS), k=1)
    actual = true_gain[:, rank[0]]-true_gain[:, rank[1]]
    compared = predicted_gain[:, rank[0]]-predicted_gain[:, rank[1]]
    meaningful = np.abs(actual) >= .5
    risk = (lam >= d["high_slip"]).any(axis=-1)
    true_risk = (actual_slip >= d["high_slip"]).any(axis=-1)
    aggregate = {**keys, "n": len(mu), "winner_within_half_ns": float((lost <= .5).mean()),
                 "impulse_regret_ns": float(lost.mean()), "rank_pairs": int(meaningful.sum()),
                 "rank_pair_accuracy": float((np.sign(actual[meaningful]) == np.sign(compared[meaningful])).mean()),
                 "high_slip_false_negative": float((~risk[true_risk]).mean()) if true_risk.any() else None,
                 "high_slip_false_positive": float(risk[~true_risk].mean()) if (~true_risk).any() else None}
    return out, aggregate, np.abs(diff[:, 1:, -1, 1]-diff[:, :1, -1, 1]).mean(axis=1)


def averages(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for _, items in groups.items():
        output = {k: items[0][k] for k in keys}
        for key in items[0]:
            if key in keys or key == "seed":
                continue
            a = [row[key] for row in items if isinstance(row[key], (int, float, np.number))]
            output[key] = float(np.mean(a)) if a else None
            if key != "n":
                output[key+"_seed_sd"] = float(np.std(a, ddof=1)) if len(a) > 1 else None
        summaries.append(output)
    return summaries


def paired(road_ids, outputs, seed, n):
    rng = np.random.default_rng(seed)
    ids = np.unique(road_ids)
    rows = []
    for (method, metric), sample in outputs.items():
        if method == "e4_a":
            continue
        bases = outputs["e4_a", metric]
        a = np.mean(sample, axis=0)
        b = np.mean(bases, axis=0)
        delta = [np.mean((a-b)[road_ids == i]) for i in ids]
        resampled = rng.choice(delta, (n, len(ids))).mean(axis=1)
        rows.append({"method": method, "baseline": "e4_a", "metric": metric, "roads": len(ids),
                     "delta": float(np.mean(delta)), "low95": float(np.percentile(resampled, 2.5)),
                     "high95": float(np.percentile(resampled, 97.5))})
    return rows
