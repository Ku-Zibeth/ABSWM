"""Counterfactual and measurement-feedback acceptance tests."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .config import write_json
from .data import load_split
from .diagnostics import chunks, rmse, tensor
from .evaluation import write_csv
from .physics import slip
from .recurrent_model import Posterior
from .recurrent_study import CANDIDATES, IMAGINED, OBSERVED, infer, prefix_cases, prepare, quality


def summarize_errors(c, d, cases, predicted, predicted_mu, method, seed, suite):
    error, lam, mu, delta, delta_mu = quality(c, cases, predicted, predicted_mu)
    actual = cases["states"]
    load = c["mass"]*c["gravity"]
    signed_truth = actual[:, 2:]-actual[:, 1:2]
    signed_predictions = predicted[:, 2:]-predicted[:, 1:2]
    pred_slip, true_slip = slip(predicted, c["radius"]), slip(actual, c["radius"])
    slip_delta = (pred_slip[:, 2:]-pred_slip[:, 1:2])-(true_slip[:, 2:]-true_slip[:, 1:2])
    fields = {"all": np.ones(len(actual), dtype=bool)}
    if "caps" in cases:
        fields.update({f"below_{cap:g}": cases["caps"] == cap for cap in np.unique(cases["caps"])})
    values, contrasts = [], []
    for subset, mask in fields.items():
        keys = {"suite": suite, "subset": subset, "method": method, "seed": seed, "windows": int(mask.sum())}
        for step in d["forecast_steps"]:
            k = step-1
            step_keys = {**keys, "step": step, "time_ms": round(1000*c["control_dt"]*step, 6)}
            for i, branch in enumerate(CANDIDATES):
                values.append({**step_keys, "branch": branch, "v_rmse": rmse(error[mask, i, k, 0]),
                               "w_rmse": rmse(error[mask, i, k, 1]), "slip_rmse": rmse(lam[mask, i, k]),
                               "fx_rmse_n": rmse(load*mu[mask, i, k])})
            for branch, branch_slice in [("all", slice(None)), *[(b, slice(i, i+1))
                                                                 for i, b in enumerate(CANDIDATES[2:])]]:
                truth = signed_truth[mask, branch_slice, k, 1]
                prediction = signed_predictions[mask, branch_slice, k, 1]
                informative = np.abs(truth) >= d["acceptance"]["min_direction_truth_rad_s"]
                direction = np.sign(truth[informative]) == np.sign(prediction[informative])
                contrasts.append({**step_keys, "branch_vs_hold": branch,
                                  "delta_v_rmse": rmse(delta[mask, branch_slice, k, 0]),
                                  "delta_w_rmse": rmse(delta[mask, branch_slice, k, 1]),
                                  "delta_slip_rmse": rmse(slip_delta[mask, branch_slice, k]),
                                  "delta_fx_rmse_n": rmse(load*delta_mu[mask, branch_slice, k]),
                                  "hold_only_delta_w_rmse": rmse(truth), "direction_cases": int(informative.sum()),
                                  "direction_accuracy": float(direction.mean()) if direction.size else None})
    return values, contrasts, np.abs(delta[:, :, -1, 1]).mean(axis=1)


@torch.inference_mode()
def feedback_test(c, d, model, cases, start, open_prediction, method, seed):
    k = d["reobserve_step"]
    x, actions = cases["states"][:, 0], cases["actions"][:, 0]
    revealed = np.concatenate((x[:, :k], slip(x[:, :k], c["radius"])[..., None], actions[:, :k, None]), axis=2)
    predicted = {"state_only": [], "state_and_latent": []}
    shifts = []
    for ids in chunks(np.arange(len(x)), d):
        memory = Posterior(start.z[ids], start.last_observation[ids])
        obs = tensor(revealed[ids], c)
        u = tensor(actions[ids], c)
        if model.transition is not None:
            _, _, imagined = model.predict(memory, u[:, :k])
            z_prior = imagined[:, -1]
        else:
            z_prior = memory.z
        if method == "e4_a":
            full = torch.cat((tensor(cases["history"][ids], c), obs), dim=1)[:, -c["history_steps"]:]
            posterior = Posterior(model.tire.encode(full), obs[:, -1])
        else:
            posterior = model.observe(obs, memory)
        shifts.extend(torch.linalg.vector_norm(posterior.z-z_prior, dim=1).cpu().numpy())
        for mode, context in (("state_only", Posterior(z_prior, obs[:, -1])), ("state_and_latent", posterior)):
            state, _, _ = model.predict(context, u[:, k:])
            predicted[mode].append(state[:, -1].cpu().numpy())
    estimates = {"open": open_prediction[:, 0, -1], **{k: np.concatenate(v) for k, v in predicted.items()}}
    rows = []
    for mode, state in estimates.items():
        error = state-x[:, -1]
        rows.append({"method": method, "seed": seed, "mode": mode, "observed_steps": 0 if mode == "open" else k,
                     "v_rmse": rmse(error[:, 0]), "w_rmse": rmse(error[:, 1]),
                     "slip_rmse": rmse(slip(state, c["radius"])-slip(x[:, -1], c["radius"])),
                     "average_posterior_change_l2": float(np.mean(shifts)) if mode == "state_and_latent" else None})
    return rows


def seed_means(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    averaged = []
    for group, members in groups.items():
        stats = {**dict(zip(keys, group)), "seeds": len(members)}
        for metric in metrics:
            sample = [r[metric] for r in members if r[metric] is not None]
            stats[metric] = float(np.mean(sample)) if sample else None
            stats[metric+"_seed_sd"] = float(np.std(sample, ddof=1)) if len(sample) > 1 else None
        averaged.append(stats)
    return averaged


def paired_action_errors(d, cases, error, seeds):
    roads = np.unique(cases["road_id"])
    rng = np.random.default_rng(d["seed"])
    rows = []
    for method, baseline in (("e4_a_stream", "e4_a"), (OBSERVED, "e4_a"), (OBSERVED, "e4_a_stream"),
                             (IMAGINED, "e4_a"), (IMAGINED, OBSERVED)):
        delta = np.mean([error[method, s]-error[baseline, s] for s in seeds], axis=0)
        clusters = np.asarray([np.mean(delta[cases["road_id"] == road]) for road in roads])
        draws = rng.choice(clusters, (d["bootstrap_samples"], len(roads)), replace=True).mean(axis=1)
        rows.append({"method": method, "baseline": baseline, "roads": len(roads),
                     "mean_endpoint_action_delta_mae_change_rad_s": float(clusters.mean()),
                     "ci95_low": float(np.percentile(draws, 2.5)), "ci95_high": float(np.percentile(draws, 97.5))})
    return rows


def acceptance(d, c, metrics, contrast, val):
    eligible = (OBSERVED, IMAGINED)
    scores = {name: float(np.mean([r["val_score"] for r in val if r["method"] == name])) for name in eligible}
    chosen = min(scores, key=scores.get)
    keys = ("v_rmse", "w_rmse", "slip_rmse", "fx_rmse_n")
    end = [r for r in metrics if r["suite"] == "matched" and r["subset"] == "all"
           and r["branch"] == "factual" and r["step"] == c["horizon_steps"]]
    action = [r for r in contrast if r["suite"] == "matched" and r["subset"] == "all"
              and r["branch_vs_hold"] == "all" and r["step"] == c["horizon_steps"]]
    fixed = next(r for r in end if r["method"] == "e4_a")
    fixed_action = next(r for r in action if r["method"] == "e4_a")["delta_w_rmse"]
    rows = []
    for name in ("e4_a_stream", *eligible):
        candidate = next(r for r in end if r["method"] == name)
        changed = {key: candidate[key]/fixed[key]-1 for key in keys}
        delta = next(r for r in action if r["method"] == name)["delta_w_rmse"]
        gain = 1-delta/fixed_action
        guard = max(changed.values()) <= d["acceptance"]["max_factual_relative_regression"]
        passes = gain >= d["acceptance"]["min_wheel_action_delta_relative_gain"] and guard
        rows.append({"method": name, "relative_wheel_action_delta_gain": gain, "relative_factual_changes": changed,
                     "passes_factual_guard": bool(guard), "passes": bool(passes)})
    return {"registered_reference": "e4_a", "validation_selected_candidate": chosen, "validation_scores": scores,
            "criteria": d["acceptance"], "endpoint_ms": c["horizon_steps"]*c["control_dt"]*1000,
            "selected_candidate_passes": next(r["passes"] for r in rows if r["method"] == chosen),
            "candidates": rows}


@torch.inference_mode()
def timing(c, model, cases, method, seed):
    windows = tensor(cases["history"][:32], c)
    candidate = tensor(cases["actions"][:32, 3], c)
    times = []
    for i in range(len(windows)):
        h = windows[i:i+1]
        earlier = model.observe(h[:, :-1])
        start = perf_counter()
        if method == "e4_a":
            posterior = Posterior(model.tire.encode(h), h[:, -1])
        else:
            posterior = model.observe(h[:, -1:], earlier)
        update = perf_counter()-start
        start = perf_counter()
        model.predict(posterior, candidate[i:i+1])
        if c["device"] != "cpu":
            torch.cuda.synchronize()
        times.append((1000*update, 1000*(perf_counter()-start)))
    update, rollout = np.percentile(times[4:], 95, axis=0)
    return {"method": method, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
            "update_p95_ms": update, "rollout_p95_ms": rollout}


def make_figure(path, forecasts, contrast):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"e4_a": "E4 A · fixed window", "e4_a_stream": "E4 A · streaming",
              OBSERVED: "Recurrent observer", IMAGINED: "Recurrent transition"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    panels = ((forecasts, "w_rmse", "Factual wheel speed RMSE [rad/s]"),
              (forecasts, "fx_rmse_n", "Factual tire force RMSE [N]"),
              (contrast, "delta_w_rmse", "Wheel speed response vs hold RMSE [rad/s]"),
              (contrast, "delta_fx_rmse_n", "Tire force response vs hold RMSE [N]"))
    for ax, (rows, field, ylabel) in zip(axes.flat, panels):
        for name, label in labels.items():
            subset = [r for r in rows if r["method"] == name and r["suite"] == "matched" and r["subset"] == "all"
                      and (r.get("branch") == "factual" or r.get("branch_vs_hold") == "all")]
            ax.plot([r["time_ms"] for r in subset], [r[field] for r in subset], "o-", label=label)
        ax.set(xlabel="Prediction horizon [ms]", ylabel=ylabel)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(c, d, base, roads, models, validation, output):
    start = perf_counter()
    test = prepare(c, d, base, "test", roads)
    prefix = prefix_cases(c, d, load_split(base/"data", "test"), roads)
    values, actions, feedback, timings, paired = [], [], [], [], {}
    for (method, seed), model in models.items():
        for suite, cases in (("matched", test), ("prefix", prefix)):
            print(f"evaluate {method} seed={seed} {suite}", flush=True)
            prediction, mu, posterior = infer(c, d, model, cases, persistent=method != "e4_a")
            rows, contrast, errors = summarize_errors(c, d, cases, prediction, mu, method, seed, suite)
            values.extend(rows)
            actions.extend(contrast)
            if suite == "matched":
                paired[method, seed] = errors
                feedback.extend(feedback_test(c, d, model, cases, posterior, prediction, method, seed))
                timings.append(timing(c, model, cases, method, seed))
    common = ("suite", "subset", "method", "windows", "step", "time_ms")
    means = seed_means(values, (*common, "branch"), ("v_rmse", "w_rmse", "slip_rmse", "fx_rmse_n"))
    contrast = seed_means(actions, (*common, "branch_vs_hold"), ("delta_v_rmse", "delta_w_rmse", "delta_slip_rmse",
                                                                "delta_fx_rmse_n", "hold_only_delta_w_rmse",
                                                                "direction_cases", "direction_accuracy"))
    write_csv(output/"forecast_by_seed.csv", values)
    write_csv(output/"forecast_comparison.csv", means)
    write_csv(output/"action_contrast_by_seed.csv", actions)
    write_csv(output/"action_contrast.csv", contrast)
    write_csv(output/"feedback.csv", feedback)
    write_csv(output/"timing.csv", timings)
    write_csv(output/"paired_action_errors.csv", paired_action_errors(d, test, paired, c["seeds"]))
    gate = acceptance(d, c, means, contrast, validation)
    folder = output/"figures"
    folder.mkdir(exist_ok=True)
    make_figure(folder/"recurrent_forecasts.png", means, contrast)
    write_json(output/"acceptance.json", {**gate, "matched_windows": len(test["history"]),
                                           "prefix_windows": {str(cap): int((prefix["caps"] == cap).sum())
                                                              for cap in d["prefix_slip_caps"]},
                                           "evaluation_seconds": perf_counter()-start})
    print(f"Recurrent forecasts: {output}", flush=True)
