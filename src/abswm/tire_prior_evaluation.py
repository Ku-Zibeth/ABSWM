"""Held-out evaluation and online refinement experiments for P6–P7."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .config import write_json
from .controllers import LeastSquaresPeak
from .decoder_study import CurveDecoder, read_json
from .diagnostic_experiments import flat_pchip, midpoint_observations
from .diagnostics import chunks, ridge_predictions, rmse, tensor
from .evaluation import load_models, write_csv
from .models import seed_everything
from .physics import TYPICAL_ROADS, friction, slip
from .recurrent_model import Posterior, RecurrentTireModel
from .recurrent_study import stream_posteriors
from .tire_prior import StructuredTire, fixed_forecast, friction_query, history_evidence, history_residual, refine
from .tire_prior_data import ACTIONS, action_suite, stress_suite
from .tire_prior_metrics import action_rows, averages, curve_data, curve_rows, grid_forecast, paired
from .tire_prior_training import existing_suite, histories


def subset(suite, indices):
    return {**suite, **{name: suite[name][indices] for name in ("history", "locators", "caps")},
            "roads": [suite["roads"][i] for i in indices]}


def context(c, model, suite, recurrent=None):
    if recurrent is not None:
        return stream_posteriors(recurrent, suite, c).z.clone()
    with torch.no_grad():
        return torch.cat([model.encode(tensor(h, c)) for h in chunks(suite["history"], c)])


@torch.no_grad()
def curves(c, model, z, history, grid):
    predicted, exact, current = [], [], []
    query = tensor(grid, c)
    for ids in chunks(np.arange(len(z)), c):
        state = z[ids]
        predicted.append(friction_query(model, state, query.expand(len(ids), -1)).cpu().numpy())
        measured = tensor(history[ids, -1:, 2], c)
        current.append(friction_query(model, state, measured)[:, 0].cpu().numpy())
        if isinstance(model, StructuredTire):
            exact.append(model.physical(model.parameters_for(state)).cpu().numpy())
    return np.concatenate(predicted), np.concatenate(exact) if exact else None, np.concatenate(current)


def traditional(c, suite, grid, kind):
    out = []
    for h in suite["history"]:
        curve = None
        if kind == "nls":
            estimator = LeastSquaresPeak(c)
            estimator(h)
            theta = estimator.theta if estimator.theta is not None else TYPICAL_ROADS["wet"]
            curve = friction(grid, theta)
        elif kind == "pchip":
            lam, mu = midpoint_observations(h, c)
            curve = flat_pchip(lam, mu, grid)
        out.append(curve if curve is not None else friction(grid, TYPICAL_ROADS["wet"]))
    return np.stack(out)


def refine_batch(c, d, model, z, observed, checkpoints, prior=None, warm=None):
    stages = {k: [[], [], []] for k in checkpoints}
    seconds = []
    for indices in chunks(np.arange(len(z)), c):
        ev = history_evidence(tensor(observed[indices], c), c)
        start = perf_counter()
        result = refine(model, z[indices], ev, d, checkpoints,
                        prior=prior, warm=warm[indices] if warm is not None else None)
        seconds.append(perf_counter()-start)
        for k, values in result.items():
            if k in stages:
                for dst, src in zip(stages[k], values):
                    dst.append(src)
    return {k: tuple(torch.cat(computed) for computed in values) for k, values in stages.items()}, sum(seconds)


def warm_refinement(c, d, model, z, suite, steps, history_size):
    raw_history = histories(suite, history_size, c)
    result = z.clone()
    previous = {}
    for time in np.unique(suite["caps"]):
        batch = np.flatnonzero(suite["caps"] == time)
        traces = suite["locators"][batch, 0]
        warm = torch.stack([previous.get(i, z[j]) for j, i in zip(batch, traces)])
        refined, _ = refine_batch(c, d, model, z[batch], raw_history[batch], [steps], warm=warm)
        result[batch] = refined[steps][0]
        previous.update(zip(traces, result[batch]))
    return result


def forecast(c, model, z, h, actions, recurrent=None, discrete=None):
    b, j, period = actions.shape
    if discrete is not None:
        x, mu = grid_forecast(np.repeat(discrete, j, axis=0), np.repeat(h[:, -1, :2], j, axis=0),
                              actions.reshape(-1, period), c)
        return x.reshape(b, j, period, 2), mu.reshape(b, j, period)
    predicted, force = [], []
    with torch.no_grad():
        for ids in chunks(np.arange(b), {**c, "batch_size": 64}):
            local = z[ids].repeat_interleave(j, dim=0)
            latest = tensor(h[ids, -1], c).repeat_interleave(j, dim=0)
            u = tensor(actions[ids], c).reshape(-1, period)
            if recurrent is None or recurrent.transition is None:
                states, mu = fixed_forecast(model, local, latest[:, :2], u, c)
            else:
                states, mu, _ = recurrent.predict(Posterior(local, latest), u)
            predicted.append(states.cpu().numpy().reshape(len(ids), j, period, 2))
            force.append(mu.cpu().numpy().reshape(len(ids), j, period))
    return np.concatenate(predicted), np.concatenate(force)


def score_selection(c, d, val, models, fitted, rows, prior, grid, output):
    candidates = ("structured_frozen", "structured_joint", "structured_prior", "history_midpoint",
                  "history_integral", "history_wheel", "history_joint")
    candidate_scores = {name: float(np.mean([r["val_score"] for r in rows if r["method"] == name])) for name in candidates}
    best = min(candidate_scores, key=candidate_scores.get)
    optimization = []
    n = min(192, len(val["history"]))
    evaluation = subset(val, np.arange(n))
    for seed in c["seeds"]:
        model = fitted[best, seed]
        z = context(c, model, evaluation)
        for length in d["refinement_histories"]:
            result, seconds = refine_batch(c, d, model, z, histories(evaluation, length, c), d["refinement_steps"])
            for k, (latent, residual, rejects) in result.items():
                prediction, exact, point = curves(c, model, latent, evaluation["history"], grid)
                info = curve_data(c, d, evaluation, grid, prediction, exact, current=point)
                error = rmse(info["errors"])
                force = rmse(info["force_error"])/(c["mass"]*c["gravity"])
                score = error+.35*force+.30*info["regret"].mean()
                optimization.append({"seed": seed, "steps": k, "history": length, "val_score": score,
                                     "curve_rmse": error, "point_mu_rmse": force, "regret": info["regret"].mean(),
                                     "history_rmse": float(residual.mean().sqrt()), "batched_seconds": seconds,
                                     "rejected_updates": float(rejects.float().mean())})
    write_csv(output/"refinement_validation.csv", optimization)
    options = averages(optimization, ("steps", "history"))
    selection = min([r for r in options if r["steps"] > 0], key=lambda r: r["val_score"])
    decisions = {"selected_structured": best, "candidate_validation": candidate_scores,
                 "refinement_steps": selection["steps"], "refinement_history": selection["history"], "gate": d["gate"]}
    # Persist the complete policy before opening the independent evaluation suites.
    write_json(output/"validation_policy.json", decisions)
    return decisions


def baselines(c, base, reference, recurrent, models):
    existing = load_models(c, base)
    study = read_json(reference/"experiment.json")["study_config"]
    options = read_json(recurrent/"experiment.json")["options"]
    results = {}
    for seed, e4 in models.items():
        world = existing["world", seed]
        e3 = CurveDecoder(c, study, world).to(c["device"])
        e3.load_state_dict(torch.load(reference/"models"/f"e3_{seed}.pt", weights_only=True)["state_dict"])
        rec = RecurrentTireModel(e4, options, imagine=True).to(c["device"])
        payload = torch.load(recurrent/"models"/f"recurrent_transition_{seed}.pt", weights_only=True)
        rec.load_state_dict(payload["state_dict"])
        for method, model in (("world", world), ("e3", e3), ("e4_a", e4), ("e4_clipped", e4),
                              ("recurrent_transition", rec.tire)):
            observer = rec.eval().requires_grad_(False) if method == "recurrent_transition" else None
            results[method, seed] = (model.eval().requires_grad_(False), observer)
    return results


def refinement_rows(c, d, suite, name, seed, model, z, grid, original, checkpoints, length):
    output = []
    observed = histories(suite, length, c)
    results, seconds = refine_batch(c, d, model, z, observed, checkpoints)
    for k, (latent, residual, rejects) in results.items():
        pred, physical, point = curves(c, model, latent, suite["history"], grid)
        current = curve_data(c, d, suite, grid, pred, physical, current=point)
        max_slip = suite["history"][..., 2].max(axis=1)
        for group, mask in {"all": max_slip >= 0, "weak_slip": max_slip < .025, "higher_slip": max_slip >= .10}.items():
            if mask.any():
                out = {"suite": name, "seed": seed, "history": length, "iterations": k, "group": group,
                       "n": int(mask.sum()), "history_rmse": float(residual[mask].mean().sqrt()),
                       "curve_rmse": rmse(current["errors"][mask]),
                       "above_rmse": rmse(current["errors"][mask][current["masks"]["above"][mask]]),
                       "peak_mae": np.abs(current["differences"][mask, 1]).mean(),
                       "height_mae": np.abs(current["differences"][mask, 0]).mean(),
                       "regret": current["regret"][mask].mean(),
                       "point_force_rmse": rmse(current["force_error"][mask]),
                       "latent_change_l2": float((latent-z)[mask].norm(dim=1).mean()),
                       "curve_change_rmse": rmse((pred-original)[mask]), "rejections": float(rejects[mask].float().mean()),
                       "batched_seconds_all_checkpoints": seconds}
                output.append(out)
    return output


def ambiguity(c, d, model, z, suite, steps, seed, grid):
    observed = histories(suite, c["history_steps"], c)
    ev = history_evidence(tensor(observed, c), c)
    direction = tensor(np.random.default_rng(seed).standard_normal(z.shape), c)
    perturb = .7*direction/direction.norm(dim=1, keepdim=True)
    a = refine(model, z, ev, d, [steps], warm=z+perturb)[steps]
    b = refine(model, z, ev, d, [steps], warm=z-perturb)[steps]
    aa, pa, _ = curves(c, model, a[0], suite["history"], grid)
    bb, pb, _ = curves(c, model, b[0], suite["history"], grid)
    ra, rb = a[1].sqrt().cpu().numpy(), b[1].sqrt().cpu().numpy()
    max_slip = suite["history"][..., 2].max(axis=1)
    similar = np.abs(ra-rb) < .003
    return [{"seed": seed, "coverage": group, "windows": int(mask.sum()),
             "usable_windows": int(ev.usable.cpu().numpy()[mask].sum()),
             "similar_residual_windows": int((mask & similar).sum()),
             "curve_disagreement_rmse": rmse((aa-bb)[mask & similar]),
             "peak_disagreement_mae": float(np.abs(pa[mask & similar, 1]-pb[mask & similar, 1]).mean())
             if (mask & similar).any() else None}
            for group, mask in {"weak": max_slip < .025, "strong": max_slip >= .10}.items() if mask.any()]


def efficiency(c, d, name, seed, model, observer, z, suite, size, iterations):
    h = tensor(suite["history"][:15], c)
    longer = tensor(histories(subset(suite, np.arange(len(h))), size, c), c)
    z = z[:len(h)]
    gr = tensor(np.linspace(0, 1, d["eval_grid"]), c)[None]
    actions = torch.zeros((1, c["horizon_steps"]), device=c["device"])
    rows = []
    for i in range(len(h)):
        sample = h[i:i+1]
        with torch.no_grad():
            start = perf_counter()
            model.encode(sample)
            enc = perf_counter()-start
            start = perf_counter()
            friction_query(model, z[i:i+1], gr)
            query = perf_counter()-start
            start = perf_counter()
            if observer is not None and observer.transition is not None:
                observer.predict(Posterior(z[i:i+1], sample[:, -1]), actions)
            else:
                fixed_forecast(model, z[i:i+1], sample[:, -1, :2], actions, c)
            roll = perf_counter()-start
        if iterations:
            start = perf_counter()
            refine(model, z[i:i+1], history_evidence(longer[i:i+1], c), d, [iterations])
            online = perf_counter()-start
        else:
            online = 0.
        rows.append((enc, query, roll, online))
    result = np.percentile(rows[3:], 95, axis=0)*1000
    return {"method": name, "seed": seed, "parameters": sum(p.numel() for p in (observer or model).parameters()),
            "refinement_steps": iterations, "encoder_p95_ms": result[0], "curve_p95_ms": result[1],
            "rollout_p95_ms": result[2], "refine_p95_ms": result[3]}


def feedback_test(c, d, model, observer, z, suite, length, iterations, name, seed):
    n = min(128, len(suite["history"]))
    suite = subset(suite, np.arange(n))
    z = z[:n]
    k, horizon = d["reobserve_steps"], c["horizon_steps"]
    raw = suite["raw"]
    actions = np.stack([raw["actions"][i, t:t+horizon] for i, t in suite["locators"]])
    states = np.stack([raw["states"][i, t+1:t+horizon+1] for i, t in suite["locators"]])
    hist = suite["history"]
    imagined, _ = forecast(c, model, z, hist, actions[:, None], observer)
    new = np.concatenate((states[:, :k], slip(states[:, :k], c["radius"])[..., None], actions[:, :k, None]), axis=-1)
    combined = np.concatenate((histories(suite, length, c), new), axis=1)
    if observer is not None:
        memory = stream_posteriors(observer, suite, c)
        with torch.no_grad():
            if observer.transition is None:
                prior = z
            else:
                _, _, imagined_z = observer.predict(Posterior(z, memory.last_observation), tensor(actions[:, :k], c))
                prior = imagined_z[:, -1].clone()
            fresh = observer.observe(tensor(new, c), memory).z
    else:
        prior = z
        with torch.no_grad():
            fresh = model.encode(tensor(combined[:, -c["history_steps"]:], c))
    if iterations:
        update, _ = refine_batch(c, d, model, fresh, combined[:, -length:], [iterations])
        fresh = update[iterations][0]
    rows = []
    estimates = {"open": imagined[:, 0, -1]}
    for tag, posterior in (("state_only", prior), ("state_and_latent", fresh)):
        pred, _ = forecast(c, model, posterior, new, actions[:, None, k:], observer)
        estimates[tag] = pred[:, 0, -1]
    for mode, x in estimates.items():
        error = x-states[:, -1]
        rows.append({"method": name, "seed": seed, "mode": mode, "n": n, "v_rmse": rmse(error[..., 0]),
                     "w_rmse": rmse(error[..., 1]), "slip_rmse": rmse(slip(x, c["radius"])-slip(states[:, -1], c["radius"]))})
    return rows


def conditional_prior(output, prior, c):
    sample = (prior.center+prior.samples*prior.scale).cpu().numpy()
    rows = []
    bandwidth = prior.bandwidth*prior.scale.cpu().numpy()
    for height in (.25, .7, .9, 1.1):
        for stiffness in (None, 20., 30.):
            kernel = ((sample[:, 0]-height)/bandwidth[0])**2
            if stiffness is not None:
                kernel += ((sample[:, 2]-stiffness)/bandwidth[2])**2
            w = np.exp(-.5*(kernel-kernel.min()))
            w /= w.sum()
            order = np.argsort(sample[:, 1])
            range_ = np.interp([.05, .5, .95], np.cumsum(w[order]), sample[order, 1])
            rows.append({"height": height, "stiffness": stiffness, "peak_q05": range_[0], "peak_median": range_[1],
                         "peak_q95": range_[2], "effective_training_roads": 1/np.square(w).sum()})
    write_csv(output/"conditional_prior.csv", rows)


def latent_probe(c, best, fitted, models, val, test, grid, output):
    attributes = ("height", "peak", "stiffness", "tail_ratio", "width95")
    from .tire_diagnosis import descriptors
    from .tire_prior_data import tire_function

    def targets(suite):
        fn = tire_function(suite["roads"])
        return descriptors(fn(np.broadcast_to(grid, (len(suite["history"]), len(grid)))), grid,
                           stiffness=fn(np.full(len(suite["history"]), 1e-6))/1e-6)

    y, truth = targets(val), targets(test)
    rows = []
    for seed in c["seeds"]:
        for name, model in (("e4_a", models[seed]), (best, fitted[best, seed])):
            train_z = context(c, model, val).cpu().numpy().astype(float)
            test_z = context(c, model, test).cpu().numpy().astype(float)
            prediction = ridge_predictions(train_z, test_z, y, .01)
            for j, feature in enumerate(attributes):
                error = prediction[:, j]-truth[:, j]
                denom = ((truth[:, j]-truth[:, j].mean())**2).sum()
                rows.append({"method": name, "seed": seed, "feature": feature, "r2": float(1-(error**2).sum()/denom),
                             "mae": float(np.abs(error).mean())})
    write_csv(output/"latent_probes.csv", rows)


def plots(output, d, grid, retained, refined_rows, regressions, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output/"figures"
    best = names[0]
    colors = {"e4_a": "#64748b", best: "#15688c", "structured_refine": "#ad4a30", "nls": "#56914e",
              "structured_mixed": "#ad4a30"}
    for suite, file in (("ab", "ab_curves.png"), ("decoupled", "stress_curves.png"),
                        ("withheld_rational", "withheld_curves.png")):
        case, results = retained[suite]
        road, caps = case["roads"], case["caps"]
        groups = list(dict.fromkeys(r["id"] for r in road))[:2]
        fig, axes = plt.subplots(len(groups), len(d["coverage_caps"]), figsize=(12, 6), layout="constrained",
                                 sharex=True, sharey=True, squeeze=False)
        for (i, j), ax in np.ndenumerate(axes):
            cap = d["coverage_caps"][j]
            ids = [k for k, r in enumerate(road) if r["id"] == groups[i] and caps[k] == cap]
            if not ids:
                ax.set_visible(False)
                continue
            k = ids[0]
            ax.plot(grid, results["true"][k], color="black", linewidth=2, label="True")
            h = case["history"][k, :, 2]
            ax.axvspan(h.min(), h.max(), color="0.88")
            for method in ("e4_a", best, "structured_refine", "nls"):
                if method in results:
                    ax.plot(grid, results[method][k], label=method, color=colors[method])
            ax.set(title=f"{road[k]['subfamily']} · {cap:.0%}", xlabel="Slip" if i else "",
                   ylabel="Friction" if j == 0 else "")
        axes[0, 0].legend(fontsize=6)
        fig.savefig(folder/file, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for length in d["refinement_histories"]:
        rows = [r for r in refined_rows if r["suite"] == "independent_burckhardt" and r["history"] == length and r["group"] == "all"]
        for ax, key, ylabel in zip(axes, ("history_rmse", "curve_rmse", "above_rmse"),
                                   ("History residual", "Full curve RMSE", "Above observed slip RMSE")):
            ax.plot([r["iterations"] for r in rows], [r[key] for r in rows], "o-", label=f"{length} rows")
            ax.set(xlabel="Refinement iterations", ylabel=ylabel)
    axes[0].legend()
    fig.savefig(folder/"refinement.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, branch in zip(axes, ("increase", "pulse")):
        for method in ("e4_a", best, "structured_refine", "structured_recurrent_refine"):
            rows = [r for r in regressions if r["suite"] == "independent_burckhardt" and r["branch"] == branch
                    and r["group"] == "all" and r["method"] == method]
            ax.plot([5*r["step"] for r in rows], [r["delta_w_rmse"] for r in rows], "o-", label=method)
        ax.set(title=branch, xlabel="Prediction horizon [ms]", ylabel="Wheel response vs hold RMSE [rad/s]")
    axes[0].legend(fontsize=6)
    fig.savefig(folder/"actions.png", dpi=150)
    plt.close(fig)


def gate(d, curve, actions, timings, method):
    label = "independent_burckhardt"
    find = lambda name: next(r for r in curve if r["suite"] == label and r["method"] == name and r["group"] == "all")
    a, b = find(method), find("e4_a")
    criteria = d["gate"]
    branches = [r for r in actions if r["suite"] == label and r["group"] == "all" and r["step"] == max(d["action_steps"])]
    lookup = {(r["method"], r["branch"]): r for r in branches}
    action_regression = {branch: lookup[method, branch]["delta_w_rmse"]/lookup["e4_a", branch]["delta_w_rmse"]-1
                         for branch in ("increase", "pulse")}
    daily = max(lookup[method, branch][metric]/lookup["e4_a", branch][metric]-1
                for branch in ACTIONS for metric in ("v_rmse", "w_rmse", "slip_rmse", "fx_rmse"))
    runtime = float(np.mean([r["refine_p95_ms"] for r in timings if r["method"] == method]))
    observed = {"curve_relative_regression": a["curve_rmse"]/b["curve_rmse"]-1,
                "force_relative_regression": a["point_force_rmse_n"]/b["point_force_rmse_n"]-1,
                "regret_relative_gain": 1-a["regret"]/b["regret"], "action_relative_regression": action_regression,
                "state_relative_regression": daily, "violation_fraction": a["physical_violation"], "refine_p95_ms": runtime}
    passes = (observed["curve_relative_regression"] <= criteria["max_curve_relative_regression"] and
              observed["force_relative_regression"] <= criteria["max_current_force_relative_regression"] and
              observed["regret_relative_gain"] >= criteria["min_regret_relative_gain"] and
              max(action_regression.values()) <= criteria["max_action_relative_regression"] and
              daily <= criteria["max_state_relative_regression"] and
              observed["violation_fraction"] <= criteria["max_violation_fraction"] and runtime <= criteria["max_refine_p95_ms"])
    return {"method": method, "passes": passes, "thresholds": criteria, "measurements": observed}


def evaluate(c, d, base, reference, recurrent, output, models, fitted, prior, val, training, original, partitions):
    start = perf_counter()
    seed_everything(c["seeds"][0], c)
    grid = np.linspace(0, 1, d["eval_grid"])
    policy = score_selection(c, d, val, models, fitted, training, prior, grid, output)
    best, size, count = (policy[k] for k in ("selected_structured", "refinement_history", "refinement_steps"))
    conditional_prior(output, prior, c)
    methods = baselines(c, base, reference, recurrent, models)
    recurrent_options = read_json(recurrent/"experiment.json")["options"]
    for (name, seed), model in fitted.items():
        observer = None
        if name == "structured_recurrent":
            observer = RecurrentTireModel(model, recurrent_options).to(c["device"]).eval().requires_grad_(False)
            model = observer.tire
        methods[name, seed] = (model, observer)
    suites = {"legacy": existing_suite(c, d, base, "test", original["test"])}
    for name in ("independent_burckhardt", "independent_magic", "withheld_rational", "decoupled", "ab"):
        suites[name] = stress_suite(c, d, partitions[name], ab=name == "ab")
    suites["memory"] = stress_suite(c, d, partitions["independent_burckhardt"][:36], memory=True)
    latent_probe(c, best, fitted, models, val, suites["independent_burckhardt"], grid, output)
    all_curve, all_bias, all_action, ranking, refined_rows, timing, feedback_rows, ambiguity_rows = [], [], [], [], [], [], [], []
    plot_cases = {k: (suites[k], {}) for k in ("ab", "decoupled", "withheld_rational")}
    paired_rows = []
    for suite_name, suite in suites.items():
        print(f"evaluate {suite_name} windows={len(suite['history'])}", flush=True)
        # A deterministic same-window action subset keeps the full ablation matrix practical.
        action_indices = np.random.default_rng(d["seed"]).choice(len(suite["history"]), min(160, len(suite["history"])),
                                                                replace=False)
        act = subset(suite, action_indices)
        actions, truth, true_mu = action_suite(c, d, act)
        distributions = {}

        def record(name, seed, learned, observer, z, prediction=None, step=0):
            if prediction is None:
                prediction, exact, point = curves(c, learned, z, suite["history"], grid)
            else:
                exact, point = None, None
            info = curve_data(c, d, suite, grid, prediction, exact, clipped=name == "e4_clipped", current=point)
            keys = {"suite": suite_name, "method": name, "seed": seed}
            curves_out, bias = curve_rows(suite, info, keys)
            all_curve.extend(curves_out)
            all_bias.extend(bias)
            if suite_name in plot_cases and seed in (c["seeds"][0], -1):
                plot_cases[suite_name][1][name] = prediction
                plot_cases[suite_name][1]["true"] = info["true"]
            if learned is None:
                forecast_values, mu = forecast(c, None, None, act["history"], actions, discrete=prediction[action_indices])
            else:
                forecast_values, mu = forecast(c, learned, z[action_indices], act["history"], actions, observer)
            rows, aggregate, contrasts = action_rows(c, d, act, forecast_values, mu, truth, true_mu, keys)
            all_action.extend(rows)
            ranking.append(aggregate)
            for metric, value in (("curve_rmse", np.sqrt(np.mean(info["errors"][action_indices]**2, axis=1))),
                                  ("regret", info["regret"][action_indices]), ("action_delta_w_mae", contrasts)):
                distributions.setdefault((name, metric), []).append(value)
            if suite_name == "legacy" and learned is not None:
                timing.append(efficiency(c, d, name, seed, learned, observer, z, suite, size, step))
                if name in ("e4_a", "recurrent_transition", best, "structured_refine", "structured_recurrent",
                            "structured_recurrent_refine"):
                    feedback_rows.extend(feedback_test(c, d, learned, observer, z, suite, size, step, name, seed))
            return prediction

        for name in ("nls", "pchip"):
            record(name, -1, None, None, None, traditional(c, suite, grid, name))
        for (name, seed), (model, observer) in methods.items():
            z = context(c, model, suite, observer)
            prediction = record(name, seed, model, observer, z)
            if name not in (best, "e4_a", "structured_recurrent"):
                continue
            raw_history = histories(suite, size, c)
            result, _ = refine_batch(c, d, model, z, raw_history, [count])
            refined_name = {best: "structured_refine", "e4_a": "e4_refine",
                            "structured_recurrent": "structured_recurrent_refine"}[name]
            record(refined_name, seed, model, observer, result[count][0], step=count)
            if name == best:
                with_prior, _ = refine_batch(c, d, model, z, raw_history, [count], prior=prior)
                record("structured_prior_refine", seed, model, observer, with_prior[count][0], step=count)
                if suite_name == "memory":
                    warm = warm_refinement(c, d, model, z, suite, count, size)
                    record("structured_warm_refine", seed, model, observer, warm, step=count)
                    if (suite["caps"] == 2.25).any():
                        span = raw_history[:, :, 2].max(axis=1)-raw_history[:, :, 2].min(axis=1)
                        write_csv(output/"memory_excitation.csv", [
                            {"time_s": t, "n": int((suite["caps"] == t).sum()),
                             "median_recent_100_slip_span": float(np.median(span[suite["caps"] == t]))}
                            for t in np.unique(suite["caps"])])
                if suite_name in ("legacy", "independent_burckhardt", "independent_magic", "withheld_rational", "ab"):
                    for length in d["refinement_histories"]:
                        refined_rows.extend(refinement_rows(c, d, suite, suite_name, seed, model, z, grid, prediction,
                                                           d["refinement_steps"], length))
                if suite_name == "ab":
                    ambiguity_rows.extend(ambiguity(c, d, model, z, suite, count, seed, grid))
        pairs = paired(np.array([r["id"] for r in act["roads"]]), distributions, d["seed"], d["bootstrap_samples"])
        paired_rows.extend({"suite": suite_name, **row} for row in pairs)
    curves_summary = averages(all_curve, ("suite", "method", "group"))
    actions_summary = averages(all_action, ("suite", "method", "group", "branch", "step"))
    refine_summary = averages(refined_rows, ("suite", "history", "iterations", "group"))
    for filename, rows in (("comparison.csv", curves_summary), ("curve_by_seed.csv", all_curve), ("bias_by_seed.csv", all_bias),
                           ("action_comparison.csv", actions_summary), ("action_by_seed.csv", all_action),
                           ("action_ranking.csv", averages(ranking, ("suite", "method"))), ("paired.csv", paired_rows),
                           ("refinement_grid.csv", refine_summary), ("refinement_by_seed.csv", refined_rows),
                           ("timing.csv", timing), ("feedback.csv", feedback_rows), ("ambiguity.csv", ambiguity_rows)):
        write_csv(output/filename, rows)
    plots(output, d, grid, plot_cases, refine_summary, actions_summary, (best,))
    outcomes = [gate(d, curves_summary, actions_summary, timing, m)
                for m in (best, "structured_refine", "structured_recurrent_refine")]
    write_json(output/"acceptance.json", {"policy": policy, "outcomes": outcomes, "evaluation_seconds": perf_counter()-start,
                                         "suite_sizes": {k: len(v["history"]) for k, v in suites.items()}})
    print(f"Physical prior experiment: {output}", flush=True)
