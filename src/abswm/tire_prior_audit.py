"""Factual forecast continuity, paired stress examples and online iteration timing."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .decoder_study import read_json
from .controllers import SlipPI
from .diagnostics import rmse, tensor
from .evaluation import write_csv
from .models import seed_everything
from .physics import slip
from .recurrent_model import RecurrentTireModel
from .tire_prior import history_evidence, refine
from .tire_prior_data import advance, pack, stress_suite, tire_function, trajectory
from .tire_prior_evaluation import baselines, context, curves, forecast, refine_batch, subset, traditional, warm_refinement
from .tire_prior_metrics import averages, curve_data
from .tire_prior_training import existing_suite, histories


def tracking_memory(c, d, roads, size):
    """Excite the tire, then track a moderate constant slip to study retained information."""
    active = roads[:36]
    speeds = np.full(len(active), 30.)
    x, u = trajectory(c, active, speeds, np.full(len(active), 2400.), seconds=.65)
    states, actions = list(x.transpose(1, 0, 2)), list(u.T)
    controllers = [SlipPI(c, 6000., 20000.) for _ in active]
    for controller, torque in zip(controllers, actions[-1]):
        controller.torque = float(torque)
    coefficient = tire_function(active)
    dt = c["control_dt"]
    for _ in range(round((2.5-.65)/dt)):
        current = states[-1]
        request = np.array([ctrl.step(float(lam), .12)
                            for ctrl, lam in zip(controllers, slip(current, c["radius"]))])
        states.append(advance(current.astype(float), request, coefficient, c))
        actions.append(request)
    states, actions = np.stack(states, axis=1).astype(np.float32), np.stack(actions, axis=1).astype(np.float32)
    anchors, times = [], []
    for i in range(len(active)):
        for time in (.65, 1.20, 1.70, 2.25):
            t = round(time/dt)
            if states[i, t, 0] > c["stop_speed"]+.05:
                anchors.append((i, t))
                times.append(time)
    suite = pack(c, states, actions, active, anchors, times)
    measured = histories(suite, size, c)[..., 2]
    median = []
    for time in np.unique(times):
        mask = suite["caps"] == time
        median.append({"time_s": time, "windows": int(mask.sum()), "median_current_slip": float(np.median(measured[mask, -1])),
                       "median_recent_slip_span": float(np.median(np.ptp(measured[mask], axis=1))),
                       "locked_fraction": float((measured[mask, -1] >= .995).mean())})
    return suite, median


def memory_audit(c, d, output, refs, fitted, best, size, iterations, roads, grid):
    suite, measured = tracking_memory(c, d, roads, size)
    write_csv(output/"tracking_memory_excitation.csv", measured)
    observed = histories(suite, size, c)
    rows = []
    for seed in c["seeds"]:
        model = fitted[best, seed]
        z = context(c, model, suite)
        cold, _ = refine_batch(c, d, model, z, observed, [iterations])
        warm = warm_refinement(c, d, model, z, suite, iterations, size)
        for name, tire, latent in (("e4_a", refs[seed], context(c, refs[seed], suite)), (best, model, z),
                                   ("structured_refine", model, cold[iterations][0]),
                                   ("structured_warm_refine", model, warm)):
            values, exact, point = curves(c, tire, latent, suite["history"], grid)
            info = curve_data(c, d, suite, grid, values, exact, current=point)
            for phase in np.unique(suite["caps"]):
                mask = suite["caps"] == phase
                rows.append({"method": name, "seed": seed, "time_s": phase, "n": int(mask.sum()),
                             "curve_rmse": rmse(info["errors"][mask]), "regret": float(info["regret"][mask].mean()),
                             "force_rmse_n": rmse(info["force_error"][mask])})
    write_csv(output/"tracking_memory.csv", averages(rows, ("method", "time_s")))


def decoupling_plot(output, grid, matrices, roads, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caps = matrices["decoupled_caps"]
    identifiers = matrices["decoupled_roads"]
    fig, axes = plt.subplots(len(roads)//2, 2, figsize=(10, 12), sharex=True, layout="constrained")
    for i, paired_roads in enumerate(zip(roads[::2], roads[1::2])):
        for j, cap in enumerate((.02, .2)):
            ax = axes[i, j]
            for suffix, road, color in zip("AB", paired_roads, ("#21618c", "#b7472a")):
                found = np.flatnonzero((identifiers == road["id"]) & (caps == cap))
                if len(found):
                    k = int(found[0])
                    for name, style, label in (("truth", "-", "true"), (selected, "--", "encoder"),
                                               ("structured_refine", ":", "refined")):
                        ax.plot(grid, matrices[f"decoupled_{name}"][k], style, color=color, label=f"{suffix} {label}")
            ax.set(title=f"{paired_roads[0]['subfamily'][:-2].replace('_', ' ').capitalize()} · {cap:.0%}",
                   ylabel="Friction" if j == 0 else "", xlabel="Slip" if i == len(axes)-1 else "")
    axes[0, 0].legend(ncol=2, fontsize=7)
    fig.savefig(output/"figures"/"decoupled_pairs.png", dpi=145)
    plt.close(fig)


def audit(c, d, base, reference, recurrent, output, references, fitted, prior, original, partitions):
    seed_everything(c["seeds"][0], c)
    policy = read_json(output/"validation_policy.json")
    best, length, iterations = (policy[k] for k in ("selected_structured", "refinement_history", "refinement_steps"))
    grid = np.linspace(0, 1, d["eval_grid"])
    methods = baselines(c, base, reference, recurrent, references)
    opts = read_json(recurrent/"experiment.json")["options"]
    for (name, seed), model in fitted.items():
        observer = None
        if name == "structured_recurrent":
            observer = RecurrentTireModel(model, opts).to(c["device"]).eval().requires_grad_(False)
            model = observer.tire
        methods[name, seed] = model, observer
    suite = existing_suite(c, d, base, "test", original["test"])
    subset_idx = np.random.default_rng(d["seed"]).choice(len(suite["history"]), 160, replace=False)
    suite = subset(suite, subset_idx)
    raw = suite["raw"]
    h = c["horizon_steps"]
    u = np.stack([raw["actions"][i, t:t+h] for i, t in suite["locators"]])[:, None]
    x = np.stack([raw["states"][i, t+1:t+h+1] for i, t in suite["locators"]])
    true_slip = slip(x, c["radius"])
    mu_true = tire_function(suite["roads"])(true_slip)
    observed = histories(suite, length, c)
    factual = []
    load = c["mass"]*c["gravity"]

    def factual_rows(name, seed, model, observer, z, discrete=None):
        prediction, mu = forecast(c, model, z, suite["history"], u, observer, discrete)
        error = prediction[:, 0]-x
        slip_error = slip(prediction[:, 0], c["radius"])-true_slip
        force_error = (mu[:, 0]-mu_true)*load
        surface = np.array([r["surface"] for r in suite["roads"]])
        for group, mask in {"all": np.ones(len(x), dtype=bool), "hard": surface == "hard", "low": surface == "low"}.items():
            for k in d["action_steps"]:
                factual.append({"method": name, "seed": seed, "group": group, "n": int(mask.sum()), "step": k,
                                "v_rmse": rmse(error[mask, k-1, 0]), "w_rmse": rmse(error[mask, k-1, 1]),
                                "slip_rmse": rmse(slip_error[mask, k-1]), "fx_rmse": rmse(force_error[mask, k-1])})

    for name in ("nls", "pchip"):
        factual_rows(name, -1, None, None, None, traditional(c, suite, grid, name))
    for (name, seed), (model, observer) in methods.items():
        z = context(c, model, suite, observer)
        factual_rows(name, seed, model, observer, z)
        if name in (best, "e4_a", "structured_recurrent"):
            refined = refine_batch(c, d, model, z, observed, [iterations])[0][iterations][0]
            label = {best: "structured_refine", "e4_a": "e4_refine",
                     "structured_recurrent": "structured_recurrent_refine"}[name]
            factual_rows(label, seed, model, observer, refined)
            if name == best:
                with_prior = refine_batch(c, d, model, z, observed, [iterations], prior=prior)[0][iterations][0]
                factual_rows("structured_prior_refine", seed, model, observer, with_prior)
    write_csv(output/"factual_by_seed.csv", factual)
    write_csv(output/"factual_comparison.csv", averages(factual, ("method", "group", "step")))

    timers = []
    for seed in c["seeds"]:
        candidate = fitted[best, seed]
        small = subset(suite, np.arange(14))
        z = context(c, candidate, small)
        for size in d["refinement_histories"]:
            evidence = tensor(histories(small, size, c), c)
            for count in d["refinement_steps"]:
                values = []
                for i in range(len(z)):
                    start = perf_counter()
                    refine(candidate, z[i:i+1], history_evidence(evidence[i:i+1], c), d, [count])
                    values.append((perf_counter()-start)*1000)
                timers.append({"seed": seed, "method": "structured_refine", "history": size, "iterations": count,
                               "p50_ms": float(np.percentile(values[3:], 50)), "p95_ms": float(np.percentile(values[3:], 95))})
    for name in ("nls", "pchip"):
        values = []
        for i in range(24):
            start = perf_counter()
            traditional(c, subset(suite, np.array([i])), grid, name)
            values.append((perf_counter()-start)*1000)
        timers.append({"seed": -1, "method": name, "history": c["history_steps"], "iterations": None,
                       "p50_ms": float(np.percentile(values[3:], 50)), "p95_ms": float(np.percentile(values[3:], 95))})
    write_csv(output/"iteration_timing.csv", timers)

    cases, matrices = [], {}
    for name in ("ab", "decoupled"):
        current = stress_suite(c, d, partitions[name], ab=name == "ab")
        def save(method, seed, values, exact=None):
            info = curve_data(c, d, current, grid, values, exact)
            if seed in (c["seeds"][0], -1):
                matrices[f"{name}_{method}"] = values
                matrices[f"{name}_truth"] = info["true"]
                matrices[f"{name}_caps"] = current["caps"]
                matrices[f"{name}_roads"] = np.asarray([r["id"] for r in current["roads"]])
            for i, road in enumerate(current["roads"]):
                row = {"suite": name, "method": method, "seed": seed, "road_id": road["id"], "pair_member": road["subfamily"],
                       "slip_cap": current["caps"][i], "observed_max": info["max_slip"][i],
                       "curve_rmse": rmse(info["errors"][i]), "regret": info["regret"][i]}
                for j, metric in enumerate(("height", "peak", "stiffness", "tail", "width95")):
                    row["true_"+metric] = info["truth"][i, j]
                    row["pred_"+metric] = info["physical"][i, j]
                cases.append(row)
        save("nls", -1, traditional(c, current, grid, "nls"))
        for seed in c["seeds"]:
            for method, model in (("e4_a", references[seed]), (best, fitted[best, seed]),
                                  ("prior_tail", fitted["prior_tail", seed]), ("structured_prior", fitted["structured_prior", seed])):
                z = context(c, model, current)
                y, params, _ = curves(c, model, z, current["history"], grid)
                save(method, seed, y, params)
                if method == best:
                    z = refine_batch(c, d, model, z, histories(current, length, c), [iterations])[0][iterations][0]
                    y, params, _ = curves(c, model, z, current["history"], grid)
                    save("structured_refine", seed, y, params)
    write_csv(output/"stress_by_road.csv", cases)
    np.savez_compressed(output/"stress_curves.npz", grid=grid, **matrices)
    decoupling_plot(output, grid, matrices, partitions["decoupled"], best)
    memory_audit(c, d, output, references, fitted, best, length, iterations, partitions["independent_burckhardt"], grid)
    print("Factual, iteration timing and paired stress audits written", flush=True)
