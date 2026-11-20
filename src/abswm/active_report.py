"""Postprocess traces and generate research figures and machine-readable decisions."""
from __future__ import annotations

import csv
from collections import deque

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .active_belief import make_observer
from .active_evaluation import options_for
from .active_experiment import accelerate
from .active_metrics import control_summary, grouped, mean_rows
from .active_information import information_values
from .active_rollout import GRID, OraclePredictor, candidate_actions, utilities
from .config import write_json
from .decoder_study import read_json
from .evaluation import write_csv


def acceptance(c, d, policy, output, rows, prediction, smoke):
    gate = policy["acceptance"]
    closed = {r["method"]: r for r in control_summary(rows)}
    exploratory, passive = closed["C13"], closed["C11"]
    p0 = closed["C6"]
    independent = {"burckhardt", "magic", "rational"}
    risk = [r for r in prediction["risk"] if r["suite"] in independent and r["method"] == "learned"
            and r["event"] == "high_slip" and r["fnr"] is not None]
    nominal = [r for r in prediction["probability"] if r["suite"] in independent and r["method"] == "learned"
               and r["nominal"] == .9 and r["step_count"] == policy["horizon"]]
    coverage = mean_rows(nominal, ("suite", "variable"), ("coverage",))
    coverage_error = max((abs(r["coverage"]-.9) for r in coverage), default=None)
    distance = (passive["distance_m"]-exploratory["distance_m"])/passive["distance_m"]
    fnr = max((r["fnr"] for r in risk), default=None)
    brier = max((r["brier"] for r in risk), default=None)
    values = {"active_relative_distance_gain": distance,
              "active_minus_passive_high_slip_s": exploratory["high_slip_s"]-passive["high_slip_s"],
              "active_minus_passive_lock_s": exploratory["lock_s"]-passive["lock_s"],
              "active_minus_passive_failure_rate": exploratory["failure_rate"]-passive["failure_rate"],
              "candidate_minus_e4_mpc_distance_m": exploratory["distance_m"]-p0["distance_m"],
              "worst_family_high_slip_fnr": fnr, "worst_family_brier": brier, "worst_family_variable_90pct_error": coverage_error}
    checks = {"active_distance": distance >= gate["min_distance_gain_fraction"],
              "high_slip": values["active_minus_passive_high_slip_s"] <= gate["max_high_slip_regression_s"],
              "lock": values["active_minus_passive_lock_s"] <= gate["max_lock_regression_s"],
              "failure": values["active_minus_passive_failure_rate"] <= gate["max_failure_regression"],
              "fnr": fnr is not None and fnr <= gate["max_high_slip_fnr"],
              "brier": brier is not None and brier <= gate["max_brier"],
              "coverage": coverage_error is not None and coverage_error <= gate["max_90pct_coverage_error"]}
    passed = all(checks.values()) and not smoke
    result = {"protocol": "smoke" if smoke else "frozen independent test", "passed": passed,
              "current_formal_model": "e4_a", "candidate": policy["variant"], "thresholds": gate,
              "checks": checks, "statistics": values}
    write_json(output/"acceptance.json", result)
    return result


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def trace_source(output):
    for path in sorted((output/"traces").glob("*.csv")):
        episode, seed, speed = path.stem.rsplit("_", maxsplit=2)
        road, method = episode.split("_", maxsplit=1)
        yield (road, method, int(seed), float(speed)), read_csv(path)


def exploration_reports(c, d, policy, output, rows, models, partitions, *, source=None, write=True):
    road_by_id = {road["id"]: road for name, roads in partitions.items() if name.startswith("test_") for road in roads}
    options = options_for(c, d, policy["horizon"], policy["observation_scale"])
    by_case = {(r["road_id"], r["method"], r["seed"], r["initial_speed"]): r for r in rows}
    events, trace_arrays = [], {}
    for (road_id, method, seed, speed), trace in trace_source(output) if source is None else source:
        if method not in {"C3", "C4", "C6", "C8", "C9", "C10", "C11", "C12", "C13"}:
            continue
        case = by_case[road_id, method, seed, speed]
        trace_arrays[road_id, method, seed, speed] = trace
        if method not in {"C8", "C9", "C10", "C13"}:
            continue
        road = road_by_id[road_id]
        oracle = accelerate(OraclePredictor(c, road), road)
        learned = models.get((policy["variant"], seed))
        belief = make_observer(c, learned, policy) if method in {"C10", "C13"} else None
        previous = case["initial_torque"]
        evaluation = None
        history = deque(maxlen=c["history_steps"])
        for row in trace:
            tick = int(row["step"])
            current = np.array([float(row["v"]), float(row["omega"])])
            h = [*current, float(row["slip"]), previous]
            if not history:
                history.extend([h]*c["history_steps"])
            else:
                history.append(h)
            if belief is not None:
                old_weights = belief.weights.copy()
                belief.observe(np.asarray(history), tick)
            if evaluation is not None:
                actions = candidate_actions(previous, policy["horizon"], c)[0]
                exact = oracle.predict(current, actions)
                forecast = belief.predict(current, actions)
                utility = utilities(forecast, actions, previous, options)
                true_utility = utilities(exact, actions, previous, options)[0]
                previous_choice, next_choice = np.argmax(old_weights@utility), np.argmax(belief.weights@utility)
                truth = exact.state[0]
                before = np.einsum("k,kath->ath", old_weights, forecast.state)
                after = np.einsum("k,kath->ath", belief.weights, forecast.state)
                force_before = old_weights@forecast.force.reshape(len(old_weights), -1)
                force_after = belief.weights@forecast.force.reshape(len(old_weights), -1)
                high = exact.high[0]
                reward_truth = exact.interval_mu[0].mean(axis=-1)
                reward_prediction = forecast.interval_mu.mean(axis=-1)
                i, j = np.triu_indices(len(reward_truth), 1)
                meaningful = np.abs(reward_truth[i]-reward_truth[j]) > d["tie_margin_mu"]

                def pair_accuracy(weights):
                    estimate = weights@reward_prediction
                    match = np.sign(estimate[i]-estimate[j]) == np.sign(reward_truth[i]-reward_truth[j])
                    return float(match[meaningful].mean()) if meaningful.any() else None

                evaluation.update(real_entropy_drop=evaluation["posterior_entropy"]-float(row["posterior_entropy"]),
                                  next_decision_gap=float(row["decision_gap"]),
                                  future_state_wheel_rmse_delta=float(np.sqrt(np.mean((after[..., 1]-truth[..., 1])**2))-
                                                                    np.sqrt(np.mean((before[..., 1]-truth[..., 1])**2))),
                                  future_force_rmse_delta=float(np.sqrt(np.mean((force_after-exact.force[0].ravel())**2))-
                                                               np.sqrt(np.mean((force_before-exact.force[0].ravel())**2))),
                                  future_risk_brier_delta=float(np.mean((belief.weights@forecast.high-high)**2)-
                                                                np.mean((old_weights@forecast.high-high)**2)),
                                  future_ranking_before=pair_accuracy(old_weights), future_ranking_after=pair_accuracy(belief.weights),
                                  future_action_changed=bool(previous_choice != next_choice),
                                  future_utility_gain=float(true_utility[next_choice]-true_utility[previous_choice]))
                evaluation = None
            chosen = int(row["candidate"])
            modified = bool(row["modification"])
            active = row["executed_exploration"] == "True"
            if chosen >= 0 and (modified or active):
                proposal = float(row["proposal"]) if row["proposal"] else None
                actions = candidate_actions(previous, policy["horizon"], c, proposal=proposal)[0]
                exact = oracle.predict(current, actions)
                values = utilities(exact, actions, previous, options)[0]
                baseline = 0 if modified else int(row["passive"])
                passive = int(row["passive"])
                event = {"road_id": road_id, "method": method, "seed": seed, "initial_speed": speed, "tick": tick,
                         "time_s": float(row["time_s"]), "modified": modified, "executed_exploration": active,
                         "reason": row["modification"], "supervisor_reason": row["safety"],
                         "proposal_nm": float(row["proposal"]) if row["proposal"] else None,
                         "requested_nm": float(row["requested"]) if row["requested"] else None,
                         "applied_nm": float(row["applied"]), "selected_action": chosen, "passive_action": passive,
                         "estimated_gain": float(row["gain"]), "estimated_decision_value": float(row["value"]),
                         "gamma": float(row["gamma"]),
                         "posterior_entropy": float(row["posterior_entropy"]) if row["posterior_entropy"] else None,
                         "short_term_utility_delta_vs_proposal": float(values[chosen]-values[baseline]),
                         "short_term_impulse_delta_vs_passive_ns": float(exact.impulse[0, chosen]-exact.impulse[0, passive]),
                         "risk_reduction_vs_proposal": bool(exact.high[0, baseline] and not exact.high[0, chosen]),
                         "real_entropy_drop": None, "next_decision_gap": None, "future_state_wheel_rmse_delta": None,
                         "future_force_rmse_delta": None, "future_risk_brier_delta": None,
                         "future_ranking_before": None, "future_ranking_after": None,
                         "future_action_changed": None, "future_utility_gain": None}
                events.append(event)
                if active:
                    evaluation = event
            previous = float(row["applied"])
    if events and write:
        write_csv(output/"exploration_events.csv", events)
    return events, trace_arrays


def save_figures(output, c, policy, rows, prediction, traces):
    root = output/"figures"
    root.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 145, "axes.grid": True, "grid.alpha": .2, "font.size": 9})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(root/f"{name}.png", dpi=145, bbox_inches="tight")
        plt.close(fig)

    cases = prediction["figures"]
    choose = [v for (name, tag, excitation), v in cases.items()
              if name in {"burckhardt", "historical_ab", "historical_decoupled"} and
              (name != "historical_decoupled" or tag.startswith("same_rise"))]
    if choose:
        chosen = choose[:3]+choose[-3:]
        fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.9), sharex=True, sharey=True)
        for ax, item in zip(axs.flat, chosen):
            weights, curves = item["weights"], item["curves"]
            mean = weights@curves
            sd = np.sqrt(weights@((curves-mean)**2))
            ax.plot(GRID, item["actual_curve"], c="black", label="Plant")
            for j, (w, curve) in enumerate(zip(weights, curves)):
                ax.plot(GRID, curve, lw=.8, alpha=.3+.6*w, label=f"Particle {j+1}" if ax is axs.flat[0] else None)
            ax.plot(GRID, mean, c="#065788", lw=1.3)
            ax.fill_between(GRID, (mean-1.645*sd).clip(0), mean+1.645*sd, color="#1d89a6", alpha=.19)
            info = item["info"]
            ax.axvspan(0, info["coverage"], color="#aec6dc", alpha=.3)
            ax.set_title(f"{info['tag']} · slip seen {info['coverage']:.1%}")
        fig.supxlabel("Slip ratio")
        fig.supylabel("Friction coefficient")
        save(fig, "posterior_curves")
        item = choose[0]
        fig, axs = plt.subplots(2, 2, figsize=(9, 6))
        time = np.arange(1, item["forecast"].shape[-2]+1)*c["control_dt"]*1000
        for idx, (ax, name) in enumerate(zip(axs.flat, ("Velocity (m/s)", "Wheel speed (rad/s)", "Slip", "Force (N)"))):
            for a, color in zip((0, 1, 4, 7), ("#216eaa", "#188463", "#bd4c39", "#9e6bac")):
                prediction_k = item["forecast"][:, a, :, idx]
                ax.plot(time, item["truth"][a, :, idx], c=color, lw=1.8)
                ax.plot(time, prediction_k.T, c=color, alpha=.4, lw=.7)
            ax.set_ylabel(name)
        fig.supxlabel("Prediction time (ms)")
        save(fig, "counterfactual_rollouts")
    bins = prediction["bins"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
    for ax, family in zip(axs, ("burckhardt", "magic", "rational")):
        ax.plot([0, 1], [0, 1], c="0.6", lw=.9)
        for method in ("learned", "library", "ensemble"):
            points = sorted([r for r in bins if (r["suite"], r["method"], r["event"]) == (family, method, "high_slip")],
                            key=lambda r: r["p_mean"])
            if points:
                ax.plot([r["p_mean"] for r in points], [r["observed_rate"] for r in points], ".-", label=method)
        ax.set_title(family)
    axs[0].legend()
    fig.supxlabel("Mean predicted high-slip probability")
    fig.supylabel("Observed event frequency")
    save(fig, "risk_calibration")
    summary = {r["method"]: r for r in control_summary(rows)}
    official = [f"C{i}" for i in range(14) if f"C{i}" in summary]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
    axs[0].bar(official, [summary[n]["distance_m"] for n in official], color="#3979a8")
    axs[0].set_ylabel("Mean distance (m)")
    axs[1].bar(official, [summary[n]["lock_s"] for n in official], color="#ad664d")
    axs[1].set_ylabel("Mean lock duration (s)")
    for ax in axs:
        ax.tick_params(axis="x", rotation=45)
    save(fig, "closed_loop_distance_lock")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.7))
    for method in official:
        r = summary[method]
        left.scatter(r["high_slip_s"], r["distance_m"], s=21)
        left.annotate(method, (r["high_slip_s"], r["distance_m"]), xytext=(3, 2), textcoords="offset points")
    left.set(xlabel="Mean high-slip duration (s)", ylabel="Mean distance (m)")
    by_family = control_summary(rows, ("method", "family"))
    for family in ("burckhardt", "magic", "rational"):
        points = {r["method"]: r["distance_m"] for r in by_family if r["family"] == family}
        if points:
            right.plot(official, [points.get(n, np.nan) for n in official], ".-", label=family)
    right.legend()
    right.tick_params(axis="x", rotation=45)
    right.set_ylabel("Mean distance by family (m)")
    save(fig, "risk_tradeoff_and_generalization")
    rank = prediction["scores"]["action_regret"]
    diag = mean_rows([r for r in rank if not r["suite"].startswith("historical")], ("method",), ("pure_regret_ns", "violation"))
    diag = sorted(diag, key=lambda r: r["pure_regret_ns"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([r["method"] for r in diag], [r["pure_regret_ns"] for r in diag], color="#3979a8")
    ax.tick_params(axis="x", rotation=70)
    ax.set_ylabel("Average braking impulse regret (N·s)")
    save(fig, "decision_regret")
    rr = [r for r in prediction["scores"]["refinement"] if r["before"] is not None]
    if rr:
        fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, 3.6))
        for key, members in grouped(rr, ("method",)).items():
            left.scatter([r["before"] for r in members], [r["after"] for r in members], s=7, alpha=.4, label=key[0])
            right.scatter([r["max_eigenvalue"] for r in members], [r["movement"] for r in members], s=7, alpha=.4)
        maximum = max(r["before"] for r in rr)
        left.plot([0, maximum], [0, maximum], color="0.4")
        left.set(xlabel="History RMSE before", ylabel="History RMSE after")
        right.set(xlabel="Largest standardized information eigenvalue", ylabel="Latent displacement")
        left.legend(fontsize=6)
        save(fig, "latent_refinement")
    for road_id, method, seed, speed in traces:
        if method != "C13" or (road_id, "C11", seed, speed) not in traces:
            continue
        fig, axs = plt.subplots(3, 2, figsize=(9.8, 7.2), sharex=True)
        for name, color in (("C11", "#346f9c"), ("C13", "#b95741")):
            trace = traces[road_id, name, seed, speed]
            time = np.array([float(r["time_s"]) for r in trace])
            for ax, key, label in zip(axs.flat, ("v", "slip", "force_n", "applied", "posterior_entropy", "gamma"),
                                      ("Vehicle (m/s)", "Slip", "Force (N)", "Applied torque (Nm)", "Entropy (nats)", "Exploration weight")):
                y = np.array([float(r[key]) if r[key] else np.nan for r in trace])
                ax.plot(time, y, color=color, lw=.9, label=name)
                if name == "C13":
                    idx = np.array([r["executed_exploration"] == "True" for r in trace])
                    safety = np.array([bool(r["safety"]) for r in trace])
                    ax.scatter(time[idx], y[idx], color="#b95741", s=6)
                    ax.scatter(time[safety], y[safety], color="black", s=3)
                ax.set_ylabel(label)
        axs[0, 0].legend()
        fig.supxlabel("Time (s)")
        save(fig, f"paired_trace_{road_id}_{speed:g}")
    independent = [r for r in prediction["scores"]["information"] if not r["suite"].startswith("historical")]
    if independent:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for (steps,), group in grouped(independent, ("observation_steps",)).items():
            ax.scatter([r["gain_nats"] for r in group], [r["decision_value"] for r in group], s=5, alpha=.2, label=f"{steps*5} ms")
        ax.set(xlabel="Expected information gain (nats)", ylabel="Expected continuation value (normalized utility)")
        ax.legend()
        save(fig, "information_value")


def deliver_results(c, d, policy, args, rows, prediction, models=None, partitions=None):
    output = args.output
    result = acceptance(c, d, policy, output, rows, prediction, args.smoke)
    events, traces = exploration_reports(c, d, policy, output, rows, models, partitions,
                                          write=not (output/"decision_audit.json").exists())
    save_figures(output, c, policy, rows, prediction, traces)
    supplemental_figures(c, d, policy, output, models)
    paired_action_scores(c, d, policy, output)
    offline_time = prediction["scores"]["offline_timing"]
    timings = []
    for (method, component), group in grouped(offline_time, ("method", "component")).items():
        values = np.array([row["ms"] for row in group])
        timings.append({"method": method, "component": component, "phase": "open_loop", "groups": len(values),
                        "p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                        "max_ms": float(values.max())})
    for (method, component), group in grouped(read_csv(output/"closed_loop_timing.csv"), ("method", "component")).items():
        timings.append({"method": method, "component": component, "phase": "closed_loop_episode_percentiles",
                        "groups": len(group), "p50_ms": float(np.median([float(r["p50_ms"]) for r in group])),
                        "p95_ms": float(np.median([float(r["p95_ms"]) for r in group])),
                        "max_ms": max(float(r["max_ms"]) for r in group)})
    write_csv(output/"timing.csv", timings)
    print(f"Acceptance: {result['checks']}", flush=True)


def supplemental_figures(c, d, policy, output, models):
    """Render diagnostics from completed logs without repeating a simulation."""
    root = output/"figures"
    root.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(root/f"{name}.png", dpi=145, bbox_inches="tight")
        plt.close(fig)

    paired = [r for r in read_csv(output/"paired_statistics.csv") if r["metric"] == "distance_m"
              and r["method"] in {"C11", "C12", "C13", "no_memory", "information_refinement", "library_active"}]
    if paired:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        labels = [f"{r['method']} − {r['baseline']}" for r in paired]
        values = np.array([float(r["mean_delta"]) for r in paired])
        lo, hi = (np.array([float(r[k]) for r in paired]) for k in ("lo95", "hi95"))
        ax.errorbar(values, labels, xerr=np.maximum(0, [values-lo, hi-values]), fmt="o", c="#286d99", capsize=3)
        ax.axvline(0, c="0.5", lw=.8)
        ax.set_xlabel("Paired braking-distance difference (m), 95% road-cluster interval")
        save(fig, "control_ablation")

    path = output/"belief_update.csv"
    if path.exists():
        by_window = grouped([r for r in read_csv(path) if r["method"] == "learned" and r["suite"] == "burckhardt"
                             and r["seed"] == str(c["seeds"][0])], ("road_id", "window", "excitation"))
        weak = [key for key, members in by_window.items() if key[2] == "weak" and len(members) >= 8]
        strong = [key for key, members in by_window.items() if key[2] == "strong" and len(members) >= 8]
        pairs = [(a, b) for a in weak for b in strong if a[0] == b[0]]
        if pairs:
            fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.5))
            for key, color in zip(pairs[0], ("#387da9", "#b4553d")):
                group = by_window[key]
                t = np.array([int(r["tick"]) for r in group])
                x = (t-t[0]+1)*c["control_dt"]*1000
                a.plot(x, [float(r["entropy_after"]) for r in group], ".-", c=color, label=key[-1])
                b.plot(x, [float(r["effective_size"]) for r in group], ".-", c=color)
            a.set_ylabel("Posterior entropy (nats)")
            b.set_ylabel("Effective hypothesis count")
            a.legend()
            fig.supxlabel("Time after first recorded innovation (ms)")
            save(fig, "belief_real_updates")

    source = dict(trace_source(output))
    paired_failures = [(key, (key[0], "C8", c["seeds"][0], key[3]))
                       for key, trace in source.items()
                       if key[1] == "C3" and float(trace[-1]["time_s"]) > c["max_time"]-1
                       and (key[0], "C8", c["seeds"][0], key[3]) in source]
    if paired_failures:
        fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.5))
        for key, color in zip(paired_failures[0], ("#aa5740", "#347d9e")):
            trace = source[key]
            x = np.array([float(r["time_s"]) for r in trace])
            mask = x <= .22
            left.plot(x[mask]*1000, np.array([float(r["applied"]) for r in trace])[mask], label=key[1], c=color)
            right.plot(x, [float(r["v"]) for r in trace], c=color)
        left.set(xlabel="Time (ms)", ylabel="Applied torque (Nm)")
        right.set(xlabel="Time (s)", ylabel="Vehicle speed (m/s)")
        left.legend()
        save(fig, "peak_seeking_failure")

    examples = [(key, trace) for key, trace in source.items() if key[1] == "C13"
                and any(r["executed_exploration"] == "True" for r in trace)]
    if examples:
        (road_id, method, seed, speed), trace = examples[0]
        match = next(r for r in read_csv(output/"closed_loop_by_road.csv") if
                     (r["road_id"], r["method"], int(r["seed"]), float(r["initial_speed"])) == (road_id, method, seed, speed))
        previous = float(match["initial_torque"])
        history = deque(maxlen=c["history_steps"])
        observer = make_observer(c, models[policy["variant"], seed], policy)
        for row in trace:
            x = np.array([float(row["v"]), float(row["omega"])])
            sample = [*x, float(row["slip"]), previous]
            history.extend([sample]*c["history_steps"] if not history else [sample])
            observer.observe(np.asarray(history), int(row["step"]))
            if row["executed_exploration"] == "True":
                break
            previous = float(row["applied"])
        actions = candidate_actions(previous, policy["horizon"], c)[0]
        forecast = observer.predict(x, actions)
        options = options_for(c, d, policy["horizon"], policy["observation_scale"])
        scores = utilities(forecast, actions, previous, options)
        gain, evoi = information_values(forecast.interval_mu[..., 0], observer.weights, scores, observer.scale)
        fig, axes = plt.subplots(2, 2, figsize=(10, 5.9), sharex=True)
        values = (observer.weights@forecast.impulse, observer.weights@forecast.high, gain, evoi)
        labels = ("Expected impulse (N·s)", "High-slip probability", "Information gain (nats)", "Continuation value")
        colors = ["#aaaaaa"]*len(actions)
        colors[int(row["passive"])] = "#3979a8"
        colors[int(row["candidate"])] = "#c06145"
        names = ("Hold", "Release\nfast", "Release\nslow", "Raise\nslow", "Raise\nfast",
                 "Raise\nhold", "Raise\nthen cut", "Pulse", "Multi-\nstage")
        for ax, data, label in zip(axes.flat, values, labels):
            ax.bar(names, data, color=colors)
            ax.set_ylabel(label)
            ax.tick_params(axis="x", labelsize=8)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color="#3979a8", label="Passive"), Patch(color="#c06145", label="Active")],
                   ncols=2, loc="upper center")
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(root/"exploration_candidates.png", dpi=145, bbox_inches="tight")
        plt.close(fig)


def paired_action_scores(c, d, policy, output):
    """Bootstrap open-loop deltas by road, holding the seed, history and actions fixed."""
    from collections import defaultdict

    keys = ("suite", "road_id", "window", "seed")
    contrasts = (("learned", "e4"), ("full", "rollout"), ("action", "rollout"),
                 ("ranking", "rollout"), ("risk", "rollout"), ("learned", "no_update"),
                 ("bayes_refinement", "learned"), ("full", "single"), ("learned", "full"))
    tables = [
        ("action_prediction.csv", {"response_wheel_rad_s": lambda r: float(r["response_wheel_rad_s"]),
                                   "force_n": lambda r: float(r["force_n"])}),
        ("action_regret.csv", {"pure_regret_ns": lambda r: float(r["pure_regret_ns"]),
                               "violation": lambda r: float(r["violation"] == "True"),
                               "top1": lambda r: float(r["top1"] == "True")}),
        ("action_ranking.csv", {"pair_accuracy": lambda r: int(r["correct_pairs"])/int(r["pairs"]) if int(r["pairs"]) else None}),
    ]
    rng = np.random.default_rng(d["partition_seed"]+770)
    output_rows = []
    for file, measures in tables:
        horizon = str(policy["horizon"])
        eligible = [r for r in read_csv(output/file) if not r["suite"].startswith("historical")
                    and r.get("step_count", horizon) == horizon]
        by_method = {method: {tuple(r[k] for k in keys): r for r in rows}
                     for (method,), rows in grouped(eligible, ("method",)).items()}
        for left, right in contrasts:
            if left not in by_method or right not in by_method:
                continue
            lhs, rhs = by_method[left], by_method[right]
            for metric, query in measures.items():
                deltas = defaultdict(list)
                for key in sorted(lhs.keys() & rhs.keys()):
                    x, y = query(lhs[key]), query(rhs[key])
                    if x is not None and y is not None:
                        deltas[key[1]].append(x-y)
                if not deltas:
                    continue
                values = np.array([np.mean(v) for v in deltas.values()])
                draws = rng.integers(len(values), size=(d["bootstrap_samples"], len(values)))
                lo, hi = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
                output_rows.append({"method": left, "baseline": right, "metric": metric, "roads": len(values),
                                    "matched_window_seed": sum(len(v) for v in deltas.values()),
                                    "mean_delta": np.mean(values), "median_delta": np.median(values),
                                    "sd": np.std(values, ddof=1) if len(values) > 1 else 0.,
                                    "min": np.min(values), "max": np.max(values), "lo95": lo, "hi95": hi})
    if output_rows:
        write_csv(output/"paired_action_statistics.csv", output_rows)


def historical_pair_figure(c, policy, output, models, original):
    """Use the same frozen observer on representative A/B and same-rise histories."""
    from pathlib import Path

    from .active_data import historical_stress
    from .active_evaluation import populate
    from .tire_prior_data import tire_function

    d = read_json(Path("configs/physics_prior.json"))
    suites = dict(historical_stress(c, original, d))
    ab, decoupled = suites["historical_ab"], suites["historical_decoupled"]
    seed = c["seeds"][0]
    model = models[policy["variant"], seed]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6), sharex=True, sharey=True)
    for i, identifier in enumerate(d["ab_ids"]):
        tests = [(ab, identifier, .02), (ab, identifier, .2), (decoupled, f"same_rise_{'ab'[i]}", .2)]
        for j, (suite, label, cap) in enumerate(tests):
            index = next(k for k, road in enumerate(suite["roads"])
                         if label in {road["id"], road["subfamily"]} and suite["caps"][k] == cap)
            road = suite["roads"][index]
            belief = populate(make_observer(c, model, policy), suite, index, c)
            ax = axes[i, j]
            ax.axvspan(0, suite["achieved"][index], color="#afc3d9", alpha=.32)
            ax.plot(GRID, tire_function([road])(GRID[None])[0], c="black", label="Plant")
            for k, (weight, curve) in enumerate(zip(belief.weights, belief.curves)):
                ax.plot(GRID, curve, c="#49749b", alpha=.15+.5*weight, lw=.8, label="Hypotheses" if k == 0 else None)
            ax.plot(GRID, belief.weights@belief.curves, c="#b65a42", label="Posterior mean")
            ax.set_title(f"{'AB'[i] if j < 2 else 'Same rise '+ 'AB'[i]} · up to {100*cap:g}% slip")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="upper center")
    fig.supxlabel("Slip ratio")
    fig.supylabel("Friction coefficient")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output/"figures"/"historical_pairs.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
