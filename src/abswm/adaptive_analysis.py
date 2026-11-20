"""Frozen-model diagnostics, validation calibration and matched feedback interventions."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from .active_evaluation import populate
from .active_experiment import accelerate
from .active_information import entropy, log_weights
from .active_metrics import LEVELS, VARIABLES, mean_rows, mixture_scores, risk_bins, risk_summary, values_for
from .active_rollout import NeuralCurvePredictor, RecurrentPredictor, candidate_actions, grid_coefficient, integrate
from .adaptive_experiment import control_options, create_controller, observer_for, table
from .adaptive_simulation import DriveStart, action_error, drive, dynamic_events
from .adaptive_training import window_history
from .feedback_control import feedback_plan
from .physics import slip
from .spatial_road import FAMILIES, SpatialRoad, scene


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def previous_safety(paths):
    closed = read_rows(paths.legacy/"closed_loop_by_road.csv")
    keys = {"C2", "C3", "C4", "C5", "C6", "C8", "C11", "C12", "C13"}
    table(paths.output, "verified_3_0_closed", [r for r in closed if r["method"] in keys])
    with (paths.legacy/"safety_events.csv").open(encoding="utf-8", newline="") as stream:
        groups = Counter((r["method"], r["reason"]) for r in csv.DictReader(stream))
    table(paths.output, "verified_3_0_safety", [{"method": k, "reason": reason, "count": n}
                                               for (k, reason), n in sorted(groups.items())])


def numerical_trials(c, d, old, models, roads, output, seeds):
    from .active_data import history_windows

    suite = history_windows(c, d, [s["segments"][0] for s in roads], 0, d["seed"]+2, all_roads=True)
    action_options = control_options(c, d, old)
    rows = []
    for j, history in enumerate(suite["history"]):
        road = SpatialRoad.build(scene("constant", [suite["roads"][j]]), c)
        state, previous = history[-1, :2], history[-1, 3]
        actions = candidate_actions(previous, d["total_horizon"], c)[0]
        truth, _ = road.forecast([*state, 0.], actions, stop=0.)
        for seed in seeds:
            e4 = NeuralCurvePredictor(c, [models["e4", seed]])
            e4.observe(history, 0)
            recurrent = RecurrentPredictor(c, models["recurrent", seed])
            recurrent.observe(history, 0)
            forecasts = {
                "e4_euler": integrate(state, actions[None], grid_coefficient(e4.curves), c),
                "e4_rk4": integrate(state, actions[None], grid_coefficient(e4.curves), c, rk4=True),
                "original_recurrent": recurrent.predict(state, actions),
                "truth_euler": road.forecast([*state, 0.], actions, rk4=False, stop=0.)[0],
                "truth_rk4": truth,
            }
            for name, predicted in forecasts.items():
                for event_mode in ("substep", "endpoint"):
                    if name == "original_recurrent" and event_mode == "substep":
                        continue
                    high = predicted.high[0] if event_mode == "substep" else (predicted.slip[0] >= .8).any(axis=-1)
                    lock = predicted.lock[0] if event_mode == "substep" else (predicted.state[0, ..., 1] <= .5).any(axis=-1)
                    score = action_error(truth, predicted, [1.], actions, action_options, previous, d["tie_margin_mu"])
                    rows.append({"road_id": suite["roads"][j]["id"], "method": name, "seed": seed, "window": j,
                                 "events": event_mode, "impulse_abs_error": float(np.mean(np.abs(predicted.impulse-truth.impulse))),
                                 "true_high": int(truth.high.sum()), "true_lock": int(truth.lock.sum()),
                                 "missed_high": int((truth.high[0] & ~high).sum()),
                                 "missed_lock": int((truth.lock[0] & ~lock).sum()), **score})
    table(output, "numerical_consistency", rows)


def choose_innovation(c, d, episodes, policy, models, scenes, output):
    cases = {r["id"]: r for r in scenes}
    rows = []
    for scale in d["innovation_scales"]:
        for threshold in d["innovation_thresholds"]:
            for episode in episodes:
                case = episode["metrics"]
                road = SpatialRoad.build(cases[case["road_id"]], c)
                observer = observer_for("prototype", c["seeds"][0], c, d, policy, models,
                                        revision="R2", refinement="none")
                observer.scale, observer.detector.threshold = scale, threshold
                trace = []
                for j, step in enumerate(episode["trace"]):
                    row = step | {}
                    observer.observe(window_history(episode["trace"], j, c["history_steps"]), j)
                    observer.predict_executed(step["applied"], j+1)
                    row.update(observer.last)
                    trace.append(row)
                by_change, false_detect, false_revision = dynamic_events(road, trace, d)
                reached = [r for r in by_change if r["reached"]]
                miss = sum(r["missed"] for r in reached)
                delay = sum(r["detect_delay_s"] or 0. for r in reached)
                valid = [r["innovation"] for r in trace if r.get("innovation") is not None]
                nll = float(np.mean(valid)+np.log(scale*np.sqrt(2*np.pi))) if valid else None
                guard = sum(bool(r.get("mismatch")) and (r.get("censored_residual") or 0.) > d["guard_sigma"] for r in trace)
                rows.append({"scale": scale, "threshold": threshold, "road_id": case["road_id"],
                             "false_detections": false_detect, "false_revisions": false_revision,
                             "reached": len(reached), "miss": miss, "detection_delay_s_sum": delay,
                             "steps": len(trace), "time_s": case["time_s"], "num_untrusted": guard,
                             "predictive_nll": nll})
    summaries = []
    for scale in d["innovation_scales"]:
        for threshold in d["innovation_thresholds"]:
            group = [r for r in rows if r["scale"] == scale and r["threshold"] == threshold]
            clocks = sum(r["time_s"] for r in group)
            reached = max(1, sum(r["reached"] for r in group))
            missing = sum(r["miss"] for r in group)/reached
            false = sum(r["false_revisions"] for r in group)/max(clocks, .01)
            delay = sum(r["detection_delay_s_sum"] for r in group)/reached
            nll = float(np.mean([r["predictive_nll"] for r in group if r["predictive_nll"] is not None]))
            guards = sum(r["num_untrusted"] for r in group)/max(1, sum(r["steps"] for r in group))
            cost = 2*missing+delay+.5*false+.15*nll+.3*guards
            summaries.append({"scale": scale, "threshold": threshold, "miss_rate": missing,
                              "mean_delay_per_reached_s": delay, "false_revision_per_s": false,
                              "nll": nll, "untrusted_fraction": guards, "cost": cost})
    table(output, "innovation_validation", summaries)
    table(output, "innovation_validation_by_road", rows)
    return min(summaries, key=lambda r: r["cost"])


def open_samples(c, d, scenes, *, smoke=False, dynamic_episodes=(), full=True):
    from .active_data import history_windows

    for k, family in enumerate(FAMILIES):
        roads = [s["segments"][0] for s in scenes if s["kind"] == "constant" and s["family"] == family]
        if not roads:
            continue
        suite = history_windows(c, d, roads, 0, d["seed"]+310+k, all_roads=True)
        indices = range(0, len(suite["history"]), 3 if not full else 1)
        for i in list(indices)[:2] if smoke else indices:
            road = suite["roads"][i]
            spatial = SpatialRoad.build(scene("constant", [road]), c)
            yield spatial, suite["history"][i], suite, i, 0., f"{road['id']}_{i}"
    by_id = {s["id"]: s for s in scenes if s["kind"] != "constant"}
    for entry in dynamic_episodes:
        trace = entry["trace"]
        key = entry["metrics"]["road_id"]
        if key not in by_id or not trace:
            continue
        spatial = SpatialRoad.build(by_id[key], c)
        near = [next((i for i, r in enumerate(trace) if r["position_m"] >= s), len(trace)-1)
                for s in spatial.starts]
        for i in sorted({j for anchor in near for j in (anchor-5, anchor+2, anchor+10, anchor+30) if 1 <= j < len(trace)}):
            yield spatial, window_history(trace, i, c["history_steps"]), trace, i, trace[i]["position_m"], f"{key}_{entry['metrics']['initial_speed']}_{i}"


def historical_samples(c, paths):
    from .active_data import historical_stress
    from .decoder_study import read_json

    original = read_json(paths.base/"data"/"manifest.json")["roads"]
    for name, suite in historical_stress(c, original, read_json(Path("configs/physics_prior.json"))):
        for i, road in enumerate(suite["roads"]):
            spatial = SpatialRoad.build(scene("constant", [road]), c)
            yield spatial, suite["history"][i], suite, i, 0., f"{name}_{road['id']}_{i}"


def _warm(observer, c, series, index):
    if isinstance(series, dict):
        return populate(observer, series, index, c, ticks=36)
    for j in range(max(0, index-64), index+1):
        observer.observe(window_history(series, j, c["history_steps"]), j)
    return observer


def _forecast_model(method, c, d, policy, models, seed, h, series, i, actions):
    if method in {"e4", "e4_closed"}:
        predictor = accelerate(NeuralCurvePredictor(c, [models[method, seed]]))
        predictor.observe(h, 0)
    else:
        kind = method if method in {"prototype", "continuous", "ensemble", "library"} else policy["source"]
        trained = None if method in {"prototype", "continuous", "ensemble", "library", "frozen", "untrained"} else policy.get("trained")
        if method.startswith("trained_"):
            trained = method.removeprefix("trained_")
        revision = "R0" if method == "frozen" else policy.get("revision", "R2")
        refinements = {"full": "full", "subspace": "subspace", "selected": "selected"}
        refine = refinements.get(method, policy.get("refinement", "none") if method == "adaptive" else "none")
        predictor = observer_for(kind, seed, c, d, policy, models, revision=revision,
                                 refinement=refine, trained=trained)
        _warm(predictor, c, series, i)
    return predictor, predictor.predict(h[-1, :2], actions)


def open_predictions(c, d, old, policy, models, samples, methods, seeds, *, dynamic_preview=True):
    for spatial, h, series, i, position, identifier in samples:
        actions = candidate_actions(h[-1, 3], d["total_horizon"], c)[0]
        truth = spatial.forecast([*h[-1, :2], position], actions, preview=dynamic_preview)[0]
        regime = np.max(h[:, 2])
        scene = spatial.spec
        tag = "weak" if regime < .04 else "strong" if regime >= .16 else "intermediate"
        family = scene["family"]
        pool = {part["family"] for part in scene["segments"]}
        info = {"case": identifier, "road_id": scene["id"], "suite": family if scene["kind"] == "constant" else scene["kind"],
                "family": family, "kind": scene["kind"], "families": "+".join(sorted(pool)),
                "excitation": tag, "position_m": position}
        for method in methods:
            trial_seeds = seeds[:1] if method in {"ensemble", "library"} else seeds
            for seed in trial_seeds:
                predictor, forecast = _forecast_model(method, c, d, policy, models, seed, h, series, i, actions)
                measure = action_error(truth, forecast, predictor.weights, actions, control_options(c, d, old),
                                       h[-1, 3], d["tie_margin_mu"])
                yield info | {"method": method, "seed": seed}, predictor, truth, forecast, measure


def _risk_probability(value, transform):
    temperature, offset = transform
    p = np.clip(value, 1e-4, 1-1e-4)
    return expit(np.log(p/(1-p))/temperature+offset)


def calibrate(default_scales, records):
    nominal = np.asarray(default_scales)
    factors = (.75, 1., 1.25, 1.5, 2.)
    scales = []
    for q, initial in enumerate(nominal):
        def criterion(factor):
            sigma = initial*factor
            scores = []
            for _, belief, truth, forecast, _ in records:
                y = values_for(truth)[0, :, -1, q]
                x = values_for(forecast)[:, :, -1, q]
                joint = log_weights(belief.weights)[:, None]-.5*((x-y)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))
                scores.append(-logsumexp(joint, axis=0).mean())
            return np.mean(scores)

        scales.append(float(initial*min(factors, key=criterion)))
    risks = {}
    for name in ("high", "lock"):
        p = np.concatenate([b.weights@getattr(f, name) for _, b, _, f, _ in records])
        y = np.concatenate([getattr(t, name)[0] for _, _, t, _, _ in records])
        candidates = [(t, o) for t in (.75, 1., 1.3, 1.65) for o in (-.75, 0., .75)]
        risks[name] = min(candidates, key=lambda pair: np.mean((_risk_probability(p, pair)-y)**2))
    return {"raw_scales": nominal.tolist(), "scales": scales, "risk": risks}


def summarize_open(records, calibration, output, phase):
    decisions, probabilities, events, spread = [], [], [], []
    for info, observer, truth, forecast, decision in records:
        decisions.append(info | decision)
        estimates = values_for(forecast)[:, :, -1]
        actual = values_for(truth)[0, :, -1]
        averaging = observer.weights@observer.curves if hasattr(observer, "curves") else None
        sd = None if averaging is None else float(np.sqrt(np.mean(observer.weights@(observer.curves-averaging)**2)))
        mean_force = observer.weights@estimates[..., -1]
        spread.append(info | {"ambiguity_curve_sd": sd, "mean_force_n": mean_force.mean(),
                              "entropy": float(entropy(observer.weights)), "residual_force_n": calibration["scales"][3],
                              "ess": float(1/(observer.weights@observer.weights))})
        for flavor, sigmas in (("raw", calibration["raw_scales"]), ("calibrated", calibration["scales"])):
            row, crps, inside, width = mixture_scores(estimates, actual, observer.weights, sigmas)
            for j, name in enumerate(VARIABLES):
                for level, covers, widths in zip(LEVELS, inside, width):
                    probabilities.append(info | {"flavor": flavor, "variable": name, "nominal": level,
                                                  "coverage": covers[:, j].mean(), "width": widths[:, j].mean(),
                                                  "nll": row[:, j].mean(), "crps": crps[:, j].mean()})
            for name in ("high", "lock"):
                p = observer.weights@getattr(forecast, name)
                if flavor == "calibrated":
                    p = _risk_probability(p, calibration["risk"][name])
                events.extend(info | {"event": name, "flavor": flavor, "method": info["method"]+"_"+flavor,
                                       "probability": q, "observed": obs}
                              for q, obs in zip(p, getattr(truth, name)[0]))
    metrics = mean_rows(probabilities, ("suite", "kind", "method", "excitation", "flavor", "variable", "nominal"),
                        ("coverage", "width", "nll", "crps"))
    by_key = {}
    for r in spread:
        by_key.setdefault((r["method"], r["case"]), []).append(r)
    for group in by_key.values():
        epistemic = np.std([r["mean_force_n"] for r in group]) if len(group) > 1 else None
        for r in group:
            r["seed_mean_force_sd_n"] = epistemic
    risks, bins = risk_summary(events), risk_bins(events)
    table(output, phase+"action_prediction", decisions)
    table(output, phase+"action_regret", [{k: v for k, v in r.items()
                                           if k in {"method", "road_id", "case", "kind", "suite", "seed", "chosen", "top1",
                                                    "violation", "pairs", "correct_pairs", "pure_regret_ns", "action_regret", "risk_regret"}}
                                          for r in decisions])
    table(output, phase+"probability_calibration", metrics)
    table(output, phase+"uncertainty_decomposition", spread)
    table(output, phase+"risk_evaluation", risks)
    table(output, phase+"risk_calibration", bins)
    return decisions, metrics, risks


def feedback_factorial(c, d, old, policy, paths, models, gains, entries, seeds):
    options = control_options(c, d, old)
    output, branch_rows, forecast_rows = [], [], []
    for spatial, entry, anchor, label in entries:
        trail = entry["trace"]
        h = window_history(trail, anchor, c["history_steps"])
        x = [*h[-1, :2], trail[anchor]["position_m"]]
        initial = [entry["metrics"][n] for n in ("initial_speed", "initial_slip", "initial_torque")]
        for seed in seeds:
            for length in d["commit_steps"]:
                base = create_controller("B13", seed, c, d, spatial, policy, models, gains, old, length)
                _warm(base.predictor, c, trail, anchor)
                bank = base.predictor.snapshot()
                plan = feedback_plan(x[:2], h[-1, 3], bank, c, d, options, length, "feedback",
                                     base.predictor.scale, detail=True)
                meta = {"road_id": spatial.spec["id"], "kind": spatial.spec["kind"], "seed": seed,
                        "anchor": anchor, "regime": label, "initial_speed": initial[0], "length": length}
                for idx, (segment, objective) in enumerate(zip(plan.segments, plan.objective)):
                    forecast_rows.append(meta | {"first": idx, "segment_nm": segment.tolist(), "value": objective,
                                                "frozen": plan.frozen[idx], "voi": plan.value[idx], "eig": plan.gain[idx],
                                                "high": plan.high[idx], "lock": plan.lock[idx], "predicted_impulse": plan.impulse[idx],
                                                "selected": idx == plan.selected, "matched": idx == plan.matched})
                branch_rows.extend(meta | r for r in plan.branches)
                start = DriveStart(anchor, x[2], h)
                outcomes = {}
                for name, selected in (("normal", plan.matched), ("active", plan.selected)):
                    for enabled in (False, True):
                        ctl = create_controller("B13", seed, c, d, spatial, policy, models, gains, old, length)
                        _warm(ctl.predictor, c, trail, anchor)
                        ctl.forced_prefix = plan.segments[selected]
                        freeze = None if enabled else anchor+d["factorial_freeze_steps"]
                        record = drive(spatial, ctl, initial, options, start=start, freeze_until=freeze)
                        series = record["trace"]
                        cuts = {}
                        for k in d["factorial_windows"]:
                            end = min(length+k, len(series))
                            values = series[length:end]
                            impulse = c["mass"]*sum(r["v"]-r["next_v"] for r in values)
                            cuts.update({f"after_{k*5}_impulse_ns": impulse,
                                         f"after_{k*5}_high_s": sum(r["high_s"] for r in values)})
                        if len(series) > length:
                            current = series[length]
                            post = create_controller("B13", seed, c, d, spatial, policy, models, gains, old, length).predictor
                            _warm(post, c, trail, anchor)
                            for j in range(1, length+1):
                                row = series[j]
                                causal = np.array([[r["v"], r["omega"], r["slip"], r["previous"]] for r in series[1:j+1]])
                                history = np.concatenate((h, causal), axis=0)[-c["history_steps"]:]
                                post.observe(history, row["step"], evidence=enabled)
                            follow = candidate_actions(current["previous"], d["total_horizon"], c)[0]
                            state = [current["v"], current["omega"]]
                            truth = spatial.forecast([*state, current["position_m"]], follow)[0]
                            prediction = post.predict(state, follow)
                            cuts.update(action_error(truth, prediction, post.weights, follow, options,
                                                     current["previous"], d["tie_margin_mu"]))
                            cuts["observed_mu"] = [r["observed_interval_mu"] for r in series[:length]]
                            cuts["post_entropy"] = float(entropy(post.weights))
                        row = meta | {"first_action": name, "real_evidence": enabled,
                                      "planned_exploration": plan.selected != plan.matched,
                                      "predicted_voi": plan.value[selected],
                                      "first_impulse_ns": c["mass"]*sum(r["v"]-r["next_v"] for r in series[:length]),
                                      "interruption": any(r["safety"] for r in series[:length]),
                                      **record["metrics"], **cuts}
                        outcomes[name, enabled] = row
                        output.append(row)
                for name in ("normal", "active"):
                    updated, fixed = outcomes[name, True], outcomes[name, False]
                    value = (fixed["distance_m"]-updated["distance_m"] if updated["stopped"] and fixed["stopped"] else None)
                    branch_rows.append(meta | {"realized_first": name, "distance_gain_from_update_m": value,
                                               "actual_interval_gain_100_ns": updated["after_100_impulse_ns"]-
                                               fixed["after_100_impulse_ns"],
                                               "predicted_voi": updated["predicted_voi"]})
    table(paths.output, "feedback_planning", forecast_rows)
    table(paths.output, "predicted_voi", [r for r in forecast_rows if r["selected"] or r["matched"]])
    table(paths.output, "observation_branches", [r for r in branch_rows if "branch" in r])
    table(paths.output, "exploration_factorial", output)
    table(paths.output, "realized_voi", [r for r in branch_rows if "realized_first" in r])
