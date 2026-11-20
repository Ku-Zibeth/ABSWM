"""Frozen open-loop scoring, observer ablations and prospective information audits."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .active_belief import BeliefObserver, make_observer
from .active_data import historical_stress, history_windows
from .active_evaluation import options_for, populate
from .active_experiment import accelerate
from .active_fast import static_forecast
from .active_information import entropy, information_values, likelihood_posterior
from .active_metrics import LEVELS, VARIABLES, mean_rows, mixture_scores, ranking_rows, risk_bins, risk_summary, values_for
from .active_rollout import GRID, NLSPredictor, NeuralCurvePredictor, OraclePredictor, RecurrentPredictor, candidate_actions, utilities
from .decoder_study import read_json
from .evaluation import write_csv
from .tire_prior import friction_query, history_evidence, refine
from .tire_prior_data import tire_function


REFINEMENTS = {
    "no_update": {"update": False},
    "full_refinement": {"update": False, "refinement": "full"},
    "subspace_refinement": {"update": False, "refinement": "subspace"},
    "selected_refinement": {"update": False, "refinement": "selected"},
    "bayes_refinement": {"refinement": "selected"},
    "no_memory": {"memory": False},
}


def observer_set(c, d, models, policy, seed, *, historical, smoke):
    if seed == c["seeds"][0]:
        yield "nls", NLSPredictor(c)
        yield "ensemble", NeuralCurvePredictor(c, [models["e4", s] for s in c["seeds"]])
        yield "library", make_observer(c, None, policy)
    yield "e4", NeuralCurvePredictor(c, [models["e4", seed]])
    yield "structured", NeuralCurvePredictor(c, [models["structured", seed]])
    if not historical:
        yield "original_recurrent", RecurrentPredictor(c, models["recurrent", seed])
        yield "structured_mixed", NeuralCurvePredictor(c, [models["structured_mixed", seed]])
    name = policy["variant"]
    selected = models[name, seed]
    yield "learned", make_observer(c, selected, policy)
    for mode, args in REFINEMENTS.items():
        yield mode, make_observer(c, selected, policy, **args)
    if not historical:
        for variant in d["variants"]:
            if variant != name:
                yield variant, make_observer(c, models[variant, seed], policy)


def deterministic_refinement(c, structured, history, d):
    p = NeuralCurvePredictor(c, [structured])
    h = torch.as_tensor(history[None], dtype=torch.float32)
    with torch.no_grad():
        z = structured.encode(h)
        evidence = history_evidence(h, c)
    checkpoints = refine(structured, z, evidence, d, (5,))
    raw, residual, _ = checkpoints[5]
    grid = torch.as_tensor(GRID[None], dtype=h.dtype)
    p.curves = friction_query(structured, raw, grid).detach().numpy()
    return p, float(residual[0])


def curve_scores(observer, actual, coverage):
    if not hasattr(observer, "curves"):
        return {}
    curves, weights = observer.curves, observer.weights
    weighted = weights@curves
    rmse = np.sqrt(np.mean((weighted-actual)**2))
    out = GRID > coverage
    best = GRID[np.argmax(weighted)]
    true = GRID[np.argmax(actual)]
    height = actual.max()
    widths = (weighted >= .95*weighted.max()).sum()*np.diff(GRID)[0]
    true_width = (actual >= .95*height).sum()*np.diff(GRID)[0]
    unweighted = np.sqrt(np.mean(np.var(curves, axis=0)))
    weighted_spread = np.sqrt(np.mean(weights@((curves-weighted)**2)))
    return {"curve_rmse": float(rmse), "unseen_curve_rmse": float(np.sqrt(np.mean((weighted[out]-actual[out])**2))),
            "weighted_curve_sd": float(weighted_spread), "unweighted_curve_sd": float(unweighted),
            "peak_slip_error": float(abs(best-true)), "peak_friction_error": float(abs(weighted.max()-height)),
            "initial_stiffness_error": float(abs((weighted[1]-actual[1])/GRID[1])),
            "peak_width_error": float(abs(widths-true_width)),
            "post_peak_ratio_error": float(abs(weighted[-1]/max(weighted.max(), 1e-10)-actual[-1]/height)),
            "friction_regret": float((height-actual[np.argmax(weighted)])/height),
            "ess": float(1/(weights@weights)), "entropy": float(entropy(weights))}


def distribution_baseline(observer, actual, prediction, horizons, sigma, info):
    idx = np.array(horizons)-1
    pred = values_for(prediction)[..., idx, :]
    truth = values_for(actual)[0][..., idx, :]
    nll, crps, covered, widths = mixture_scores(pred, truth, observer.weights, sigma)
    for step, horizon in enumerate(horizons):
        for i, name in enumerate(VARIABLES):
            for j, level in enumerate(LEVELS):
                yield info | {"step_count": horizon, "variable": name, "nominal": level,
                              "coverage": float(covered[j, :, step, i].mean()),
                              "width": float(widths[j, :, step, i].mean()),
                              "nll": float(nll[:, step, i].mean()), "crps": float(crps[:, step, i].mean())}


def probability_only_forecast(observer, actions, current, actual, horizons, info, summaries):
    before = perf_counter()
    prediction = observer.predict(current, actions)
    elapsed = 1000*(perf_counter()-before)
    summaries["offline_timing"].append(info | {"component": "multi_candidate_rollout", "ms": elapsed})
    if info["method"] == "learned" and info["window"] % 8 == 0:
        before = perf_counter()
        observer.predict(current, actions[:1])
        summaries["offline_timing"].append(info | {"component": "single_candidate_rollout", "ms": 1000*(perf_counter()-before)})
    truth = values_for(actual)[0]
    expected = np.einsum("k,kath->ath", observer.weights, values_for(prediction))
    for n in horizons:
        diff = expected[:, :n]-truth[:, :n]
        pred_delta = expected[1:, n-1]-expected[0, n-1]
        true_delta = truth[1:, n-1]-truth[0, n-1]
        rmse = np.sqrt(np.mean(diff**2, axis=(0, 1)))
        act = np.sqrt(np.mean((pred_delta-true_delta)**2, axis=0))
        summaries["action_prediction"].append(info | {"step_count": n, **dict(zip(VARIABLES, rmse)),
                                                        **{f"response_{k}": v for k, v in zip(VARIABLES, act)}})
    return prediction


def add_rank_and_risk(c, d, policy, history, observer, options, actions, truth, prediction, info, summaries):
    score = utilities(prediction, actions, history[-1, 3], options)
    oracle = utilities(truth, actions, history[-1, 3], options)[0]
    weighted = observer.weights@score
    chosen = int(np.argmax(weighted))
    reward = observer.weights@prediction.interval_mu.mean(axis=-1)
    row = ranking_rows(truth, reward, chosen, oracle, truth.high[0], truth.lock[0], d["tie_margin_mu"])
    summaries["action_ranking"].append(info | {k: row[k] for k in ("pairs", "correct_pairs", "top1", "top1_reward_tie")})
    summaries["action_regret"].append(info | row)
    events = {"high_slip": (prediction.high, truth.high[0]), "lock": (prediction.lock, truth.lock[0]),
              "safe_action": (~(prediction.high | prediction.lock), ~(truth.high[0] | truth.lock[0]))}
    for limit in (.4, .6, .9):
        events[f"slip_{limit:.1f}"] = (prediction.max_slip.max(axis=-1) >= limit,
                                      truth.max_slip[0].max(axis=-1) >= limit)
    for event, (pred, true) in events.items():
        probabilities = observer.weights@pred
        summaries["risks"].extend(info | {"event": event, "action": a, "probability": float(p),
                                           "observed": bool(y)} for a, (p, y) in enumerate(zip(probabilities, true)))
    return score, oracle, chosen


def information_audit(c, options, observer, prediction, truth, score, oracle, actions, info):
    rows = []
    for n in sorted({1, min(3, len(actions[0])), len(actions[0])}):
        observation = prediction.interval_mu[..., :n].mean(axis=-1)
        h, value = information_values(observation, observer.weights, score, observer.scale/np.sqrt(n))
        actual = truth.interval_mu[0, :, :n].mean(axis=-1)
        weights = observer.weights.copy()
        post = likelihood_posterior(weights, actual[:, None]-observation.T, observer.scale/np.sqrt(n))
        selected = int(np.argmax(weights@score))
        for a in range(len(actions)):
            decided = int(np.argmax(post[a]@score))
            feedback_value = None
            if n == 1 and info["window"] % 8 == 0 and actions.shape[1] > 1:
                # Keep the actual commitment at one step. For each hypothesized
                # future state, score all feasible continuation sequences.
                next_actions = candidate_actions(actions[a, 0], actions.shape[1]-1, c)[0]
                continuation = []
                for k, curve in enumerate(observer.curves):
                    current = prediction.state[k, a, 0]
                    result = static_forecast(current, next_actions, curve, c)
                    continuation.append(utilities(result, next_actions, actions[a, 0], options)[0])
                _, feedback = information_values(observation[:, a:a+1], weights, np.array(continuation), observer.scale)
                feedback_value = float(feedback[0])
            rows.append(info | {"observation_steps": n, "candidate": a, "gain_nats": float(h[a]),
                                "decision_value": float(value[a]), "finite_feedback_decision_value": feedback_value,
                                "entropy_before": float(entropy(weights)),
                                "realized_hypothetical_entropy_drop": float(entropy(weights)-entropy(post[a])),
                                "post_observation_action": decided,
                                "post_observation_utility_delta_same_start": float(oracle[decided]-oracle[selected]),
                                "true_current_action_cost_vs_passive": float(oracle[selected]-oracle[a])})
    return rows


def case_predictions(c, d, policy, models, name, suite, indices, summaries, figure_cases, smoke):
    previous_options = read_json(Path("configs/physics_prior.json"))
    historical = name.startswith("historical")
    primary_horizon = policy["horizon"]
    long_horizon = max(primary_horizon, policy["study"]["training_steps"])
    if historical:
        long_horizon = primary_horizon
    horizons = sorted({1, min(5, long_horizon), min(10, long_horizon), long_horizon})
    options = options_for(c, d, primary_horizon, policy["observation_scale"])
    sigma = policy["predictive_scale"]
    for i in indices:
        history, road = suite["history"][i], suite["roads"][i]
        coverage = float(suite["achieved"][i])
        base = {"suite": name, "road_id": road["id"], "family": road["family"], "tag": road["subfamily"],
                "window": int(i), "coverage": coverage,
                "excitation": "weak" if coverage < .04 else "strong" if coverage >= .16 else "intermediate"}
        actions = candidate_actions(history[-1, 3], long_horizon, c)[0]
        decision_actions = candidate_actions(history[-1, 3], primary_horizon, c)[0]
        oracle = accelerate(OraclePredictor(c, road), road)
        decision_truth = oracle.predict(history[-1, :2], decision_actions)
        truth = decision_truth if long_horizon == primary_horizon else oracle.predict(history[-1, :2], actions)
        actual_curve = tire_function([road])(GRID[None])[0]
        for seed in c["seeds"][:1] if smoke else c["seeds"]:
            items = list(observer_set(c, d, models, policy, seed, historical=historical, smoke=smoke))
            if historical or i % 4 == 0:
                observer, residual = deterministic_refinement(c, models["structured", seed], history, previous_options)
                items.append(("deterministic_refinement", observer))
            for method, predictor in items:
                info = base | {"method": method, "seed": seed}
                observer = accelerate(predictor)
                if isinstance(observer, (BeliefObserver, RecurrentPredictor)):
                    recorded, corrections = [], []

                    def collect(o):
                        if not isinstance(o, BeliefObserver):
                            return
                        if o.last_event is not None:
                            recorded.append(vars(o.last_event))
                        if o.last_refine is not None:
                            corrections.append(o.last_refine)

                    # Each batch starts with a causal prefix. New rows carry their actual executed torque.
                    populate(observer, suite, i, c, collect=collect)
                    if recorded and method in {"learned", "bayes_refinement", "library"}:
                        summaries["belief_update"].extend(info | row for row in recorded)
                    if corrections:
                        keys = ("rank", "before", "after", "movement", "max_eigenvalue")
                        values = {k: [row[k] for row in corrections if row.get(k) is not None] for k in keys}
                        summaries["refinement"].append(info | {"updates": len(corrections),
                                                                **{k: np.mean(v) if v else None for k, v in values.items()}})
                elif method != "deterministic_refinement":
                    observer.observe(history, c["history_steps"])
                curve = curve_scores(observer, actual_curve, coverage)
                if curve:
                    summaries["curve"].append(info | curve)
                if method == "deterministic_refinement":
                    summaries["refinement"].append(info | {"updates": 1, "rank": None, "before": None,
                                                            "after": np.sqrt(residual), "movement": None, "max_eigenvalue": None})
                prediction = probability_only_forecast(observer, actions, history[-1, :2],
                                                        truth, horizons, info, summaries)
                decision_forecast = (prediction if primary_horizon == long_horizon else
                                     observer.predict(history[-1, :2], decision_actions))
                if method in {"learned", "no_update", "e4", "ensemble", "library", "bayes_refinement"}:
                    summaries["prob"].extend(distribution_baseline(observer, decision_truth, decision_forecast,
                                             [n for n in horizons if n <= primary_horizon], sigma, info))
                score, offline_utility, chosen = add_rank_and_risk(c, d, policy, history, observer, options,
                                                                 decision_actions, decision_truth, decision_forecast,
                                                                 info, summaries)
                if method == "learned" and seed == c["seeds"][0]:
                    summaries["information"].extend(information_audit(c, options, observer, decision_forecast,
                                                    decision_truth, score, offline_utility, decision_actions, info))
                    group = (name, road["subfamily"], base["excitation"])
                    if group not in figure_cases:
                        figure_cases[group] = {"history": history, "curves": observer.curves, "weights": observer.weights,
                                               "actual_curve": actual_curve, "forecast": values_for(decision_forecast),
                                               "truth": values_for(decision_truth)[0], "torque": decision_actions,
                                               "info": info, "chosen": chosen}


def run_open_test(c, d, partitions, original, policy, args, models):
    output = args.output
    protocol = read_json(output/"training_protocol.json")
    config = d | {"variants": protocol["variants"]}
    summaries, figures = defaultdict(list), {}
    seed = d["partition_seed"]+170
    for i, family in enumerate(("burckhardt", "magic", "rational")):
        roads = partitions[f"test_{family}"]
        suite = history_windows(c, d, roads, 0, seed+i, all_roads=True)
        indices = np.arange(len(suite["history"]))
        if args.smoke:
            indices = indices[:3]
        print(f"Open-loop {family}: {len(indices)} windows", flush=True)
        case_predictions(c, config, policy, models, family, suite, indices, summaries, figures, args.smoke)
    if not args.smoke:
        for name, suite in historical_stress(c, original, read_json(Path("configs/physics_prior.json"))):
            indices = np.arange(len(suite["history"]))
            print(f"Open-loop {name}: {len(indices)} windows", flush=True)
            case_predictions(c, config, policy, models, name, suite, indices, summaries, figures, False)
    mapping = {"action_prediction": "action_prediction", "action_ranking": "action_ranking", "action_regret": "action_regret",
               "belief_update": "belief_update", "refinement": "refinement_comparison", "curve": "curve_diagnostics",
               "information": "information_gain", "offline_timing": "offline_timing"}
    for key, name in mapping.items():
        if summaries[key]:
            write_csv(output/f"{name}.csv", summaries[key])
    risk = risk_summary(summaries["risks"])
    write_csv(output/"risk_evaluation.csv", risk)
    bins = risk_bins(summaries["risks"])
    write_csv(output/"risk_calibration.csv", bins)
    probability = mean_rows(summaries["prob"], ("suite", "method", "excitation", "step_count", "variable", "nominal"),
                            ("coverage", "width", "nll", "crps"))
    write_csv(output/"probability_calibration.csv", probability)
    if not args.smoke:
        from .active_representation import audit_representation
        audit_representation(c, models, original, output)
    return {"risk": risk, "bins": bins, "probability": probability, "scores": summaries, "figures": figures}
