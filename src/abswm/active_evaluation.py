"""Validation-only selection, frozen counterfactual and closed-loop evaluations."""
from __future__ import annotations

import itertools
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .active_belief import make_observer
from .active_data import historical_stress, history_prefix, history_windows
from .active_experiment import (CONTROLLERS, accelerate, digest, load_baselines, road_selection, run_closed)
from .active_information import entropy, information_values, log_weights
from .active_metrics import (LEVELS, VARIABLES, control_summary, mean_rows, mixture_scores, paired_control,
                             ranking_rows, risk_bins, risk_summary, values_for)
from .active_rollout import GRID, NLSPredictor, NeuralCurvePredictor, OraclePredictor, RecurrentPredictor, candidate_actions, utilities
from .active_training import load_trained
from .config import write_json
from .decoder_study import read_json
from .evaluation import write_csv
from .tire_prior_data import tire_function


ABLATIONS = ("no_bayes", "no_memory", "information_refinement", "generic_eig", "fixed_evoi",
             "library_active", "library_passive")
CONTRASTS = (("C6", "C2"), ("C6", "C5"), ("C7", "C6"), ("C8", "C3"), ("C9", "C8"),
             ("C10", "C9"), ("C11", "C6"), ("C12", "C11"), ("C13", "C11"),
             ("C13", "generic_eig"), ("C13", "fixed_evoi"), ("C13", "library_active"),
             ("library_active", "library_passive"), ("no_bayes", "C11"), ("no_memory", "C11"),
             ("information_refinement", "C11"))
NO_SEED = {"C0", "C1", "C3", "C4", "C5", "library_active", "library_passive"}


def options_for(c, d, horizon, scale=None):
    options = d["options"] | {"horizon": horizon, "slew_nm": c["control_dt"]*c["torque_rate"]}
    if scale is not None:
        options["observation_scale"] = scale/np.sqrt(d["likelihood_reliability"])
    return options


def forecasts_on_validation(c, d, partitions, model, policy, limit):
    seed = d["partition_seed"]+734
    roads = partitions["validation_burckhardt"][:12]+partitions["validation_magic"][:6]
    suite = history_windows(c, d, roads, limit, seed)
    horizon = policy["horizon"]
    for i, history in enumerate(suite["history"]):
        observer = make_observer(c, model, policy)
        populate(observer, suite, i, c)
        actions = candidate_actions(history[-1, 3], horizon, c)[0]
        oracle = accelerate(OraclePredictor(c, suite["roads"][i]), suite["roads"][i])
        actual = oracle.predict(history[-1, :2], actions)
        forecast = observer.predict(history[-1, :2], actions)
        yield observer, history, actions, actual, forecast


def populate(observer, suite, i, c, *, ticks=30, collect=None):
    origin, initial, follow = history_prefix(suite, i, ticks, c)
    observer.observe(initial, origin)
    for offset in range(1, len(follow)):
        past = np.concatenate((initial, follow[1:offset+1]), axis=0)[-c["history_steps"]:]
        observer.observe(past, origin+offset)
        if collect is not None:
            collect(observer)
    return observer


def select_policy(c, d, paths, partitions, p0, args, models, protocol):
    output = args.output
    training = read_training(output)
    by_variant = mean_rows(training, ("method",), ("val_score", "val_rollout", "val_action", "val_ranking", "val_risk"))
    write_csv(output/"loss_ablation_validation.csv", by_variant)
    variant = min(by_variant, key=lambda r: r["val_score"])["method"]
    horizon = round(p0["selected_horizon_ms"]/(1000*c["control_dt"]))
    policy = {"study": protocol["study"], "training_key": protocol["training_key"], "variant": variant,
              "horizon": horizon, "horizon_ms": p0["selected_horizon_ms"], "gamma": 0.,
              "acceptance": d["acceptance"], "primary_comparison": ["C13", "C11"],
              "probability_horizon_ms": p0["selected_horizon_ms"], "probability_threshold": .5}
    likelihood_results, prediction_cache = [], {}
    sample_count = 12 if args.smoke else 64
    for scale in d["observation_scales"]:
        rows = []
        for seed in c["seeds"]:
            for belief, _, _, true, prediction in forecasts_on_validation(
                    c, d, partitions, models[variant, seed], policy | {"observation_scale": scale}, sample_count):
                true_m = true.interval_mu[0, :, 0]
                error = (prediction.interval_mu[..., 0]-true_m)/belief.scale
                nll = -logsumexp(log_weights(belief.weights)[:, None]-.5*error**2-
                                 np.log(belief.scale*np.sqrt(2*np.pi)), axis=0)
                rows.append((belief.weights, values_for(prediction), values_for(true)[0],
                             nll.mean(), np.mean((belief.weights@prediction.high-true.high[0])**2)))
        loss = np.mean([r[3] for r in rows])
        likelihood_results.append({"scale": scale, "mean_nll": float(loss),
                                   "mean_brier": float(np.mean([r[4] for r in rows]))})
        prediction_cache[scale] = rows
    scale = min(likelihood_results, key=lambda r: r["mean_nll"])["scale"]
    write_csv(output/"likelihood_validation.csv", likelihood_results)
    calibrated = []
    rows = prediction_cache[scale]
    for i, nominal in enumerate((.1, 4., .05, 200.)):
        def nll(scale_log):
            sigma = nominal*np.exp(scale_log)
            return np.mean([(-logsumexp(log_weights(w)[:, None, None]-.5*((p[..., i]-y[..., i])/sigma)**2-
                                        np.log(sigma*np.sqrt(2*np.pi)), axis=0)).mean() for w, p, y, *_ in rows])

        result = minimize_scalar(nll, bounds=(-3, 3), method="bounded", options={"xatol": .02})
        calibrated.append(float(nominal*np.exp(result.x)))
    policy |= {"observation_scale": scale, "predictive_scale": calibrated}
    # Compare information weights on a fixed set of causal validation windows.
    # This score credits the oracle realized utility of the chosen action and
    # the short-term selection change only, leaving closed-loop testing separate.
    options = options_for(c, d, horizon, scale)
    gammas = (0., 1., 10., 100., 300.)
    runs = {gamma: [] for gamma in gammas}
    changes = dict.fromkeys(gammas, 0)
    for seed in c["seeds"]:
        for belief, history, actions, true, forecast in forecasts_on_validation(
                c, d, partitions, models[variant, seed], policy, sample_count):
            score = utilities(forecast, actions, history[-1, 3], options)
            oracle = utilities(true, actions, history[-1, 3], options)[0]
            expected = belief.weights@score
            pri = int(np.argmax(expected))
            _, voi = information_values(forecast.interval_mu[..., 0], belief.weights, score, belief.scale)
            regret = max(0., float(belief.weights@score.max(axis=1)-expected.max()))
            dynamic = (entropy(belief.weights)/np.log(len(belief.weights))*
                       np.clip((history[-1, 0]-c["stop_speed"])/15, 0, 1)*min(regret/.015, 1.))
            allowed = ((belief.weights@forecast.high <= options["explore_high_limit"]) &
                       (belief.weights@forecast.lock <= options["explore_lock_limit"]))
            for gamma in gammas:
                chosen = int(np.argmax(expected+gamma*dynamic*voi*allowed))
                runs[gamma].append(oracle[chosen])
                changes[gamma] += int(actions[chosen, 0] != actions[pri, 0])
    gamma_rows = [{"gamma": g, "oracle_utility": float(np.mean(runs[g])), "changed_first_actions": changes[g],
                   "windows": len(runs[g])} for g in gammas]
    write_csv(output/"exploration_validation.csv", gamma_rows)
    active = [r for r in gamma_rows if r["gamma"] > 0 and r["changed_first_actions"]]
    gamma = max(active, key=lambda r: (r["oracle_utility"], -r["gamma"]))["gamma"] if active else gammas[-1]
    policy |= {"gamma": gamma, "exploration_selection": "validation-only one-step oracle utility among weights that changed at least one real first action",
               "unchanged_on_validation": not bool(active),
               "checkpoints": {p.name: digest(p) for p in sorted((output/"models").glob("*.pt"))},
               "partition_sha256": digest(output/"frozen_partitions.json")}
    write_json(output/"validation_policy.json", policy)
    write_json(output/"model_registry.json", {"current_formal_model": "e4_a", "status": "research_candidate",
                                              "candidate": variant, "weights": policy["checkpoints"],
                                              "baseline_registry": str(paths[0]),
                                              "validation_policy_sha256": digest(output/"validation_policy.json")})
    print(f"Frozen policy: {variant}, {p0['selected_horizon_ms']} ms, scale={scale}, gamma={gamma}", flush=True)
    return policy


def read_training(output):
    import csv

    with (output/"training_summary.csv").open(encoding="utf-8", newline="") as stream:
        return [dict(row) | {k: float(v) for k, v in row.items() if k.startswith("val_")}
                for row in csv.DictReader(stream)]


def validate_policy(output, policy):
    if policy["partition_sha256"] != digest(output/"frozen_partitions.json"):
        raise ValueError("Frozen test partition changed")
    for name, expected in policy["checkpoints"].items():
        if expected != digest(output/"models"/name):
            raise ValueError(f"Checkpoint changed since validation: {name}")


def exploration_sensitivity(c, d, partitions, policy, trained, output, smoke):
    """Compare the fixed and individual dynamic factors on validation data."""
    from .active_control import PredictiveController

    if (output/"exploration_weight_sensitivity.csv").exists():
        return
    options = options_for(c, d, policy["horizon"], policy["observation_scale"])
    schemes = ("fixed", "entropy", "speed", "disagreement", "dynamic")
    changes = {name: [] for name in schemes}
    for seed in c["seeds"]:
        model = trained[policy["variant"], seed]
        for belief, history, actions, actual, forecast in forecasts_on_validation(
                c, d, partitions, model, policy, 12 if smoke else 64):
            score = utilities(actual, actions, history[-1, 3], options)[0]
            # Reuse the forecast shared by all weight rules.
            belief.predict = lambda *_: forecast
            for weighting in schemes:
                controller = PredictiveController(c, options, belief, information="decision",
                                                  gamma=policy["gamma"], weighting=weighting)
                _, decision = controller.evaluate(history[-1, :2], actions, history[-1, 3])
                changed = decision.information_changed
                changes[weighting].append((int(changed), decision.gamma, score[decision.candidate],
                                           score[decision.candidate]-score[decision.passive]))
    rows = []
    for label, values in changes.items():
        metrics = np.array(values)
        rows.append({"weighting": label, "windows": len(values), "changed_first_actions": int(metrics[:, 0].sum()),
                     "mean_weight": float(metrics[:, 1].mean()), "mean_oracle_utility": float(metrics[:, 2].mean()),
                     "mean_utility_delta_vs_passive": float(metrics[:, 3].mean())})
    write_csv(output/"exploration_weight_sensitivity.csv", rows)


def runner_specs(c, d, partitions, policy, smoke):
    roads = road_selection(partitions, d)
    initial = d["initial_conditions"]
    methods = (*CONTROLLERS, "structured_mpc", *ABLATIONS)
    ablation_roads = {r["id"] for family in d["formal_closed_roads"]
                      for r in partitions[f"test_{family}"][:2]}
    if smoke:
        roads = [roads[0]]
        initial = initial[:1]
    for method, road, condition in itertools.product(methods, roads, initial):
        if method in ABLATIONS and road["id"] not in ablation_roads:
            continue
        seeds = (0,) if method in NO_SEED else c["seeds"][:1] if smoke else c["seeds"]
        for seed in seeds:
            yield road, method, seed, condition, policy["horizon"]


def run_test_closed(c, d, paths, partitions, policy, args):
    phase = "closed_loop"
    specs = list(runner_specs(c, d, partitions, policy, args.smoke))
    examples = {partitions[f"test_{family}"][0]["id"] for family in d["formal_closed_roads"]}
    traces = {(road["id"], method, seed, condition[0]) for road, method, seed, condition, _ in specs
              if road["id"] in examples and seed in {0, c["seeds"][0]}
              and method in {"C2", "C3", "C4", "C6", "C8", "C9", "C10", "C11", "C12", "C13", "generic_eig"}}
    rows = run_closed(paths, args.output, c, d, specs, phase, args.workers, policy=policy, trace_ids=traces)
    write_csv(args.output/"closed_loop_summary.csv", control_summary(rows))
    write_csv(args.output/"generalization.csv", control_summary(rows, ("method", "family")))
    contrasts = paired_control(rows, d["bootstrap_samples"], d["partition_seed"], CONTRASTS)
    write_csv(args.output/"paired_statistics.csv", contrasts)
    write_csv(args.output/"control_ablation.csv", [r for r in contrasts if r["metric"] != "stopped"])
    return rows


def run_evaluation(c, d, paths, partitions, original, p0, args):
    output = args.output
    protocol = read_json(output/"training_protocol.json")
    stage = args.stage
    if bool(protocol["smoke"]) != args.smoke:
        raise ValueError("Smoke run does not match training protocol")
    trained = load_trained(output, c, protocol["study"])
    policy_path = output/"validation_policy.json"
    if stage == "validation" or (stage == "all" and not policy_path.exists()):
        policy = select_policy(c, d, paths, partitions, p0, args, trained, protocol)
    else:
        policy = read_json(policy_path)
    validate_policy(output, policy)
    if stage in {"closed", "test", "all"}:
        rows = run_test_closed(c, d, paths, partitions, policy, args)
    if stage in {"open", "test", "all"}:
        _, legacy, _ = load_baselines(paths)
        from .active_open_loop import run_open_test
        from .active_report import deliver_results, historical_pair_figure

        if stage == "open":
            rows = [read_json(p)["metrics"] for p in sorted((output/"episodes"/"closed_loop").glob("*.json"))]
            if len(rows) != len(list(runner_specs(c, d, partitions, policy, args.smoke))):
                raise ValueError("Run the frozen closed-loop stage before open-loop reporting")
        models = legacy | trained
        prediction = run_open_test(c, d, partitions, original, policy, args, models)
        deliver_results(c, d, policy, args, rows, prediction, models, partitions)
        if not args.smoke:
            historical_pair_figure(c, policy, output, models, original)
    if stage in {"audit", "all", "test"}:
        from .active_audit import audit

        audit(c, d, args, paths, partitions, policy, trained)
        exploration_sensitivity(c, d, partitions, policy, trained, output, args.smoke)
