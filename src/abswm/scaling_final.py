"""Evaluate the validation-registered E4 A checkpoints on sealed final scenarios."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .active_control import PI, PredictiveController
from .active_experiment import accelerate, digest, environment
from .active_metrics import control_summary, grouped, paired_control
from .active_rollout import Forecast, NLSPredictor, NeuralCurvePredictor, candidate_actions
from .adaptive_analysis import open_samples
from .adaptive_experiment import seal, table
from .adaptive_simulation import SpatialOracle, drive
from .config import fingerprint
from .decoder_study import read_json
from .scaling_data import _road_ids
from .scaling_evaluation import evaluate, paired_summary, summary
from .scaling_followup import freeze
from .spatial_road import SpatialRoad, make_partitions
from .training_scaling import REPO


def distinct_models(paths):
    """Deduplicate an entire three-seed checkpoint set, keeping every stage alias."""
    groups, keys, aliases = {}, {}, {}
    for method, files in paths.items():
        entries = {seed: {"path": str(path), "sha256": digest(path)} for seed, path in files.items()}
        identity = tuple((seed, spec["sha256"]) for seed, spec in sorted(entries.items()))
        if identity not in keys:
            keys[identity] = method
            groups[method] = entries
        aliases[method] = keys[identity]
    return groups, aliases


def _init_worker(c, models, policy, folder):
    global WORKER
    environment(c["torch_threads"])
    loaded = {}
    for name, weights in policy["models"].items():
        for seed, spec in weights.items():
            seed = int(seed)
            if digest(spec["path"]) != spec["sha256"]:
                raise ValueError("A final checkpoint changed after registration")
            model = deepcopy(models[seed])
            model.load_state_dict(torch.load(spec["path"], weights_only=True)["state_dict"])
            loaded[name, seed] = model.eval()
    WORKER = c, loaded, policy, folder


def _closed_job(job):
    c, models, policy, folder = WORKER
    scene, initial, method, seed = job
    identity = fingerprint([scene["id"], initial, method, seed, policy["key"]])
    path = folder/"episodes"/f"{identity[:24]}.json"
    if path.exists():
        return read_json(path)
    road = SpatialRoad.build(scene, c)
    if method == "history_PI":
        controller = PI(c, tuple(policy["history_pi_gains"]))
    else:
        if method == "Oracle-MPC":
            predictor = SpatialOracle(road, preview=False)
        elif method == "NLS-MPC":
            predictor = accelerate(NLSPredictor(c))
        else:
            predictor = accelerate(NeuralCurvePredictor(c, [models[method, seed]]))
        controller = PredictiveController(c, policy["mpc"], predictor)
    episode = drive(road, controller, initial, policy["mpc"], keep=method == "history_PI")
    episode["metrics"] |= {"method": method, "seed": seed, "road_id": scene["id"], "family": scene["family"],
                           "kind": scene["kind"], "surface_type": "constant" if scene["kind"] == "constant" else "dynamic",
                           "initial_speed": initial[0], "initial_slip": initial[1], "initial_torque": initial[2]}
    seal(path, episode)
    return episode


def _samples(c, policy, scenes, histories):
    records = []
    for spatial, history, _, _, position, key in open_samples(c, {"seed": policy["history_seed"]}, scenes,
                                                             dynamic_episodes=histories, full=True):
        # Curve diagnostics use the dominant current segment during a blend. Rollout truth uses spatial RK4.
        active = int(np.searchsorted(spatial.starts, position, side="right"))
        if active and position-spatial.starts[active-1] < .5*spatial.widths[active-1]:
            active -= 1
        road = spatial.spec["segments"][active]
        records.append((spatial, history, position, road, key))
    return records


def _open_labels(records, c, horizons):
    labels = {}
    for n in horizons:
        actions = candidate_actions([r[1][-1, 3] for r in records], n, c)
        forecasts = [spatial.forecast([*h[-1, :2], position], proposals, stop=0.)[0]
                     for (spatial, h, position, _, _), proposals in zip(records, actions)]
        fields = [np.concatenate([getattr(f, k) for f in forecasts]) for k in
                  ("state", "slip", "mu", "interval_mu", "max_slip", "min_wheel")]
        labels[n] = actions, Forecast(*fields, forecasts[0].dt, forecasts[0].load)
    return labels


def _open_results(study, folder, policy, scenes, episodes):
    c = study.c
    records = _samples(c, policy, scenes, episodes)
    labels = _open_labels(records, c, sorted({policy["mpc"]["horizon"], c["horizon_steps"]}))
    histories = np.stack([r[1] for r in records])
    roads = [r[3] for r in records]
    rows, timings = [], []
    for name, weights in policy["models"].items():
        for seed, spec in weights.items():
            seed = int(seed)
            target = folder/"open_cache"/f"{policy['key'][:16]}_{name}_{seed}.json"
            if target.exists():
                out = read_json(target)
            else:
                model = deepcopy(study.models[seed])
                model.load_state_dict(torch.load(spec["path"], weights_only=True)["state_dict"])
                result, timing = evaluate(model, histories, roads, labels, c, policy["mpc"], policy["tie_margin"],
                                          method=name, seed=seed, timing=True)
                for row, (spatial, _, position, _, key) in zip(result, records):
                    kind = spatial.spec["kind"]
                    row |= {"case": key, "road_id": spatial.spec["id"], "position_m": position, "kind": kind,
                            "surface_type": "constant" if kind == "constant" else "dynamic"}
                out = {"rows": result, "timing": timing}
                seal(target, out)
            rows.extend(out["rows"])
            timings.extend(out["timing"])
    return rows, timings


def _slices(rows, outcome):
    axes = [("surface_type",), ("surface_type", "family"), ("kind",)]
    if outcome == "open":
        axes += [("surface_type", "regime")]
    for features in [(), *axes]:
        for labels, group in grouped(rows, features).items():
            yield dict(zip(features, labels)), group


def _comparison(open_rows, closed, study, policy):
    d = study.d
    learned = list(policy["models"])
    contrasts = [(name, "A0") for name in learned if name != "A0"]
    comparisons = contrasts+[(name, ref) for name in learned for ref in ("NLS-MPC", "Oracle-MPC")]
    paired = []
    for labels, group in _slices(open_rows, "open"):
        for name, ref in contrasts:
            left = [r for r in group if r["method"] == name]
            right = [r for r in group if r["method"] == ref]
            paired.extend(labels | {"domain": "open", "method": name, "baseline": ref} | row for row in
                          paired_summary(left, right, d["bootstrap_samples"], d["partition_seed"]))
    for labels, group in _slices(closed, "closed"):
        paired.extend(labels | {"domain": "closed"} | r for r in
                      paired_control(group, d["bootstrap_samples"], d["partition_seed"], comparisons))
    return paired


def _acceptance(open_rows, closed, policy):
    g = policy["gates"]
    by_open = {k: summary(v) for k, v in grouped(open_rows, ("method",)).items()}
    by_closed = {tuple(r[k] for k in ("method", "surface_type")): r for r in
                 control_summary(closed, keys=("method", "surface_type"))}
    a = by_open["A0",]
    result = {}
    for (method,), b in by_open.items():
        if method == "A0":
            continue
        metrics = {"action_rmse_gain": 1-b["action_wheel_rmse"]/a["action_wheel_rmse"],
                   "force_rmse_regression": b["force_rmse"]/a["force_rmse"]-1,
                   "action_regret_gain": 1-b["action_regret_ns"]/a["action_regret_ns"],
                   "ranking_drop": a["ranking_accuracy"]-b["ranking_accuracy"],
                   "violation_increase": b["violation_rate"]-a["violation_rate"],
                   "high_slip_fnr_increase": b["high_fnr"]-a["high_fnr"] if None not in (b["high_fnr"], a["high_fnr"]) else None}
        constant = [r for r in open_rows if r["method"] == method and r["surface_type"] == "constant"]
        family = {k[0]: summary(v)["high_fnr"] for k, v in grouped(constant, ("family",)).items()}
        metrics["worst_family_high_slip_fnr"] = max(family.values()) if family and None not in family.values() else None
        for kind in ("constant", "dynamic"):
            baseline, candidate = by_closed["A0", kind], by_closed[method, kind]
            complete = not baseline["failed"] and not candidate["failed"]
            ratio = candidate["distance_m"]/max(baseline["distance_m"], 1e-12)
            if kind == "constant":
                metrics["constant_distance_gain"] = 1-ratio if complete else None
            else:
                metrics["dynamic_distance_regression"] = ratio-1 if complete else None
            for key, field in (("high_slip_time", "high_slip_s"), ("lock_time", "lock_s"),
                               ("failure_rate", "failure_rate")):
                metrics[key+"_increase"] = max(metrics.get(key+"_increase", -float("inf")),
                                              candidate[field]-baseline[field])
        gates = {}
        for name, value in metrics.items():
            minimum = "min_"+name
            limit = g[minimum] if minimum in g else g["max_"+name]
            gates[name] = value is not None and (value >= limit if minimum in g else value <= limit)
        result[method] = {"metrics": metrics, "family_high_slip_fnr": family, "gates_pass": gates,
                          "eligible": not policy["smoke"] and all(gates.values())}
    return result


def run_s4(study, folder, selection):
    start = perf_counter()
    if selection["completed"] != "S3" or digest(folder/"s3_protocol.json") != selection["policy_sha256"]:
        raise ValueError("S4 requires the completed, matching S3 validation selection")
    root, c = study.output, study.c
    paths = selection["checkpoint_paths"]
    models, aliases = distinct_models({k: paths[k] for k in dict.fromkeys(["A0", "S1", "S2", selection["s3_best"]])})
    partitions = root/"frozen_partitions.json"
    previous = read_json(partitions)
    if study.smoke:
        extra = {"seed": study.d["partition_seed"]+413200,
                 "constant_roads": {"train": [0, 0, 0], "validation": [0, 0, 0], "test": [1, 1, 1]},
                 "dynamic_repeats": {"train": 0, "validation": 0, "test": 1}}
        final = make_partitions(c, extra, _road_ids(previous))["test"]
        final["dynamic"] = final["dynamic"][:1]
    init = read_json(REPO/"configs"/"adaptive_belief_feedback.json")["initial_conditions"]
    gains = read_json(study.base/"controller.json")["selected"]
    policy = {"models": models, "aliases": aliases, "mpc": study.options, "initial_conditions": init[:1] if study.smoke else init,
              "stop_speed": c["stop_speed"], "history_pi_gains": [gains["kp"], gains["ki"]],
              "oracle": "current spatial road only; no preview", "history_seed": study.d["partition_seed"]+8317,
              "partitions_sha256": digest(partitions), "s3_protocol_sha256": selection["policy_sha256"],
              "gates": study.d["upgrade_gates"], "tie_margin": study.d["tie_margin"], "smoke": study.smoke,
              "distance_policy": "Distance gates require a common 2 m/s endpoint in every paired case. Retain all case rows.",
              "paired_bootstrap": {"clusters": "road", "samples": study.d["bootstrap_samples"]},
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                ("scaling_final.py", "adaptive_simulation.py", "active_control.py",
                                 "spatial_road.py", "adaptive_analysis.py", "active_data.py")}}
    if study.smoke:
        policy["smoke_scenarios"] = final
    policy["key"] = fingerprint(policy)
    freeze(folder/"s4_policy.json", policy)
    if (folder/"s4_status.json").exists():
        return read_json(folder/"s4_status.json")
    if not study.smoke:
        final = previous["final_test"]
    scenes = final["constant"]+final["dynamic"]
    dynamic_ids = {sc["id"] for sc in final["dynamic"]}
    seeds = c["seeds"][:1] if study.smoke else c["seeds"]
    jobs = [(sc, init, name, seed) for sc, init, name in
            product(scenes, policy["initial_conditions"], [*models, "NLS-MPC", "Oracle-MPC", "history_PI"])
            if name != "history_PI" or sc["id"] in dynamic_ids for seed in (seeds if name in models else [0])]
    closed, histories, timings = [], [], []
    with ProcessPoolExecutor(max_workers=study.workers, initializer=_init_worker,
                             initargs=(c, study.models, policy, folder)) as pool:
        for i, episode in enumerate(pool.map(_closed_job, jobs), 1):
            meta = episode["metrics"]
            if meta["method"] == "history_PI":
                histories.append(episode)
            else:
                closed.append(meta)
                timings.extend({k: meta[k] for k in ("method", "seed", "road_id", "initial_speed")} | t
                               for t in episode["timing"])
            if i % 24 == 0 or i == len(jobs):
                print(f"S4 episodes {i}/{len(jobs)}", flush=True)
    rows, opened = _open_results(study, folder, policy, scenes, histories)
    control = [labels | item | {"safety_rate": float(np.mean(
               [r["safety_rate"] for r in block if r["method"] == item["method"]]))}
               for labels, block in _slices(closed, "closed") for item in control_summary(block)]
    general = [labels | {"method": name, "domain": "open"} | summary(block) for labels, group in _slices(rows, "open")
               for (name,), block in grouped(group, ("method",)).items()]
    table(folder, "action_prediction", rows)
    table(folder, "action_regret", general)
    table(folder, "risk_evaluation", general)
    table(folder, "closed_loop_by_case", closed)
    table(folder, "closed_loop_summary", control)
    table(folder, "generalization", [*general, *({"domain": "closed"} | r for r in control)])
    table(folder, "paired_statistics", _comparison(rows, closed, study, policy))
    table(folder, "timing", [*timings, *opened])
    acceptance = _acceptance(rows, closed, policy)
    seal(folder/"acceptance.json", {"policy_key": policy["key"], "status": "evaluated", "candidates": acceptance})
    status = {"completed": "S4", "policy_key": policy["key"], "selected": aliases,
              "elapsed_s": perf_counter()-start, "closed_cases": len(closed), "open_rows": len(rows),
              "smoke": study.smoke, "eligible": [k for k, v in acceptance.items() if v["eligible"]]}
    seal(folder/"s4_status.json", status)
    print(status, flush=True)
    return status
