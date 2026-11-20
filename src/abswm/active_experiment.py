"""Protocol, model adapters and paired simulation workers."""
from __future__ import annotations

import hashlib
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .active_control import PI, PeakSeeking, PredictiveController, simulate
from .active_data import frozen_partitions
from .active_rollout import NLSPredictor, NeuralCurvePredictor, OraclePredictor, RecurrentPredictor
from .config import fingerprint, write_json
from .decoder_study import read_json
from .evaluation import write_csv
from .recurrent_model import RecurrentTireModel, registered_reference
from .tire_prior import StructuredTire


P0 = ("C0", "C1", "C2", "C4", "C5", "C6", "C7", "structured_mpc")
CONTROLLERS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment(threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def inputs(registry_path, recurrent, prior):
    registry = read_json(registry_path)
    base, reference = Path(registry["base_run"]), Path(registry["reference_run"])
    original = read_json(base/"data"/"manifest.json")["roads"]
    historical = read_json(prior/"frozen_partitions.json")
    baseline = [reference/"models"/f"e4_a_{seed}.pt" for seed in registry["seeds"]]
    baseline += [recurrent/"models"/f"recurrent_transition_{seed}.pt" for seed in registry["seeds"]]
    for variant in ("history_joint", "structured_mixed"):
        baseline += [prior/"models"/f"{variant}_{seed}.pt" for seed in registry["seeds"]]
    dependencies = [base/"data"/"manifest.json", base/"controller.json", registry_path, *baseline]
    return base, reference, original, historical, {str(p): digest(p) for p in dependencies}


def load_baselines(paths):
    registry, base, reference, recurrent, prior = paths
    c, refs = registered_reference(registry, base, reference)
    recurrent_info = read_json(recurrent/"experiment.json")
    models = {}
    for seed, ref in refs.items():
        rec = RecurrentTireModel(ref, recurrent_info["options"], imagine=True)
        path = recurrent/"models"/f"recurrent_transition_{seed}.pt"
        payload = torch.load(path, weights_only=True)
        if payload["experiment_key"] != recurrent_info["experiment_key"]:
            raise ValueError(f"Invalid recurrent weight provenance for {path}")
        rec.load_state_dict(payload["state_dict"])
        models["recurrent", seed] = rec.eval().requires_grad_(False)
        models["e4", seed] = ref
        for label, file in (("structured", "history_joint"), ("structured_mixed", "structured_mixed")):
            structured = StructuredTire(ref)
            payload = torch.load(prior/"models"/f"{file}_{seed}.pt", weights_only=True)
            if payload["key"] != read_json(prior/"experiment.json")["experiment_key"]:
                raise ValueError(f"Invalid structured checkpoint provenance: {file}_{seed}")
            structured.load_state_dict(payload["state"])
            models[label, seed] = structured.eval().requires_grad_(False)
    pi = read_json(base/"controller.json")["selected"]
    return c, models, (pi["kp"], pi["ki"])


def controller_for(method, seed, road, c, options, models, gains, policy=None):
    if method == "C0":
        return PI(c, gains)
    if method in {"C1", "C5"}:
        predictor = NLSPredictor(c)
    elif method in {"C2", "C6", "C8"}:
        predictor = NeuralCurvePredictor(c, [models["e4", seed]])
    elif method == "structured_mpc":
        predictor = NeuralCurvePredictor(c, [models["structured", seed]])
    elif method == "C4":
        predictor = OraclePredictor(c, road)
    elif method == "C7":
        predictor = RecurrentPredictor(c, models["recurrent", seed])
    elif method == "C3":
        return PeakSeeking(c)
    else:
        return probabilistic_controller(method, seed, c, options, models, policy)
    if method in {"C1", "C2"}:
        return PI(c, gains, predictor)
    predictor = accelerate(predictor, road)
    return PredictiveController(c, options, predictor, assistant=method == "C8")


def accelerate(predictor, road=None):
    from .active_fast import njit, oracle_forecast, static_forecast

    if njit is not None:
        if isinstance(predictor, OraclePredictor):
            # The compiled kernel receives the same family parameters as this Oracle.
            predictor.predict = lambda current, actions: oracle_forecast(predictor, current, actions, road)
        elif not isinstance(predictor, RecurrentPredictor):
            predictor.predict = lambda current, actions: static_forecast(current, actions, predictor.curves, predictor.c)
    return predictor


def probabilistic_controller(method, seed, c, options, models, policy):
    from .active_belief import make_observer

    chosen, gamma = policy["variant"], policy["gamma"]
    active, assistant, risk = "", method in {"C9", "C10"}, method == "C12"
    weighting = "dynamic"
    memory, update, refinement = True, True, "none"
    if method in {"C10", "C13", "library_active", "fixed_evoi"}:
        active = "decision"
    if method == "generic_eig":
        active = "entropy"
    if method == "fixed_evoi":
        weighting = "fixed"
    if method == "no_bayes":
        update = False
    if method == "no_memory":
        memory = False
    if method == "information_refinement":
        refinement = "selected"
    model = None if method in {"library_active", "library_passive"} else models[chosen, seed]
    observer = make_observer(c, model, policy, update=update, memory=memory, refinement=refinement)
    return PredictiveController(c, options, observer, assistant=assistant, risk=risk, information=active,
                                gamma=gamma, weighting=weighting)


def init_closed_worker(paths, output, c, policy):
    global WORKER
    environment(c["torch_threads"])
    _, models, gains = load_baselines(paths)
    if policy is not None:
        from .active_training import load_trained
        models.update(load_trained(output, c, policy["study"],
                                   variants={policy["variant"]}, expected_key=policy["training_key"]))
    WORKER = (models, gains)


def closed_job(job):
    models, gains = WORKER
    road, method, seed, initial, c, options, policy, trace_requested, path, key = job
    if path.exists():
        saved = read_json(path)
        if saved["key"] == key:
            return saved["metrics"], saved["timing"], saved["trace"]
    ctl = controller_for(method, seed, road, c, options, models, gains, policy)
    result, trace, lat = simulate(road, c, ctl, initial[0], initial_slip=initial[1],
                                  initial_torque=initial[2], keep_trace=trace_requested)
    provenance = {"method": method, "seed": seed, "road_id": road["id"], "family": road["family"],
                  "surface": road["subfamily"], "initial_speed": initial[0], "initial_slip": initial[1],
                  "initial_torque": initial[2], "horizon_ms": round(options["horizon"]*c["control_dt"]*1000)}
    metrics = provenance | result
    timing = [provenance | item for item in lat]
    for row in trace:
        for name, value in row.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                row[name] = None
            elif isinstance(value, (np.bool_, np.floating, np.integer)):
                row[name] = value.item()
    temporary = path.with_suffix(".tmp")
    write_json(temporary, {"key": key, "metrics": metrics, "timing": timing, "trace": trace})
    temporary.replace(path)
    return metrics, timing, trace


def run_closed(paths, output, c, d, specs, phase, workers, *, policy=None, trace_ids=()):
    folder = output/"episodes"/phase
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    from .active_fast import njit

    key = fingerprint({"version": 1, "d": d, "c": c, "policy": policy,
                       "backend": "numba" if njit is not None else "numpy",
                       "inputs": digest(output/"experiment.json"),
                       "source": {name: digest(Path(__file__).with_name(name)) for name in
                                  ("active_control.py", "active_rollout.py", "active_information.py",
                                   "active_fast.py", "active_experiment.py",
                                   *(() if policy is None else ("active_belief.py",)))}})
    options = d["options"] | {"slew_nm": c["control_dt"]*c["torque_rate"]}
    if policy is not None:
        options |= {"observation_scale": policy["observation_scale"]/np.sqrt(d["likelihood_reliability"])}
    for road, method, seed, initial, horizon in specs:
        trace = (road["id"], method, seed, initial[0]) in trace_ids
        filename = f"{road['id']}_{method}_{seed}_{initial[0]}_{horizon}.json"
        jobs.append((road, method, seed, initial, c, options | {"horizon": horizon}, policy,
                     trace, folder/filename, fingerprint([key, road, method, seed, initial, horizon, trace])))
    rows, timing, traces = [], [], {}
    with ProcessPoolExecutor(max_workers=workers, initializer=init_closed_worker,
                             initargs=(paths, output, c, policy)) as pool:
        for row, latency, trace in pool.map(closed_job, jobs, chunksize=1):
            rows.append(row)
            timing.extend(latency)
            if trace:
                filename = f"{row['road_id']}_{row['method']}_{row['seed']}_{row['initial_speed']}.csv"
                traces[filename] = trace
            if len(rows) % 12 == 0 or len(rows) == len(jobs):
                print(f"{phase}: {len(rows)}/{len(jobs)}", flush=True)
    write_csv(output/f"{phase}_by_road.csv", rows)
    if timing:
        write_csv(output/f"{phase}_timing.csv", timing)
    for name, trace in traces.items():
        write_csv(output/"traces"/name, trace)
    return rows


def road_selection(partitions, d):
    result = []
    for family, count in d["formal_closed_roads"].items():
        result.extend(partitions[f"test_{family}"][:count])
    return result


def p0_validate(paths, output, c, d, partitions, workers, *, smoke=False):
    roads = partitions["validation_burckhardt"][:d["validation_closed_roads"]]
    if smoke:
        roads = roads[:1]
    first = c["seeds"][0]
    seed = {"C2": first, "C6": first, "C7": first, "structured_mpc": first}
    horizons = [round(ms/(1000*c["control_dt"])) for ms in d["horizon_ms"]]
    initial = d["initial_conditions"][0]
    specs = [(r, method, seed.get(method, 0), initial, h)
             for r, method, h in itertools.product(roads, ("C4", "C6"), horizons)]
    specs += [(r, method, seed.get(method, 0), initial, horizons[0])
              for r, method in itertools.product(roads, [m for m in P0 if m not in {"C4", "C6"}])]
    rows = run_closed(paths, output, c, d, specs, "p0_validation", workers)
    by_horizon = {}
    for ms in d["horizon_ms"]:
        scores = [row for row in rows if row["method"] == "C4" and row["horizon_ms"] == ms]
        by_horizon[str(ms)] = float(np.mean([r["distance_m"]+30*r["high_slip_s"]+50*r["lock_s"]+
                                           1000*(not r["stopped"]) for r in scores]))
    chosen = int(min(by_horizon, key=by_horizon.get))
    oracle = [r for r in rows if r["method"] == "C4" and r["horizon_ms"] == chosen]
    fixed = [r for r in rows if r["method"] == "C0"]
    valid = (all(r["stopped"] for r in oracle) and np.mean([r["lock_s"] for r in oracle]) <= .005
             and np.mean([r["distance_m"] for r in oracle]) <= 1.05*np.mean([r["distance_m"] for r in fixed]))
    result = {"selected_horizon_ms": chosen, "oracle_validation_cost": by_horizon,
              "oracle_sanity_passed": bool(valid), "oracle_gate": "all completed, mean lock <= 5 ms, mean distance <= 105% Fixed PI"}
    write_json(output/"p0_policy.json", result)
    print(f"P0 validation: {result}", flush=True)
    return result


def prepare_experiment(output, paths, config_path):
    registry, base, reference, recurrent, prior = paths
    _, _, original, historical, fingerprints = inputs(registry, recurrent, prior)
    d = read_json(config_path)
    c = read_json(base/"config.json")
    partitions = frozen_partitions(c, d, original, historical)
    output.mkdir(parents=True, exist_ok=True)
    partition_path = output/"frozen_partitions.json"
    if partition_path.exists() and read_json(partition_path) != partitions:
        raise ValueError("Frozen road partitions differ. Use a new output directory.")
    write_json(partition_path, partitions)
    experiment = {"configuration": d, "base": c, "frozen_partition_sha256": digest(partition_path),
                  "reused_inputs_sha256": fingerprints, "threads_per_worker": c["torch_threads"]}
    manifest = output/"experiment.json"
    if manifest.exists() and read_json(manifest) != experiment:
        raise ValueError("Experiment configuration changed. Use a new output directory.")
    write_json(manifest, experiment)
    return c, d, partitions, original
