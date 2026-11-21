"""S3 with fixed windows and updates, and validation-only S4 registration."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import csv

import numpy as np

from .active_experiment import digest, environment
from .adaptive_experiment import seal, table
from .config import fingerprint
from .data import generate_split, load_split
from .decoder_study import read_json
from .scaling_data import atomic_torch, dataset, describe_windows
from .scaling_evaluation import paired_summary, summary
from .scaling_training import batches, train_budget
from .training_scaling import validation_selection


def read_table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def freeze(path, payload):
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"Incompatible frozen record: {path}")
    else:
        seal(path, payload)


def assign_tiers(locators, drives_per_road, multiplier, seed):
    """Balanced, nested road allocation for a fixed list of window slots."""
    rng = np.random.default_rng(seed)
    base = locators[:, 0]//drives_per_road
    tier = np.empty(len(base), dtype=np.int32)
    for rid in np.unique(base):
        slots = rng.permutation(np.flatnonzero(base == rid))
        if len(slots) < multiplier:
            raise ValueError("Insufficient windows to cover all assigned road tiers")
        tier[slots] = np.arange(len(slots)) % multiplier
    return tier


def replace_roads(locators, tiers, multiplier, original_roads, drives_per_road, lengths, c, seed):
    """Retain original acquisition times when possible without repeating new windows."""
    rng = np.random.default_rng(seed)
    out = locators.copy()
    out[:, 0] += (tiers % multiplier)*original_roads*drives_per_road
    seen = set()
    capped, collision = 0, 0
    horizon = c["horizon_steps"]
    for i, (drive, t) in enumerate(out):
        if drive == locators[i, 0]:
            continue
        drive, t = int(drive), int(t)
        last = int(lengths[drive])-horizon-1
        if last < 1:
            raise ValueError("Replacement drive has no eligible window")
        if t > last or (drive, t) in seen:
            early = min(last, c["history_steps"])
            lo, hi = (0, early) if t <= early else (early+1, last)
            candidates = [k for k in range(lo, hi+1) if (drive, k) not in seen]
            if not candidates:
                candidates = [k for k in range(last+1) if (drive, k) not in seen]
            if not candidates:
                raise ValueError("Not enough distinct replacement windows")
            capped += t > last
            collision += (drive, t) in seen
            t = int(rng.choice(candidates))
            out[i, 1] = t
        seen.add((drive, t))
    return out, {"redrawn_past_end": capped, "redrawn_collisions": collision}


def road_cache(study, folder, train_roads, reserve, base_points, multipliers):
    c = study.c
    paths = {f"C{i}": folder/"cache"/f"C{i}.pt" for i in range(1, len(multipliers))}
    if all(p.exists() and p.with_suffix(".json").exists() and p.with_suffix(".npy").exists() for p in paths.values()):
        return paths
    base = load_split(study.base/"data", "train")
    n = len(train_roads)
    routes = len(c["initial_speeds"])*len(c["excitation_kinds"])
    all_roads = train_roads+reserve
    if len(all_roads) != n*max(multipliers) or any(r["family"] != train_roads[i % n]["family"]
                                                 for i, r in enumerate(all_roads)):
        raise ValueError("The frozen reserve does not match the registered road mixture")
    print(f"S3 building archived + reserve trajectories for {len(all_roads)} roads", flush=True)
    extra = generate_split(reserve, c | {"data_seed": study.d["partition_seed"]+377}, 0)
    extra["road_index"] += n
    pool = {k: np.concatenate((base[k], extra[k]), axis=0) for k in base}
    del extra, base
    tiers = assign_tiers(base_points, routes, max(multipliers), study.d["data_sampling_seed"]+107)
    for i, mul in enumerate(multipliers[1:], start=1):
        path = paths[f"C{i}"]
        points, changes = replace_roads(base_points, tiers, mul, n, routes, pool["lengths"], c,
                                       study.d["data_sampling_seed"]+107)
        roads = all_roads[:mul*n]
        tensors = dataset(c, study.study, pool, roads, points)
        detail = describe_windows(points, tensors["history"].numpy(), pool, roads, len(points))
        if detail["sampled_roads"] != len(roads):
            raise ValueError("An assigned road is absent from the fixed window budget")
        unique, counts = np.unique(pool["road_index"][points[:, 0]], return_counts=True)
        detail |= changes | {"min_windows_per_road": int(counts.min()), "median_windows_per_road": float(np.median(counts)),
                             "max_windows_per_road": int(counts.max()), "road_ids": [roads[i]["id"] for i in unique]}
        atomic_torch(path, tensors)
        np.save(path.with_suffix(".npy"), points, allow_pickle=False)
        seal(path.with_suffix(".json"), detail)
        print(f"S3 C{i}: {len(roads)} roads, {len(points)} windows", flush=True)
    return paths


def run_s3(study, folder):
    mark = perf_counter()
    root = study.output
    previous = read_json(root/"stage_status.json")
    if previous["completed"] not in {"S2 B3", "S2 B4"}:
        raise ValueError("A completed S2 run is required")
    prior = read_table(root/"training_summary.csv")
    choices = [r for r in read_table(root/"window_scaling.csv") if r["seed"] == "pooled" and r["guardrails"] == "True"]
    winner = min(choices, key=lambda r: (float(r["score"]), int(r["train_windows"]), r["method"]))
    n = int(winner["train_windows"])
    checkpoints = {label: {int(r["seed"]): r["checkpoint"] for r in prior if r["method"] == method}
                   for label, method in (("A0", "A0"), ("S1", previous["s1_best"]), ("S2", winner["method"]))}
    updates = {int(r["updates"]) for r in prior if r["method"] == winner["method"]}
    if len(updates) != 1 or not all(set(study.models) == set(v) for v in checkpoints.values()):
        raise ValueError("Inconsistent seed checkpoints or optimizer budget")
    budget = updates.pop()
    full = read_json(root/"frozen_partitions.json")
    multipliers = study.d["road_multipliers"]
    reserved = full["s3_training_reserve"]
    if study.smoke:
        # Use a complete surface block for a small matched-budget exercise.
        routes = len(study.c["initial_speeds"])*len(study.c["excitation_kinds"])
        limit = 3
        old = full["train"][:limit]
        reserve = reserved[:limit]
        base = load_split(study.base/"data", "train")
        count = limit*routes
        smoke_reference = {k: v[:count] for k, v in base.items()}
        smoke_points = np.array([(i, t) for i in range(count) for t in range(1, 9)], dtype=np.int32)
        paths = _smoke_road(study, folder, old, reserve, smoke_reference, smoke_points)
        budget = 1
        multipliers = [1, 2]
        window_count = len(smoke_points)
    else:
        old, reserve = full["train"], reserved
        window_count = n
        checkpoints["C0"] = checkpoints["S2"]
    policy = {"source_experiment": read_json(root/"experiment.json")["experiment_key"],
              "frozen_partitions_sha256": digest(root/"frozen_partitions.json"),
              "s2_selected": winner["method"], "windows": window_count, "updates": budget,
              "epochs": budget/batches(window_count, study.c), "multipliers": multipliers, "smoke": study.smoke,
              "batch_size": study.c["batch_size"], "selection": study.d["selection"],
              "starts_from_registered_checkpoint": True,
              "donor_and_reference_hashes": {name: {str(s): digest(p) for s, p in weight.items()}
                                      for name, weight in checkpoints.items()},
              "code_sha256": digest(Path(__file__))}
    freeze(folder/"s3_protocol.json", policy)
    done_path = folder/"s3_status.json"
    if done_path.exists():
        return read_json(done_path)
    if not study.smoke:
        pts = np.load(root/"cache"/f"windows_{n//study.n0}.npy", allow_pickle=False)
        paths = road_cache(study, folder, old, reserve, pts, multipliers)
    study.prepare_validation()
    for label in ("A0", "C0"):
        for seed, p in checkpoints.get(label, {}).items():
            study.checkpoint_summary(label, seed, Path(p))
    futures = {}
    with ProcessPoolExecutor(max_workers=study.workers, initializer=environment,
                             initargs=(study.c["torch_threads"],)) as pool:
        for label, path in paths.items():
            checkpoints[label] = {}
            for seed, model in study.models.items():
                job = {"path": folder/"models"/f"{label}_{seed}", "arm": label, "seed": seed,
                       "windows": window_count, "updates": [budget], "model": model,
                       "c": study.c, "study": study.study, "train": path, "val": study.val_path}
                futures[pool.submit(train_budget, job)] = (label, seed)
        for future in as_completed(futures):
            label, seed = futures[future]
            path = future.result()[budget]
            checkpoints[label][seed] = path
            study.checkpoint_summary(label, seed, Path(path))
            print(f"S3 {label} seed={seed} evaluation exported to cache", flush=True)
    reference = summary(study.rows["A0"])
    results, comparisons, train_rows, learning = [], [], [], []
    for i, mul in enumerate(multipliers):
        label = f"C{i}"
        rows = study.rows[label]
        score, safe = validation_selection(summary(rows), reference, study.d)
        details = {"method": label, "roads": len(old)*mul, "score": score, "guardrails": safe}
        results.append(details | {"seed": "pooled"} | summary(rows))
        for seed in study.models:
            record = details | {"seed": seed, "checkpoint": study.checkpoints[label, seed]}
            state = study.training[label, seed]
            train_rows.append(record | {k: v for k, v in state.items() if k != "history"})
            learning.extend(record | r for r in state["history"])
            results.append(record | summary([r for r in rows if r["seed"] == seed]))
        for control in ("A0", "C0") if i else ("A0",):
            comparisons.extend({"method": label, "reference": control} | r for r in
                               paired_summary(rows, study.rows[control], study.d["bootstrap_samples"],
                                              study.d["partition_seed"]))
    table(folder, "road_scaling", results)
    table(folder, "s3_training_summary", train_rows)
    table(folder, "s3_learning_curves", learning)
    table(folder, "s3_paired_statistics", comparisons)
    table(folder, "s3_action_prediction", [r for k, rows in study.rows.items() if k.startswith("C") for r in rows])
    best = min((r for r in results if r["seed"] == "pooled" and r["guardrails"]),
               key=lambda r: (r["score"], r["roads"]))["method"]
    record = {"completed": "S3", "s2_best": winner["method"], "s3_best": best,
              "elapsed_s": perf_counter()-mark, "policy_sha256": digest(folder/"s3_protocol.json"),
              "offer_s4": True, "checkpoint_paths": {k: {str(s): p for s, p in files.items()}
                                                     for k, files in checkpoints.items() if k in {"A0", "S1", "S2", best}}}
    seal(done_path, record)
    print(f"S3 completed: {best}; final test awaiting registration", flush=True)
    return record


def _smoke_road(study, folder, old, reserve, base, points):
    c = study.c
    new = generate_split(reserve, c | {"data_seed": study.d["partition_seed"]+377}, 0)
    new["road_index"] += len(old)
    combined = {k: np.concatenate([base[k], new[k]]) for k in base}
    routes = len(c["initial_speeds"])*len(c["excitation_kinds"])
    tiers = assign_tiers(points, routes, 2, 123)
    changed, _ = replace_roads(points, tiers, 2, len(old), routes, combined["lengths"], c, 123)
    paths = {"C0": folder/"cache"/"smoke_C0.pt", "C1": folder/"cache"/"smoke_C1.pt"}
    atomic_torch(paths["C0"], dataset(c, study.study, base, old, points))
    atomic_torch(paths["C1"], dataset(c, study.study, combined, old+reserve, changed))
    return paths
