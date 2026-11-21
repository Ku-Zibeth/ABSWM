"""Preserve the original E4 A windows and extend them without repeated locators."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import make_history, sample_roads
from .decoder_study import make_training_data, read_json
from .spatial_road import make_partitions


def nested_locators(data, c, archived, families, multiplier, seed):
    """Draw additional unique windows with the original per-window sampler.

    Within each added block, match the archived family counts exactly.
    The immutable prefix can contain sampling-with-replacement duplicates.
    """
    archived = np.asarray(archived, dtype=np.int32)
    indices = np.asarray(data["road_index"])
    family = np.asarray(families)[indices]
    counts = Counter(family[archived[:, 0]])
    horizon = c["horizon_steps"]
    eligible = {name: np.flatnonzero((family == name) & (data["lengths"] > horizon+1)) for name in counts}
    seen = {tuple(row) for row in archived.tolist()}
    rng = np.random.default_rng(seed)
    result = [archived]
    for _ in range(multiplier-1):
        block = []
        for name, budget in counts.items():
            options = eligible[name]
            remaining = int(np.sum(data["lengths"][options]-horizon))
            remaining -= sum(family[i] == name for i, _ in seen)
            if remaining < budget:
                raise ValueError(f"Insufficient unseen E4 A windows in {name}")
            while budget:
                i = int(rng.choice(options))
                last = int(data["lengths"][i])-horizon-1
                upper = min(last, c["history_steps"]) if rng.random() < .4 else last
                t = int(rng.integers(0, upper+1))
                if (i, t) not in seen:
                    seen.add((i, t))
                    block.append((i, t))
                    budget -= 1
        block = np.asarray(block, dtype=np.int32)
        result.append(block[rng.permutation(len(block))])
    return np.concatenate(result)


def extract(data, c, locators):
    h = c["horizon_steps"]
    i, t = np.asarray(locators).T
    states = data["states"]
    actions = data["actions"]
    history = np.stack([make_history(states[a], actions[a], int(b), c["history_steps"], c) for a, b in locators])
    return {"history": history, "states": states[i[:, None], t[:, None]+np.arange(h+1)],
            "actions": actions[i[:, None], t[:, None]+np.arange(h)],
            "mu": data["mu"][i, t], "peak": data["peak"][i]}


def describe_windows(locators, history, data, roads, size):
    t, states = locators[:size, 0], history[:size]
    ids = data["road_index"][t]
    family = [roads[int(i)]["family"] for i in ids]
    # Count identical byte histories separately from duplicate trajectory-time locators.
    flat = np.ascontiguousarray(states.reshape(size, -1))
    unique_history = np.unique(flat.view(np.dtype((np.void, flat.dtype.itemsize*flat.shape[1])))).size
    high = states[..., 2].max(axis=1)
    counts = np.histogram(high, bins=[0., .02, .05, .10, .20, .50, 1.000001])[0]
    return {"windows": size, "unique_windows": int(np.unique(locators[:size], axis=0).shape[0]),
            "distinct_observed_histories": int(unique_history), "distinct_trajectories": int(len(np.unique(t))),
            "sampled_roads": int(len(np.unique(ids))), "available_training_roads": len(roads),
            "family_counts": dict(Counter(family)), "slip_bin_counts": counts.tolist(),
            "slip_coverage_p10": float(np.quantile(high, .1)), "slip_coverage_p50": float(np.median(high)),
            "slip_coverage_p90": float(np.quantile(high, .9))}


def dataset(c, study, data, roads, locators):
    sample = extract(data, c, locators)
    indices = data["road_index"][locators[:, 0]]
    grid = np.linspace(0., 1., study["train_grid_points"])
    return make_training_data(c, sample, indices, roads, grid)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def _road_ids(obj):
    if isinstance(obj, dict):
        if "id" in obj:
            # Dynamic scenarios still need constituent ids for disjointness.
            return {obj["id"]}.union(*(_road_ids(s) for s in obj.get("segments", ())))
        return set().union(*(_road_ids(value) for value in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_road_ids(value) for value in obj))
    return set()


def freeze_partitions(c, d, original, historical):
    seen = _road_ids(original)
    for path in historical:
        if path.exists():
            seen.update(_road_ids(read_json(path)))
    test_profile = {"seed": d["partition_seed"], "constant_roads": {
        "train": [0, 0, 0], "validation": [0, 0, 0], "test": d["final_constant_roads"]},
        "dynamic_repeats": {"train": 0, "validation": 0, "test": d["final_dynamic_repeats"]}}
    final = make_partitions(c, test_profile, seen)["test"]
    seen.update(_road_ids(final))
    # An unused reserve only. The stage report does not fit S3 or open the test.
    total = len(original["train"])*max(d["road_multipliers"])
    profile = c | {"roads": {"reserve": total-len(original["train"])}, "data_seed": d["partition_seed"]+273}
    extra = sample_roads(profile)["reserve"]
    if seen & _road_ids(extra):
        raise ValueError("The future road-scaling reserve overlaps another split")
    return {"train": original["train"], "validation": original["val"], "final_test": final,
            "s3_training_reserve": extra, "historical_only_ids": sorted(_road_ids(original["test"]))}
