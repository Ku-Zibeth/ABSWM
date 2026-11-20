"""Road-disjoint counterfactual suites for ABSWM 3.0."""
from __future__ import annotations

import numpy as np

from .active_rollout import candidate_actions, true_forecast
from .data import make_history, sample_roads
from .tire_prior_data import burckhardt, generate_family, pack, stress_suite, trajectory


def frozen_partitions(c, config, original, history=None):
    seed = config["partition_seed"]
    counts = config["test_roads"]
    profile = {**c, "data_seed": seed, "roads": {"test": counts["burckhardt"]}}
    history = history or {}
    new = {
        "train_burckhardt": [burckhardt(r) for r in original["train"]],
        "validation_burckhardt": [burckhardt(r) for r in original["val"]],
        "train_magic": generate_family("magic", config["magic_train_roads"], seed+1),
        "validation_magic": generate_family("magic", config["magic_validation_roads"], seed+2),
        "test_burckhardt": [burckhardt(r) for r in sample_roads(profile)["test"]],
        "test_magic": generate_family("magic", counts["magic"], seed+3),
        "test_rational": generate_family("rational", counts["rational"], seed+4),
    }
    historical_ids = {r["id"] for r in original["test"]}
    historical_ids.update(r["id"] for roads in history.values() for r in roads)
    for name, roads in new.items():
        ids = {r["id"] for r in roads}
        if len(ids) != len(roads) or ids & historical_ids:
            raise ValueError(f"Road held out incorrectly in {name}")
        historical_ids.update(ids)
    return new


def history_windows(c, options, roads, count, seed, *, all_roads=False):
    """Generate paired causal prefixes with a variety of achieved slip ranges."""
    rng = np.random.default_rng(seed)
    drives = [road for road in roads for _ in range(2)]
    speed = np.tile(c["initial_speeds"], len(roads))
    rates = rng.uniform(900, 4200, len(drives))
    x, u = trajectory(c, drives, speed, rates, seconds=1.30)
    slip = np.clip(1-c["radius"]*x[..., 1]/np.maximum(x[..., 0], 1e-5), 0, 1)
    cap = np.array([.012, .035, .08, .18, .36, .64])
    anchors, caps = [], []
    if all_roads:
        specs = [(i, level) for i in range(len(drives)) for level in cap]
    else:
        ids = np.arange(count) % len(drives)
        rng.shuffle(ids)
        specs = zip(ids, rng.choice(cap, count))
    for i, level in specs:
        crossed = np.flatnonzero(slip[i] >= level)
        if not len(crossed):
            continue
        t = max(1, int(crossed[0])-1)
        if x[i, t, 0] <= c["stop_speed"]+.05:
            continue
        anchors.append((i, t))
        caps.append(float(level))
    suite = pack(c, x, u, drives, anchors, caps)
    suite["achieved"] = np.max(suite["history"][..., 2], axis=1)
    return suite


def branches(c, d, suite, *, random_seed=None):
    h = suite["history"]
    rng = None if random_seed is None else np.random.default_rng(random_seed)
    actions = candidate_actions(h[:, -1, 3], d["training_steps"], c, random=rng)
    truth = true_forecast(suite["roads"], h[:, -1, :2], actions, c)
    rows = np.arange(len(h))
    end_high = truth.max_slip >= .8
    touched = np.cumsum(end_high, axis=-1) > 0
    recovery = touched & (truth.slip <= .2)
    rec_step = np.where(recovery.any(axis=-1),
                        recovery.argmax(axis=-1)-end_high.argmax(axis=-1)+1, -1)
    return {"history": h, "actions": actions.astype(np.float32),
            "state": truth.state.astype(np.float32), "slip": truth.slip.astype(np.float32),
            "mu": truth.mu.astype(np.float32), "reward": truth.interval_mu.mean(axis=-1).astype(np.float32),
            "high": truth.high[rows].astype(np.float32),
            "lock": truth.lock[rows].astype(np.float32),
            "recovery_s": np.where(rec_step < 0, np.nan, rec_step*c["control_dt"]).astype(np.float32),
            "max_slip": truth.max_slip.max(axis=-1).astype(np.float32)}


def history_prefix(suite, i, ticks, c):
    drive, step = suite["locators"][i]
    states, actions = suite["raw"]["states"][drive], suite["raw"]["actions"][drive]
    origin = max(0, step-ticks)
    h = make_history(states, actions, origin, c["history_steps"], c)
    follow = make_history(states, actions, step, ticks+1, c)[-(step-origin+1):]
    return origin, h, follow


def historical_stress(c, original, prior):
    ab = [burckhardt(r) for r in original["test"] if r["id"] in prior["ab_ids"]]
    from .tire_prior_data import stress_pairs
    for name, roads in (("historical_ab", ab), ("historical_decoupled", stress_pairs())):
        suite = stress_suite(c, prior, roads, ab=name == "historical_ab")
        suite["achieved"] = np.max(suite["history"][..., 2], axis=1)
        yield name, suite
