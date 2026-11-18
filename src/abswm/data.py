from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import fingerprint, write_json
from .physics import TYPICAL_ROADS, advance, friction, peak_slip, slip, valid_road


def road_id(theta) -> str:
    return hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest()[:16]


def sample_roads(c):
    rng = np.random.default_rng(c["data_seed"])
    seen, splits = set(), {}
    bases = list(TYPICAL_ROADS.values())
    for split, count in c["roads"].items():
        roads = []
        while len(roads) < count:
            family = len(roads) % len(bases)
            theta = np.asarray(bases[family]) * rng.uniform(0.8, 1.2, 3)
            identifier = road_id(theta)
            if valid_road(theta) and identifier not in seen:
                seen.add(identifier)
                roads.append({"id": identifier, "family": list(TYPICAL_ROADS)[family],
                              "theta": theta.tolist(), "peak": float(peak_slip(theta))})
        splits[split] = roads
    return splits


def make_history(states, actions, t: int, history_steps: int, c):
    """Causal o_t=(v_t,w_t,lambda_t,u_{t-1}); left pad with first observation."""
    indices = np.maximum(np.arange(t-history_steps+1, t+1), 0)
    x = states[indices]
    prev_u = np.zeros(history_steps, dtype=np.float64)
    nonzero = indices > 0
    prev_u[nonzero] = actions[indices[nonzero]-1]
    return np.column_stack((x, slip(x, c["radius"]), prev_u)).astype(np.float32)


def generate_split(roads, c, split_index):
    rng = np.random.default_rng(c["data_seed"] + 101*(split_index+1))
    specs = [(i, v, k) for i in range(len(roads)) for v in c["initial_speeds"]
             for k in c["excitation_kinds"]]
    theta = np.array([roads[i]["theta"] for i, _, _ in specs])
    speeds = np.array([v for _, v, _ in specs])
    steps = max(c["history_steps"] + c["horizon_steps"] + 1,
                round(c["data_time"] / c["control_dt"]))
    states = np.zeros((len(specs), steps+1, 2), dtype=np.float32)
    actions = np.zeros((len(specs), steps), dtype=np.float32)
    states[:, 0, 0], states[:, 0, 1] = speeds, speeds/c["radius"]
    lengths = np.full(len(specs), steps+1, dtype=np.int32)
    x = states[:, 0].astype(np.float64)
    active = np.ones(len(specs), dtype=bool)
    releasing = np.zeros(len(specs), dtype=bool)
    u = np.zeros(len(specs))
    # Random excitation is independent of hidden road parameters.
    ramp_rates = rng.uniform(0.5, 1.5, len(specs))
    hold_steps = max(1, round(0.15/c["control_dt"]))
    random_levels = rng.uniform(0.05, 0.95, (len(specs), steps//hold_steps+1))
    is_ramp = np.array([k == "ramp" for _, _, k in specs])
    for t in range(steps):
        lam = slip(x, c["radius"])
        releasing = (releasing | (lam > 0.85)) & (lam > 0.15)
        ramp = np.mod((t+1)*c["control_dt"]*ramp_rates, 1.0)
        requested = c["torque_max"] * np.where(is_ramp, ramp, random_levels[:, t//hold_steps])
        requested = np.where(releasing | ~active, 0.0, requested)
        du = c["torque_rate"]*c["control_dt"]
        u = np.clip(u + np.clip(requested-u, -du, du), 0, c["torque_max"])
        u = np.where(active, u, 0.0)
        actions[:, t] = u
        new_x, _ = advance(x, u, theta, c)
        x = np.where(active[:, None], new_x, x)
        states[:, t+1] = x
        stopped = active & (x[:, 0] <= c["stop_speed"])
        lengths[stopped] = t+2
        active &= ~stopped
    mu = friction(slip(states, c["radius"]), theta[:, None, :]).astype(np.float32)
    return {"states": states, "actions": actions, "mu": mu, "lengths": lengths,
            "road_index": np.array([i for i, _, _ in specs], dtype=np.int32),
            "initial_speed": speeds, "kind": np.array([k for _, _, k in specs]),
            "peak": peak_slip(theta).astype(np.float32)}


def generate_data(c, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != fingerprint(c):
            raise ValueError("Dataset config differs; use a different output directory")
        if all((output / f"{split}.npz").exists() for split in c["roads"]):
            return manifest
    roads = sample_roads(c)
    manifest = {"config_hash": fingerprint(c), "config": c, "roads": roads,
                "observation_order": ["v_t", "omega_t", "lambda_t", "torque_previous"],
                "label_policy": "mu is auxiliary training truth; peak trains direct baseline only; theta is evaluator-only"}
    for i, split in enumerate(("train", "val", "test")):
        traces = generate_split(roads[split], c, i)
        np.savez_compressed(output / f"{split}.npz", **traces)
        print(f"data {split}: {len(roads[split])} roads, {len(traces['states'])} traces", flush=True)
    write_json(manifest_path, manifest)
    return manifest


def load_split(directory: Path, split: str):
    with np.load(Path(directory) / f"{split}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def windows(data, c, count, seed):
    """Sample entire causal windows; labels are returned separately from observations."""
    rng = np.random.default_rng(seed)
    histories, actions, states, mu, peaks, coverage, locators = [], [], [], [], [], [], []
    horizon = c["horizon_steps"]
    eligible = np.flatnonzero(data["lengths"] > horizon+1)
    if not len(eligible):
        raise ValueError("No trajectory is long enough for a rollout window")
    for _ in range(count):
        i = int(rng.choice(eligible))
        last_t = int(data["lengths"][i])-horizon-1
        upper = min(last_t, c["history_steps"]) if rng.random() < 0.4 else last_t
        t = int(rng.integers(0, upper+1))
        h = make_history(data["states"][i], data["actions"][i], t, c["history_steps"], c)
        histories.append(h)
        actions.append(data["actions"][i, t:t+horizon])
        states.append(data["states"][i, t:t+horizon+1])
        mu.append(data["mu"][i, t])
        peaks.append(data["peak"][i])
        coverage.append([h[:, 2].min(), h[:, 2].max()])
        locators.append([i, t])
    return {"history": np.asarray(histories), "actions": np.asarray(actions),
            "states": np.asarray(states), "mu": np.asarray(mu), "peak": np.asarray(peaks),
            "coverage": np.asarray(coverage), "locators": np.asarray(locators)}
