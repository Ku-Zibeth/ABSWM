from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    config = load_config(path.parent / parent) if parent else {}
    config.update(raw)
    validate_config(config)
    return config


def validate_config(c: dict) -> None:
    positive = ["physics_dt", "control_dt", "mass", "inertia", "radius", "gravity",
                "stop_speed", "torque_max", "torque_rate", "history_steps", "horizon_steps",
                "batch_size", "epochs", "data_time", "max_time", "reference_period"]
    for key in positive:
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    ratio = c["control_dt"] / c["physics_dt"]
    if abs(ratio - round(ratio)) > 1e-9 or ratio < 1:
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    if not 0 < c["reference_min"] < c["fixed_reference"] < c["reference_max"] < 1:
        raise ValueError("invalid reference limits")
    if not c["seeds"] or len(c["seeds"]) != len(set(c["seeds"])):
        raise ValueError("seeds must be nonempty and unique")
    for split in ("train", "val", "test"):
        if c["roads"][split] < 1:
            raise ValueError("each split needs at least one road")
    if any(v <= c["stop_speed"] for v in c["initial_speeds"]):
        raise ValueError("initial speed must exceed stop_speed")
    if not set(c["excitation_kinds"]) <= {"ramp", "steps"}:
        raise ValueError("unknown excitation kind")


def fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
