"""ABSWM 3.1 provenance, controller registry, episode cache and CPU workers."""
from __future__ import annotations

import gzip
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .active_control import PI, PeakSeeking, PredictiveController
from .active_evaluation import validate_policy
from .active_experiment import accelerate, digest, environment, load_baselines, probabilistic_controller
from .active_rollout import NLSPredictor, NeuralCurvePredictor, grid_coefficient, integrate
from .active_training import load_trained
from .adaptive_belief import AdaptiveObserver, ContinuousTire, ParticleSource
from .adaptive_simulation import RecentNLS, SpatialOracle, drive, dynamic_events
from .adaptive_training import make_model
from .config import fingerprint, write_json
from .decoder_study import read_json
from .evaluation import write_csv
from .feedback_control import CommittedController, ProtectionOnly, StrongPeak
from .spatial_road import SpatialRoad, make_partitions


CORE = tuple(f"B{i}" for i in range(15))
NO_TRAINING_SEED = {"B0", "B1", "B2", "B3", "B4", "B9", "protection", "old_peak",
                    "recent_nls", "preview", "library", "library_feedback"}
ABLATIONS = {
    "fixed_bank": {"revision": "R0"}, "forget": {"revision": "R1"},
    "regenerate": {"revision": "R2", "refinement": "none"},
    "refine_full": {"revision": "R2", "refinement": "full"},
    "refine_subspace": {"revision": "R2", "refinement": "subspace"},
    "refine_selected": {"revision": "R2", "refinement": "selected"},
    "window_encoder": {"revision": "window"}, "prototype": {"source": "prototype", "trained": None},
    "continuous": {"source": "continuous", "trained": None},
    "library": {"source": "library", "trained": None},
    "library_feedback": {"source": "library", "trained": None},
}


@dataclass(frozen=True)
class Paths:
    registry: Path
    base: Path
    reference: Path
    recurrent: Path
    prior: Path
    legacy: Path
    output: Path

    @property
    def baseline(self):
        return self.registry, self.base, self.reference, self.recurrent, self.prior


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def table(output, name, rows):
    rows = [clean(row) for row in rows]
    if not rows:
        return
    columns = tuple(dict.fromkeys(k for row in rows for k in row))
    write_csv(output/f"{name}.csv", [{k: row.get(k) for k in columns} for row in rows])


def seal(path, payload):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, clean(payload))
    temporary.replace(path)


def _road_ids(value):
    if isinstance(value, dict) and "id" in value:
        return {value["id"]}
    if isinstance(value, dict):
        value = value.values()
    return {key for item in value for key in _road_ids(item)}


def verify_legacy(paths):
    register = read_json(paths.legacy/"model_registry.json")
    policy_path = paths.legacy/"validation_policy.json"
    if register["current_formal_model"] != "e4_a" or digest(policy_path) != register["validation_policy_sha256"]:
        raise ValueError("3.0 model registry and validation policy do not match")
    policy = read_json(policy_path)
    validate_policy(paths.legacy, policy)
    original = read_json(paths.legacy/"experiment.json")
    for name, expected in original["reused_inputs_sha256"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Sealed 3.0 input differs: {path}")
    audit = read_json(paths.legacy/"decision_audit.json")
    if audit["closed_sha256"] != digest(paths.legacy/"closed_loop_by_road.csv"):
        raise ValueError("3.0 closed-loop data differs from the audited result")
    for name, expected in audit["log_sha256"].items():
        if digest(paths.legacy/name) != expected:
            raise ValueError(f"3.0 event log differs: {name}")
    return original, policy


def setup(paths, d, smoke=False):
    original, old_policy = verify_legacy(paths)
    c = original["base"]
    previous = [paths.base/"data"/"manifest.json", paths.prior/"frozen_partitions.json",
                paths.legacy/"frozen_partitions.json"]
    seen = set().union(*(_road_ids(read_json(p)["roads"] if p == previous[0] else read_json(p)) for p in previous))
    parts = make_partitions(c, d, seen)
    paths.output.mkdir(parents=True, exist_ok=True)
    sources = Path(__file__).parent
    files = [*sorted(sources.glob("adaptive_*.py")), sources/"feedback_control.py", sources/"spatial_road.py",
             *sorted(sources.glob("active_*.py")), Path("scripts/adaptive_belief_feedback_study.py")]
    manifest = {"name": "ABSWM 3.1", "smoke": smoke, "configuration": d, "base": c,
                "seeds": c["seeds"], "partition_sha256": fingerprint(parts),
                "source_sha256": {p.name: digest(p) for p in files},
                "legacy": {"registry": digest(paths.legacy/"model_registry.json"),
                           "experiment": digest(paths.legacy/"experiment.json"),
                           "audit": digest(paths.legacy/"decision_audit.json"),
                           "open_training_data": digest(paths.legacy/"data"/"mixed_more.pt")}}
    key = fingerprint(manifest)
    target = paths.output/"experiment.json"
    if target.exists() and read_json(target)["key"] != key:
        raise ValueError("The study configuration, inputs or code changed. Choose a new --output directory.")
    if not target.exists():
        seal(paths.output/"frozen_partitions.json", parts)
        seal(paths.output/"dynamic_road_manifest.json", {name: p["dynamic"] for name, p in parts.items()})
        seal(target, manifest | {"key": key})
    elif fingerprint(read_json(paths.output/"frozen_partitions.json")) != manifest["partition_sha256"]:
        raise ValueError("The stored frozen partitions have changed")
    return c, old_policy, parts, key


def models_for(paths, c, d, old_policy):
    _, models, gains = load_baselines(paths.baseline)
    name = old_policy["variant"]
    models.update(load_trained(paths.legacy, c, old_policy["study"], variants={name},
                               expected_key=old_policy["training_key"]))
    for seed in c["seeds"]:
        models["prototype", seed] = models[name, seed]
        models["continuous", seed] = ContinuousTire(models[name, seed], d["continuous_scale"]).eval().requires_grad_(False)
    return models, gains


def control_options(c, d, old_policy):
    return old_policy["study"]["options"] | {"horizon": old_policy["horizon"],
                                             "slew_nm": c["torque_rate"]*c["control_dt"],
                                             "observation_scale": old_policy["observation_scale"]/
                                             np.sqrt(old_policy["study"]["likelihood_reliability"])}


def load_new_models(paths, models, trained, key):
    for name, expected in trained.items():
        kind, distribution, count = name.split("_")
        seed = int(count)
        target = paths.output/"models"/f"{name}.pt"
        if digest(target) != expected:
            raise ValueError(f"Frozen model changed: {target}")
        record = torch.load(target, weights_only=True)
        if record["key"] != key:
            raise ValueError(f"Model belongs to a different training experiment: {target}")
        donor = models["e4" if kind == "e4" else "prototype", seed]
        model = make_model(kind, donor, read_json(paths.output/"experiment.json")["configuration"])
        model.load_state_dict(record["state"])
        models[kind+"_"+distribution, seed] = model.eval().requires_grad_(False)


def observer_for(kind, seed, c, d, policy, models, *, revision=None, refinement=None, trained=None):
    if kind == "ensemble":
        model = tuple(models["e4", s] for s in c["seeds"])
    elif kind == "library":
        model = None
    else:
        model = models[kind if trained is None else f"{kind}_{trained}", seed]
    return AdaptiveObserver(c, d, ParticleSource(kind, model, seed),
                            revision=revision or policy.get("revision", "R2"),
                            refinement=refinement or policy.get("refinement", "none"),
                            scale=policy.get("innovation_scale"), threshold=policy.get("innovation_threshold"))


def create_controller(method, seed, c, d, road, policy, models, gains, old, length=None, override=None):
    seed = seed or c["seeds"][0]
    options = control_options(c, d, old)
    peak = policy.get("peak", d["peak_candidates"][0])
    if method in {"B0", "B1", "B5"}:
        predictor = None if method == "B0" else NLSPredictor(c) if method == "B1" else NeuralCurvePredictor(c, [models["e4", seed]])
        return PI(c, gains, predictor)
    if method in {"B2", "B3", "B6", "recent_nls", "preview", "e4_closed", "strong_deterministic"}:
        if method in {"B2", "recent_nls"}:
            predictor = NLSPredictor(c) if method == "B2" else RecentNLS(c)
        elif method in {"B3", "preview"}:
            predictor = SpatialOracle(road, preview=method == "preview")
            return PredictiveController(c, options, predictor)
        else:
            key = "e4_closed" if method == "e4_closed" else "e4"
            predictor = NeuralCurvePredictor(c, [models[key, seed]])
        controller = PredictiveController(c, options, accelerate(predictor))
        if method == "strong_deterministic":
            controller.peaks = StrongPeak(c, peak)
        return controller
    if method in {"B7", "B8", "old_cvar", "old_rk4"}:
        mapping = {"B7": "C11", "B8": "C13", "old_cvar": "C12", "old_rk4": "C11"}
        controller = probabilistic_controller(mapping[method], seed, c, options, models, old)
        if method == "old_rk4":
            predictor = controller.predictor

            def predict(state, actions):
                coefficients = grid_coefficient(predictor.curves)
                expanded = np.broadcast_to(actions, (len(predictor.curves), *actions.shape))
                return integrate(state, expanded, coefficients, c, rk4=True)

            predictor.predict = predict
        return controller
    if method == "B4":
        return StrongPeak(c, peak)
    if method == "old_peak":
        return PeakSeeking(c)
    if method == "protection":
        return ProtectionOnly(c)
    params = {"source": policy.get("source", "prototype"), "trained": policy.get("trained"),
              "revision": policy.get("revision", "R2"), "refinement": policy.get("refinement", "none")}
    params.update(ABLATIONS.get(method, {}))
    params.update(override or {})
    if method == "B9":
        params.update(source="ensemble", trained=None)
    observer = observer_for(params["source"], seed, c, d, policy, models,
                            revision=params["revision"], refinement=params["refinement"], trained=params["trained"])
    mode = {"B12": "entropy", "B13": "feedback", "B14": "frozen_feedback", "task_proxy": "task",
            "library_feedback": "feedback", "strong_feedback": "feedback"}.get(method, "passive")
    search = StrongPeak(c, peak) if method in {"B11", "strong_feedback"} else None
    commit = length or policy.get("commit_steps", d["commit_steps"][1])
    return CommittedController(c, d, options, observer, commit, mode, peak=search)


def specs(scenes, methods, c, d, *, smoke=False, length=None, trace=False, audit=False, single_seed=False):
    initial = d["initial_conditions"][:1] if smoke else d["initial_conditions"]
    for road, method, state in itertools.product(scenes, methods, initial):
        seeds = (0,) if method in NO_TRAINING_SEED else c["seeds"][:1] if smoke or single_seed else c["seeds"]
        for seed in seeds:
            yield {"scene": road, "method": method, "seed": seed, "initial": state, "length": length,
                   "trace": trace, "audit": audit, "override": {}}


def _init_episode_worker(paths, c, d, old, weights, training_key):
    global WORKER
    environment(c["torch_threads"])
    models, gains = models_for(paths, c, d, old)
    if weights:
        load_new_models(paths, models, weights, training_key)
    WORKER = (models, gains)


def _episode_job(payload):
    c, d, old, policy, spec, folder, key = payload
    x = spec["scene"]
    seed, condition, method = spec["seed"], spec["initial"], spec["method"]
    name = f"{x['id']}_{method}_{seed}_{condition[0]:g}_{fingerprint(spec)[:12]}.json.gz"
    target = folder/name
    cache_key = fingerprint([key, spec, policy])
    if target.exists():
        with gzip.open(target, "rt", encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved["key"] == cache_key:
            return saved["episode"]
    models, gains = WORKER
    road = SpatialRoad.build(x, c)
    control = create_controller(method, seed, c, d, road, policy, models, gains, old,
                                spec["length"], spec["override"])
    episode = drive(road, control, condition, control_options(c, d, old),
                    keep=spec["trace"] or bool(road.starts.size), audit=spec["audit"])
    changes, false_detect, false_rev = dynamic_events(road, episode["trace"], d)
    if not road.starts.size:
        false_detect, false_rev = episode["metrics"]["detections"], episode["metrics"]["revisions"]
    episode["changes"] = changes
    meta = {"method": method, "seed": seed, "road_id": x["id"], "family": x["family"], "kind": x["kind"],
            "initial_speed": condition[0], "initial_slip": condition[1], "initial_torque": condition[2],
            "commit_steps": getattr(control, "length", 1)}
    episode["metrics"] = meta | episode["metrics"] | {"false_detection": false_detect, "false_revision": false_rev}
    if not spec["trace"]:
        episode["trace"] = []
    temporary = target.with_suffix(".tmp")
    episode = clean(episode)
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump({"key": cache_key, "episode": episode}, stream, allow_nan=False)
    temporary.replace(target)
    return episode


def run_episodes(paths, c, d, old, policy, jobs, phase, key, workers, *, weights=None, training_key=None):
    folder = paths.output/"episodes"/phase
    folder.mkdir(parents=True, exist_ok=True)
    payload = [(c, d, old, policy, job, folder, key) for job in jobs]
    if not payload:
        return []
    result = []
    init = (paths, c, d, old, weights, training_key)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_episode_worker, initargs=init) as pool:
        for episode in pool.map(_episode_job, payload, chunksize=1):
            result.append(episode)
            if len(result) % 12 == 0 or len(result) == len(payload):
                print(f"{phase}: {len(result)}/{len(payload)}", flush=True)
    table(paths.output, phase+"_by_case", [r["metrics"] for r in result])
    return result
