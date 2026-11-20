"""Reproduce only episodes whose decision and supervisor logs are missing."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import active_experiment
from .active_control import simulate
from .active_evaluation import options_for, runner_specs
from .active_experiment import controller_for, init_closed_worker
from .active_report import exploration_reports, trace_source
from .config import write_json
from .decoder_study import read_json
from .evaluation import write_csv


AUDIT_METHODS = {"C8", "C9", "C10", "C13"}


def episode_key(episode):
    return episode["road_id"], episode["method"], episode["seed"], episode["initial_speed"]


def safety_rows(case, trace):
    events = []
    for row in trace:
        if row["safety"]:
            events.append({key: case[key] for key in ("road_id", "method", "seed", "initial_speed")} |
                          {"step": int(row["step"]), "time_s": float(row["time_s"]),
                           "slip": float(row["slip"]), "requested_nm": float(row["requested"]) if row["requested"] else None,
                           "applied_nm": float(row["applied"]), "reason": row["safety"],
                           "high_probability": float(row["high_probability"]) if row["high_probability"] else None,
                           "lock_probability": float(row["lock_probability"]) if row["lock_probability"] else None})
    return events


def audit_case(payload):
    case, road, initial, c, d, policy, output = payload
    models, gains = active_experiment.WORKER
    key = episode_key(case)
    options = options_for(c, d, policy["horizon"], policy["observation_scale"])
    control = controller_for(case["method"], case["seed"], road, c, options, models, gains, policy)
    result, raw, _ = simulate(road, c, control, initial[0], initial_slip=initial[1], initial_torque=initial[2], keep_trace=True)
    for name, value in result.items():
        saved = case[name]
        same = bool(value == saved)
        if isinstance(value, float):
            same = np.isclose(value, saved, atol=1e-12, rtol=1e-12)
        if not same:
            raise ValueError(f"Replay of {key} does not reproduce {name}")

    def cell(value):
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return ""
        return str(value)

    trace = [{name: cell(value) for name, value in step.items()} for step in raw]
    rows, _ = exploration_reports(c, d, policy, output, [case], models, {"test_audit": [road]},
                                  source=[(key, trace)], write=False)
    return rows, safety_rows(case, trace)


def audit(c, d, args, paths, partitions, policy, models):
    output = args.output
    closed_hash = active_experiment.digest(output/"closed_loop_by_road.csv")
    policy_hash = active_experiment.digest(output/"validation_policy.json")
    record = output/"decision_audit.json"
    if record.exists():
        previous = read_json(record)
        if (previous["closed_sha256"] == closed_hash and previous["validation_policy_sha256"] == policy_hash
                and all(active_experiment.digest(output/name) == value for name, value in previous["log_sha256"].items())):
            print("Decision audit: reused complete logs for the frozen closed-loop run", flush=True)
            return
    expected = list(runner_specs(c, d, partitions, policy, args.smoke))
    episodes = [read_json(p)["metrics"] for p in sorted((output/"episodes"/"closed_loop").glob("*.json"))]
    if len(episodes) != len(expected):
        raise ValueError("Complete the frozen closed-loop evaluation before auditing")
    cases = {episode_key(e): e for e in episodes}
    existing = dict(trace_source(output))
    events, _ = exploration_reports(c, d, policy, output, episodes, models, partitions, source=existing.items(), write=False)
    guards = [event for key, trace in existing.items() for event in safety_rows(cases[key], trace)]
    replay = []
    for road, method, seed, initial, _ in expected:
        key = (road["id"], method, seed, initial[0])
        if key in existing:
            continue
        e = cases[key]
        needs_decisions = method in AUDIT_METHODS and (e["modification_count"] or e["exploration_count"])
        if needs_decisions or e["safety_count"]:
            replay.append((e, road, initial, c, d, policy, output))
    if replay:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_closed_worker,
                                 initargs=(paths, output, c, policy)) as executor:
            for index, (decisions, safety) in enumerate(executor.map(audit_case, replay), start=1):
                events.extend(decisions)
                guards.extend(safety)
                if index % 24 == 0 or index == len(replay):
                    print(f"decision audit: {index}/{len(replay)} reproduced", flush=True)
    seen = sum(bool(r["executed_exploration"]) for r in events)
    modified = sum(bool(r["modified"]) for r in events)
    if (seen != sum(e["exploration_count"] for e in episodes if e["method"] in AUDIT_METHODS)
            or modified != sum(e["modification_count"] for e in episodes)
            or len(guards) != sum(e["safety_count"] for e in episodes)):
        raise ValueError("Decision audit did not account for every registered event")
    if events:
        write_csv(output/"exploration_events.csv", events)
    if guards:
        write_csv(output/"safety_events.csv", guards)
    logs = {name: active_experiment.digest(output/name)
            for name, rows in (("exploration_events.csv", events), ("safety_events.csv", guards)) if rows}
    write_json(record, {"episodes": len(episodes), "cached_traces": len(existing),
                         "reproduced_for_missing_events": len(replay), "changed_proposals": modified,
                         "active_c10_c13": seen, "safety": len(guards), "closed_sha256": closed_hash,
                         "validation_policy_sha256": policy_hash, "log_sha256": logs})
