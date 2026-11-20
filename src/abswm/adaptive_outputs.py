"""Phase outputs, road-cluster comparisons, and research-only acceptance."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .active_metrics import control_summary, grouped, paired_control
from .adaptive_experiment import seal, table


CONTRASTS = (("B6", "B5"), ("B4", "old_peak"), ("strong_deterministic", "B4"), ("B11", "B4"),
             ("B10", "B7"), ("B10", "B9"), ("B10", "library"), ("B13", "B10"), ("B12", "B10"),
             ("task_proxy", "B10"), ("B13", "B14"), ("B13", "task_proxy"), ("B13", "library_feedback"),
             ("forget", "fixed_bank"), ("regenerate", "fixed_bank"), ("refine_selected", "regenerate"),
             ("refine_subspace", "refine_full"), ("continuous", "prototype"),
             ("e4_closed", "B6"), ("pretrain", "B10"), ("strong_feedback", "B11"),
             ("preview", "B3"), ("recent_nls", "B2"), ("old_rk4", "B7"))


def report_episodes(output, episodes, phase="", *, traces=False):
    events, segments, changes, timing, beliefs, all_rows = [], [], [], [], [], []
    for e in episodes:
        m = e["metrics"]
        meta = {n: m[n] for n in ("method", "seed", "road_id", "kind", "family", "initial_speed", "commit_steps")}
        events.extend(meta | r for r in e["audit"])
        segments.extend(meta | r for r in e["segments"])
        changes.extend(meta | r for r in e["changes"])
        timing.extend(meta | r for r in e["timing"])
        for r in e["trace"]:
            if r.get("innovation") is not None:
                beliefs.append(meta | {n: r.get(n) for n in (
                    "step", "time_s", "position_m", "innovation", "cusum", "detection", "mismatch",
                    "closest_residual", "censored_residual", "revision", "generation", "screen_intervals",
                    "entropy_before", "entropy_after", "ess", "curve_disagreement", "observed_mu",
                    "predictive_mu_mean", "refine_rank", "refine_before", "refine_after", "refine_movement")})
            if r["actual_exploration"]:
                all_rows.append(meta | {n: r.get(n) for n in ("step", "time_s", "position_m", "segment",
                                                               "gain", "value", "applied", "planned_exploration",
                                                               "observed_interval_mu", "safety")})
        if traces and e["trace"]:
            suffix = m["road_id"]+"_"+m["method"]+f"_{m['seed']}_{m['initial_speed']:g}_{m['commit_steps']}"
            table(output/"traces", suffix, e["trace"])
    table(output, phase+"safety_intervention_audit", [r for r in events if r["horizon_ms"] == 5])
    table(output, phase+"protection_counterfactual", events)
    table(output, phase+"exploration_interruption", segments)
    table(output, phase+"exploration_events", all_rows)
    table(output, phase+"change_detection", changes)
    table(output, phase+"dynamic_road_risk", [r for r in changes if r["reached"]])
    table(output, phase+"force_recovery", [r for r in changes if r["reached"]])
    table(output, phase+"adaptation_delay", changes)
    table(output, phase+"belief_revision", beliefs)
    table(output, phase+"mismatch_detection", [r for r in beliefs if r.get("mismatch") or r.get("detection")])
    table(output, phase+"hypothesis_generation", [r for r in beliefs if r.get("revision") == "regenerate"])
    table(output, phase+"latent_refinement", [r for r in beliefs if r.get("refine_rank") is not None])
    table(output, phase+"timing", timing)
    return changes


def closed_results(output, episodes, d, phase=""):
    rows = [e["metrics"] for e in episodes]
    constant = [r for r in rows if r["kind"] == "constant"]
    dynamic = [r for r in rows if r["kind"] != "constant"]
    table(output, phase+"integrated_summary", control_summary(rows, ("method", "kind")))
    table(output, phase+"constant_road_summary", control_summary(constant))
    table(output, phase+"constant_road_by_family", control_summary(constant, ("method", "family")))
    table(output, phase+"dynamic_road_summary", control_summary(dynamic, ("method", "kind")))
    table(output, phase+"dynamic_road_by_case", dynamic)
    paired = []
    for name, group in (("constant", constant), ("dynamic", dynamic)):
        paired.extend({"partition": name, **r} for r in paired_control(group, d["bootstrap_samples"], d["seed"], CONTRASTS))
    table(output, phase+"paired_statistics", paired)
    table(output, phase+"control_ablation", paired)
    table(output, phase+"protection_only", [r for r in rows if r["method"] == "protection"])
    table(output, phase+"peak_seeking_comparison", [r for r in rows if r["method"] in
          {"old_peak", "B4", "strong_deterministic", "B11", "strong_feedback"}])
    return constant, dynamic


def duration_results(output, episodes, d, phase=""):
    episodes = list(episodes)
    rows = [x["metrics"] for x in episodes]
    table(output, phase+"exploration_duration", control_summary(rows, ("method", "kind", "commit_steps")))
    contrasts = []
    for (length,), group in grouped(rows, ("commit_steps",)).items():
        contrasts.extend({"commit_steps": length, **r} for r in paired_control(
            group, d["bootstrap_samples"], d["seed"], (("B13", "B14"), ("B13", "B10"))))
    table(output, phase+"duration_paired", contrasts)


def _pairs(rows, first, second):
    key = ("road_id", "seed", "initial_speed", "kind")
    a = {tuple(r[k] for k in key): r for r in rows if r["method"] == first}
    b = {tuple(r[k] for k in key): r for r in rows if r["method"] == second}
    return [(a[k], b[k]) for k in sorted(a.keys() & b.keys())]


def _paired_change(changes, first, second, metric, samples, seed):
    key = ("road_id", "seed", "initial_speed", "change")
    left = {tuple(r[k] for k in key): r for r in changes if r["method"] == first and r["reached"]}
    right = {tuple(r[k] for k in key): r for r in changes if r["method"] == second and r["reached"]}
    by_road = defaultdict(list)
    for k in sorted(left.keys() & right.keys()):
        a, b = left[k], right[k]
        if metric == "force_recovery_s":
            # Use common restricted follow-up. Censored cases stay in the denominator.
            limit = min(a["followup_s"], b["followup_s"], .5)
            av = limit if a[metric] is None else min(limit, a[metric])
            bv = limit if b[metric] is None else min(limit, b[metric])
        else:
            av, bv = a[metric], b[metric]
            if av is None or bv is None:
                continue
        by_road[k[0]].append(av-bv)
    groups = np.array([np.mean(v) for v in by_road.values()])
    if not len(groups):
        return {"road_clusters": 0, "mean_delta": None, "lo95": None, "hi95": None}
    rng = np.random.default_rng(seed)
    draw = rng.integers(len(groups), size=(samples, len(groups)))
    limits = np.quantile(groups[draw].mean(axis=1), [.025, .975])
    return {"road_clusters": len(groups), "mean_delta": groups.mean(), "median_delta": np.median(groups),
            "sd": groups.std(ddof=1) if len(groups) > 1 else 0., "worst_delta": groups.max(),
            "lo95": limits[0], "hi95": limits[1]}


def adaptation_statistics(output, changes, d):
    contrasts = (("B13", "B14"), ("B13", "B10"), ("B10", "fixed_bank"),
                 ("regenerate", "fixed_bank"), ("refine_selected", "regenerate"), ("forget", "fixed_bank"))
    rows = []
    for a, b in contrasts:
        for metric in ("force_recovery_s", "impulse_loss_500ms_ns", "high_500ms_s", "lock_500ms_s",
                       "guard_rate_500ms", "detect_delay_s", "hypothesis_recovery_s"):
            comparison = _paired_change(changes, a, b, metric, d["bootstrap_samples"], d["seed"])
            rows.append({"method": a, "baseline": b, "metric": metric, **comparison})
    table(output, "adaptation_paired_statistics", rows)
    return rows


def _distance_gain(rows, left, right):
    matched = _pairs(rows, left, right)
    if not matched:
        return None
    a, b = zip(*matched)
    if not all(r["stopped"] for r in (*a, *b)):
        # Completion is a separate gate. Censored distances cannot win the distance gate.
        return None
    return 1-np.mean([r["distance_m"] for r in a])/np.mean([r["distance_m"] for r in b])


def _delta(rows, left, right, name):
    paired = _pairs(rows, left, right)
    return float(np.mean([float(a[name])-float(b[name]) for a, b in paired])) if paired else None


def acceptance(output, c, d, policy, constant, dynamic, adaptation, probability, risks, smoke):
    gate = policy["thresholds"]
    fixed = [r for r in constant if r["method"] == "B10"]
    old = [r for r in constant if r["method"] == "B7"]
    old_guard = sum(r["mismatch_guard_count"] for r in old)/max(1, sum(r["steps"] for r in old))
    new_guard = sum(r["mismatch_guard_count"] for r in fixed)/max(1, sum(r["steps"] for r in fixed))
    false_rev = sum(r["false_revision"] for r in fixed)/max(.01, sum(r["time_s"] for r in fixed))
    cov = [r for r in probability if r["kind"] == "constant" and r["method"] == "adaptive"
           and r["nominal"] == .9 and r["flavor"] == "calibrated"]
    denominator = {(r["suite"], r["excitation"], r["variable"]): r["width"] for r in probability
                   if r["kind"] == "constant" and r["method"] == "e4" and r["nominal"] == .9 and r["flavor"] == "calibrated"}
    comparison = [r["width"]/max(denominator[r["suite"], r["excitation"], r["variable"]], 1e-8)
                  for r in cov if (r["suite"], r["excitation"], r["variable"]) in denominator]
    base_risk = [r for r in risks if r["suite"] in {"burckhardt", "magic", "rational"}
                 and r["method"] == "adaptive_calibrated" and r["event"] == "high" and r["fnr"] is not None]
    raw_risk = [r for r in risks if r["suite"] in {"burckhardt", "magic", "rational"}
                and r["method"] == "adaptive_raw" and r["event"] == "high" and r["fnr"] is not None]
    recovery = next(r["mean_delta"] for r in adaptation if r["method"] == "B10" and
                    r["baseline"] == "fixed_bank" and r["metric"] == "force_recovery_s")
    no_learn = _distance_gain(dynamic, "B13", "B14")
    no_active = _distance_gain(dynamic, "B13", "B10")
    static_delta = _distance_gain(constant, "B13", "B7")
    measures = {"dynamic_gain_vs_frozen_feedback": no_learn, "dynamic_gain_vs_passive": no_active,
                "constant_gain_vs_legacy": static_delta, "recovery_delta_s": recovery,
                "raw_fnr": max((r["fnr"] for r in raw_risk), default=None),
                "fnr": max((r["fnr"] for r in base_risk), default=None),
                "brier": max((r["brier"] for r in base_risk), default=None),
                "coverage_error": max((abs(r["coverage"]-.9) for r in cov), default=None),
                "interval_width_ratio": max(comparison, default=None), "false_refresh_per_second": false_rev,
                "mismatch_guard_reduction": 1-new_guard/old_guard if old_guard > 0 else (1. if new_guard == 0 else None)}
    safety = {}
    for label, group in (("constant", constant), ("dynamic", dynamic)):
        safety[label] = {name: _delta(group, "B13", "B14", name) for name in ("high_slip_s", "lock_s", "stopped")}
    checks = {
        "constant_distance": static_delta is not None and static_delta >= -gate["constant_distance_max_regression"],
        "recovery": recovery is not None and recovery <= -gate["min_dynamic_recovery_improvement_s"],
        "dynamic_gain": no_learn is not None and no_active is not None and
        min(no_learn, no_active) >= gate["min_active_dynamic_distance_gain"],
        "false_refresh": false_rev <= gate["max_false_refresh_per_second"],
    }
    for name, cutoff in (("raw_fnr", "max_fnr"), ("fnr", "max_fnr"), ("brier", "max_brier"),
                         ("coverage_error", "max_coverage_error"), ("interval_width_ratio", "max_interval_width_ratio")):
        checks[name] = measures[name] is not None and measures[name] <= gate[cutoff]
    checks["mismatch_guards"] = (measures["mismatch_guard_reduction"] is not None and
                                  measures["mismatch_guard_reduction"] >= gate["min_mismatch_guard_relative_reduction"])
    for label, rows in safety.items():
        checks[label+"_safety"] = (all(v is not None for v in rows.values()) and
                                    rows["stopped"] >= -gate["max_failure_increase"] and
                                    rows["high_slip_s"] <= gate["max_high_slip_increase_s"] and
                                    rows["lock_s"] <= gate["max_lock_increase_s"])
    result = {"passed": all(checks.values()) and not smoke, "checks": checks, "values": measures, "safety": safety,
              "thresholds": gate, "smoke": smoke, "current_formal_model": "e4_a",
              "policy_key": policy["key"], "candidate": policy["source"]+"_"+str(policy.get("trained"))}
    seal(output/"acceptance.json", result)
    return result
