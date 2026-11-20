"""Independent spatial simulation, control attribution, and adaptation metrics."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .active_control import Decision, PredictiveController, Supervisor
from .active_information import information_values
from .active_metrics import values_for
from .active_rollout import CurvePredictor, NLSPredictor, candidate_actions, lower_cvar, utilities
from .physics import friction, slip


AUDIT_HORIZONS = (1, 5, 10, 20)


class SpatialOracle(CurvePredictor):
    """The sole online interface allowed to receive a position or plant specification."""
    def __init__(self, road, preview=False):
        super().__init__(road.c)
        self.road, self.preview = road, preview
        self.position = 0.

    def observe(self, history, tick):
        pass

    def predict(self, current, actions):
        x, _ = self.road.forecast([*current, self.position], actions, preview=self.preview)
        return x


class RecentNLS(NLSPredictor):
    def observe(self, history, tick):
        self.timing = {}
        if tick < 12 or tick % self.c["reference_period"]:
            return
        start = perf_counter()
        if tick % 40 == 0:
            self.estimator.theta = None
        self.reference = self.estimator(history[-20:])
        if self.estimator.theta is not None:
            from .active_rollout import GRID
            self.curves = friction(GRID, self.estimator.theta)[None]
        self.timing["nls"] = 1000*(perf_counter()-start)


@dataclass
class DriveStart:
    tick: int
    position: float
    history: np.ndarray


def merge_audit(road, x, previous, decision, applied, reason, controller, options, meta):
    if not reason or reason == "invalid_state":
        return []
    c = road.c
    delta = c["torque_rate"]*c["control_dt"]
    raw = decision.requested if np.isfinite(decision.requested) else 0.
    original = float(np.clip(raw, max(0., previous-delta), min(c["torque_max"], previous+delta)))
    h = max(AUDIT_HORIZONS)
    commands = np.repeat(np.array([[original], [applied]]), h, axis=1)
    prediction, _ = road.forecast(x, commands)
    predictor = getattr(controller, "predictor", None)
    ranking = getattr(decision, "ranked_candidates", ())
    expected = None
    if predictor is not None and (isinstance(predictor, SpatialOracle) or hasattr(predictor, "curves")):
        expected = predictor.predict(x[:2], commands)
        if not ranking and isinstance(controller, PredictiveController):
            proposal = decision.proposal if np.isfinite(decision.proposal) else None
            actions = candidate_actions(previous, controller.options["horizon"], c, proposal=proposal)[0]
            ranked = predictor.predict(x[:2], actions)
            settings = controller.options
            weights = predictor.weights
            per_hypothesis = utilities(ranked, actions, previous, settings)
            base = weights@per_hypothesis
            if controller.risk:
                mix = settings["cvar_mix"]
                base = (1-mix)*base+mix*lower_cvar(per_hypothesis, weights, settings["cvar_level"])
            if controller.information and decision.gamma:
                n = min(controller.information_steps, actions.shape[-1])
                gain, value = information_values(ranked.interval_mu[..., :n].mean(axis=-1), weights,
                                                per_hypothesis, settings["observation_scale"]/math.sqrt(n))
                signal = gain/max(math.log(len(weights)), 1e-12) if controller.information == "entropy" else value
                safe = ((weights@ranked.high <= settings["explore_high_limit"]) &
                        (weights@ranked.lock <= settings["explore_lock_limit"]))
                base += decision.gamma*signal*safe
            ranking = tuple(map(int, np.argsort(-base, kind="stable")))
    audit = []
    for n in AUDIT_HORIZONS:
        truth = prefix_forecast(prediction, n)
        scores = utilities(truth, commands[:, :n], previous, options)[0]
        ref = None if expected is None else prefix_forecast(expected, n)
        estimates = (None, None) if ref is None else predictor.weights@utilities(ref, commands[:, :n], previous, options)
        audit.append(meta | {"horizon_ms": round(n*c["control_dt"]*1000), "reason": reason,
                             "requested": decision.requested, "limited_request": original, "executed": applied,
                             "candidate_ranking": list(ranking), "predicted_high": decision.high_probability,
                             "predicted_lock": decision.lock_probability,
                             "interrupted_exploration": bool(getattr(decision, "planned_exploration",
                                                                     decision.information_changed)),
                             "predicted_original": estimates[0], "predicted_executed": estimates[1],
                             "actual_original": scores[0], "actual_executed": scores[1],
                             "actual_utility_delta": scores[1]-scores[0],
                             "original_high": bool(truth.high[0, 0]), "executed_high": bool(truth.high[0, 1]),
                             "original_lock": bool(truth.lock[0, 0]), "executed_lock": bool(truth.lock[0, 1]),
                             "original_impulse": truth.impulse[0, 0], "executed_impulse": truth.impulse[0, 1]})
    return audit


def prefix_forecast(x, n):
    from .active_rollout import Forecast

    return Forecast(x.state[..., :n, :], x.slip[..., :n], x.mu[..., :n], x.interval_mu[..., :n],
                    x.max_slip[..., :n], x.min_wheel[..., :n], x.dt, x.load)


def drive(road, controller, initial, options, *, start=None, freeze_until=None, keep=True, audit=False):
    c = road.c
    dt = c["control_dt"]
    speed, init_slip, init_torque = initial
    if start is None:
        initial_row = [speed, speed*(1-init_slip)/c["radius"], init_slip, init_torque]
        start = DriveStart(0, 0., np.repeat(np.array([initial_row]), c["history_steps"], axis=0))
    history = deque(start.history, maxlen=c["history_steps"])
    v, w, lam, previous = history[-1]
    start_v = v
    position, elapsed, steps = start.position, start.tick*dt, 0
    safety = Supervisor(c)
    if start.tick:
        safety.releasing = lam >= .8 or w <= .5
    trace, events, timings = [], [], {}
    high_time, lock_time, potential, variation = 0., 0., 0., 0.
    interventions, off_model, modified = 0, 0, 0
    detections, revisions = 0, 0
    plans = {}
    failed = ""
    predictor = getattr(controller, "predictor", None)
    command = np.zeros((1, 1))
    for tick in range(start.tick, math.ceil(c["max_time"]/dt)):
        if v <= c["stop_speed"]:
            break
        if isinstance(predictor, SpatialOracle):
            predictor.position = position
        if freeze_until is not None:
            controller.real_evidence = not (start.tick < tick <= freeze_until)
        mark = perf_counter()
        decision = controller.decide(np.asarray(history), tick)
        applied, reason = safety.apply([v, w], lam, previous, decision.requested, decision.credible)
        controller.applied(decision, applied, reason)
        if reason == "invalid_state":
            failed = reason
            break
        if hasattr(predictor, "predict_executed") and predictor.tick == tick:
            predictor.predict_executed(applied, tick+1)
        latency = 1000*(perf_counter()-mark)
        own_timing = getattr(controller, "timing", getattr(predictor, "timing", {}))
        for key, value in (own_timing | {"control": latency}).items():
            timings.setdefault(key, []).append(value)
        mark = getattr(predictor, "last", {})
        legacy = getattr(predictor, "last_event", None)
        if legacy is not None:
            mark = {"innovation": legacy.surprise, "mismatch": legacy.mismatch, "ess": legacy.effective_size,
                    "entropy_after": legacy.entropy_after}
        detections += bool(mark.get("detection"))
        revisions += bool(mark.get("revision"))
        meta = {"step": tick, "time_s": elapsed, "position_m": position,
                "innovation": mark.get("innovation"), "cusum": mark.get("cusum")}
        if audit:
            events.extend(merge_audit(road, [v, w, position], previous, decision, applied, reason,
                                      controller, options, meta))
        command[0, 0] = applied
        result, extra = road.forecast([v, w, position], command)
        new_v, new_w = result.state[0, 0, 0]
        new_position, duration, high, locked, possible = extra[:, 0, 0]
        real_explore = decision.information_changed and not reason and math.isclose(applied, decision.requested, abs_tol=1e-7)
        if getattr(decision, "segment", -1) >= 0 and decision.planned_exploration:
            plan = plans.setdefault(decision.segment, {"segment": decision.segment, "start_step": tick,
                                                        "planned_steps": decision.planned_steps, "executed_steps": 0,
                                                        "explored": False, "reason": "", "voi": decision.value,
                                                        "expected_impulse": decision.planned_impulse,
                                                        "actual_impulse": 0., "first_divergence_step": None})
            plan["actual_impulse"] += c["mass"]*(v-new_v)
            if reason:
                plan["reason"] = reason
            elif math.isclose(applied, decision.requested, abs_tol=1e-7):
                plan["executed_steps"] += 1
            if real_explore and not plan["explored"]:
                plan["first_divergence_step"] = tick
                plan["explored"] = True
        elif not hasattr(decision, "segment") and decision.information_changed:
            plans[tick] = {"segment": tick, "start_step": tick, "planned_steps": 1,
                           "executed_steps": int(real_explore), "explored": real_explore, "reason": reason,
                           "voi": decision.value, "expected_impulse": None,
                           "actual_impulse": c["mass"]*(v-new_v),
                           "first_divergence_step": tick if real_explore else None}
        if keep:
            trace.append(meta | {"v": v, "omega": w, "slip": lam, "previous": previous,
                                 "next_v": new_v, "next_omega": new_w, "next_position": new_position,
                                 "force_N": c["mass"]*(v-new_v)/max(duration, 1e-12), "duration": duration,
                                 "possible_mu": possible/max(duration, 1e-12),
                                 "observed_interval_mu": result.interval_mu[0, 0, 0],
                                 "high_s": high, "lock_s": locked, "applied": applied, "safety": reason,
                                 "actual_exploration": real_explore, **vars(decision), **mark})
        high_time += high
        lock_time += locked
        potential += possible
        elapsed += duration
        variation += abs(applied-previous)
        interventions += bool(reason)
        off_model += reason == "untrusted_prediction"
        modified += bool(decision.modification)
        steps += 1
        v, w, previous, position = new_v, new_w, applied, new_position
        lam = float(slip([v, w], c["radius"]))
        history.append([v, w, lam, previous])
    stopped = bool(v <= c["stop_speed"] and not failed)
    time = elapsed-start.tick*dt
    impulse = c["mass"]*(start_v-v)
    segments = [p | {"complete": p["executed_steps"] == p["planned_steps"],
                      "censored": p["executed_steps"] != p["planned_steps"] and not p["reason"],
                      "executed_ms": round(1000*dt*p["executed_steps"])} for p in plans.values()]
    metrics = {"distance_m": position-start.position, "time_s": time, "end_speed": v, "stopped": stopped,
               "failure": failed or ("" if stopped else "timeout"), "decel_m_s2": (start_v-v)/max(time, 1e-12),
               "impulse_ns": impulse, "friction_utilization": impulse/(c["mass"]*c["gravity"]*max(potential, 1e-12)),
               "high_slip_s": high_time, "lock_s": lock_time, "torque_variation_nm": variation, "safety_count": interventions,
               "mismatch_guard_count": off_model, "safety_rate": interventions/max(steps, 1), "steps": steps,
               "detections": detections, "revisions": revisions,
               "modification_count": modified, "exploration_count": sum(p["explored"] for p in segments),
               "planned": len(segments), "completed": sum(p["complete"] for p in segments),
               "interrupted": sum(bool(p["reason"]) for p in segments)}
    latency = [{"component": name, "samples": len(values), "p50_ms": np.median(values),
                "p95_ms": np.percentile(values, 95), "max_ms": np.max(values)} for name, values in timings.items()]
    return {"metrics": metrics, "trace": trace, "audit": events, "timing": latency, "segments": segments}


def dynamic_events(road, trace, d):
    observed = [i for i, row in enumerate(trace) if row.get("detection")]
    revised = [i for i, row in enumerate(trace) if row.get("revision")]
    used, associated = set(), set()
    changes = []
    dt = road.c["control_dt"]
    for k, position in enumerate(road.starts):
        reached = next((i for i, row in enumerate(trace) if row["next_position"] >= position), None)
        if reached is None:
            changes.append({"change": k, "position": position, "reached": False, "missed": None})
            continue
        row = trace[reached]
        crossing = row["time_s"]+row["duration"]*np.clip((position-row["position_m"])/
                                                       max(row["next_position"]-row["position_m"], 1e-12), 0, 1)
        next_change = road.starts[k+1] if k+1 < len(road.starts) else np.inf
        window = [i for i in range(reached, len(trace)) if trace[i]["position_m"] < next_change]
        probe = np.linspace(0., 1., road.profiles.shape[1])
        detectable = next((i for i in window if abs(trace[i]["observed_interval_mu"]-
                                                   np.interp((trace[i]["slip"]+float(slip(
                                                       [trace[i]["next_v"], trace[i]["next_omega"]], road.c["radius"])))/2,
                                                             probe, road.profiles[k])) >= d["detectable_mu_delta"]), None)
        last = min(window[-1], reached+round(d["change_association_ms"]/(1000*dt)))
        detection = next((i for i in observed if reached < i <= last and i not in used), None)
        if detection is not None:
            used.add(detection)
        associated.update(i for i in revised if reached < i <= last)
        refreshed = next((i for i in revised if reached < i <= last), None)
        length = d["force_recovery_steps"]
        recovered = next((i for i in window if i+length-1 <= window[-1] and
                          all(trace[j]["observed_interval_mu"] >= d["force_recovery_fraction"]*trace[j]["possible_mu"]
                              for j in range(i, i+length))), None)
        better = next((i for i in window if i >= (refreshed or reached) and i+length-1 <= window[-1] and
                       all(trace[j].get("closest_residual", np.inf) is not None and
                           trace[j].get("closest_residual", np.inf) <= 2. for j in range(i, i+length))), None)
        through = window[:100]
        lost = road.c["mass"]*road.c["gravity"]*sum(
            max(0., r["possible_mu"]-r["observed_interval_mu"])*r["duration"] for r in (trace[i] for i in through))
        last_row = trace[window[-1]]
        followup = last_row["time_s"]+last_row["duration"]-crossing
        changes.append({"change": k, "position": position, "reached": True, "crossing_s": crossing,
                        "detectable_s": None if detectable is None else trace[detectable]["time_s"]+trace[detectable]["duration"],
                        "missed": detection is None,
                        "detect_delay_s": None if detection is None else trace[detection]["time_s"]-crossing,
                        "detect_after_observable_s": None if detection is None or detectable is None else
                        trace[detection]["time_s"]-trace[detectable]["time_s"]-trace[detectable]["duration"],
                        "revision_delay_s": None if refreshed is None else trace[refreshed]["time_s"]-crossing,
                        "hypothesis_recovery_s": None if better is None else max(0., trace[better]["time_s"]-crossing),
                        "force_recovery_s": None if recovered is None else max(0., trace[recovered]["time_s"]-crossing),
                        "force_recovery_censored": recovered is None, "followup_s": max(0., followup),
                        "impulse_loss_500ms_ns": lost,
                        "post_distance_m": last_row["next_position"]-position,
                        "high_500ms_s": sum(trace[i]["high_s"] for i in through),
                        "lock_500ms_s": sum(trace[i]["lock_s"] for i in through),
                        "guard_rate_500ms": float(np.mean([bool(trace[i]["safety"]) for i in through])),
                        "first_exploration_s": next((trace[i]["time_s"]-crossing for i in window
                                                     if i > reached and trace[i]["actual_exploration"]), None)})
    return changes, len(observed)-len(used), len(revised)-len(associated)


def action_error(truth, predicted, weights, actions, options, previous, margin):
    weights = np.asarray(weights)
    y = values_for(truth)[0]
    p = np.einsum("k,kath->ath", weights, values_for(predicted))
    oracle = utilities(truth, actions, previous, options)[0]
    score = weights@utilities(predicted, actions, previous, options)
    chosen = int(np.argmax(score))
    br = truth.impulse[0]
    m = weights@predicted.impulse
    meaningful = np.triu(np.abs(br[:, None]-br) > margin*truth.load*truth.dt*actions.shape[-1], 1)
    order = np.sign(br[:, None]-br) == np.sign(m[:, None]-m)
    safe = ~(truth.high[0] | truth.lock[0])
    residual = (p[:, -1]-p[0, -1])-(y[:, -1]-y[0, -1])
    return {"wheel_rmse": float(np.sqrt(np.mean((p[..., 1]-y[..., 1])**2))),
            "force_rmse": float(np.sqrt(np.mean((p[..., 3]-y[..., 3])**2))),
            "action_wheel_rmse": float(np.sqrt(np.mean(residual[1:, 1]**2))),
            "action_force_rmse": float(np.sqrt(np.mean(residual[1:, 3]**2))),
            "pairs": int(meaningful.sum()), "correct_pairs": int((meaningful & order).sum()),
            "top1": chosen == int(np.argmax(oracle)), "chosen": chosen, "violation": not safe[chosen],
            "pure_regret_ns": float(br.max()-br[chosen]), "action_regret": float(oracle.max()-oracle[chosen]),
            "risk_regret": float(oracle[safe].max()-oracle[chosen]) if safe[chosen] else None}
