from __future__ import annotations

import math
import time

import numpy as np
import torch
from scipy.optimize import least_squares

from .data import make_history
from .physics import TYPICAL_ROADS, advance_scalar, friction, peak_slip


class LeastSquaresPeak:
    """Only causal speed increments and observed slip; no true force or theta."""
    def __init__(self, c):
        self.c, self.theta = c, None

    def __call__(self, history):
        x = np.asarray(history, dtype=float)
        lam = (x[1:, 2] + x[:-1, 2])/2
        mu = -(x[1:, 0]-x[:-1, 0])/(self.c["control_dt"]*self.c["gravity"])
        mask = (lam > 0.002) & (mu >= 0) & (mu <= 2.0)
        lam, mu = lam[mask], mu[mask]
        if len(lam) < 6 or np.ptp(lam) < 0.003:
            return self.c["fixed_reference"] if self.theta is None else float(peak_slip(self.theta))
        starts = [self.theta] if self.theta is not None else list(TYPICAL_ROADS.values())
        best = None
        for initial in starts:
            result = least_squares(lambda t: friction(lam, t)-mu, initial,
                                   bounds=([0.03, 3.0, 0.001], [2.0, 160.0, 1.3]),
                                   x_scale="jac", max_nfev=80 if self.theta is None else 30)
            if np.all(np.isfinite(result.x)) and (best is None or result.cost < best.cost):
                best = result
        if best is not None:
            self.theta = best.x
        return self.c["fixed_reference"] if self.theta is None else float(peak_slip(self.theta))


class NeuralPeak:
    def __init__(self, model, device):
        self.model, self.device = model, device

    def __call__(self, history):
        with torch.inference_mode():
            h = torch.as_tensor(history[None], dtype=torch.float32, device=self.device)
            return float(self.model.peak(h)[0].cpu())


class SlipPI:
    def __init__(self, c, kp, ki):
        self.c, self.kp, self.ki = c, kp, ki
        self.integral, self.torque, self.releasing = 0.0, 0.0, False

    def step(self, lam, reference):
        c = self.c
        self.releasing = (self.releasing or lam > 0.8) and lam > 0.2
        error = reference-lam
        raw = self.kp*error+self.ki*self.integral
        if self.releasing:
            raw, self.integral = 0.0, 0.0
        delta = c["torque_rate"]*c["control_dt"]
        applied = float(np.clip(raw, max(0, self.torque-delta), min(c["torque_max"], self.torque+delta)))
        # Conditional integration handles both amplitude and slew-rate saturation.
        if not self.releasing and (raw-applied)*error <= 1e-10:
            self.integral += error*c["control_dt"]
        self.torque = applied
        return applied


def simulate_closed(theta, initial_speed, c, gains, selector=None, oracle=False):
    v, w = float(initial_speed), float(initial_speed/c["radius"])
    dt = c["control_dt"]
    states, actions, references, estimates, durations = [[v, w]], [], [], [], []
    reference, estimate = c["fixed_reference"], c["fixed_reference"]
    controller = SlipPI(c, *gains)
    true_peak = float(peak_slip(theta))
    timings = []
    distance, elapsed, high_slip = 0.0, 0.0, 0.0
    for step in range(math.ceil(c["max_time"]/dt)):
        lam = min(1.0, max(0.0, 1-c["radius"]*w/v))
        if oracle:
            estimate = true_peak
        elif selector is not None and step >= c["history_steps"] and step % c["reference_period"] == 0:
            h = make_history(np.asarray(states), np.asarray(actions), step, c["history_steps"], c)
            start = time.perf_counter()
            estimate = float(selector(h))
            timings.append(1000*(time.perf_counter()-start))
            if not math.isfinite(estimate):
                raise FloatingPointError("Reference estimator returned a non-finite slip")
        requested = float(np.clip(estimate, c["reference_min"], c["reference_max"]))
        delta = c["reference_rate"]*dt
        reference += float(np.clip(requested-reference, -delta, delta))
        torque = controller.step(lam, reference)
        new_v, new_w, increment = advance_scalar(v, w, torque, theta, c)
        duration = dt
        if new_v <= c["stop_speed"]:
            fraction = (v-c["stop_speed"])/max(v-new_v, 1e-12)
            duration = dt*fraction
            increment = (v+c["stop_speed"])*duration/2
            new_w = w+fraction*(new_w-w)
            new_v = c["stop_speed"]
        distance += increment
        elapsed += duration
        high_slip += duration*(lam > 0.8)
        actions.append(torque)
        references.append(reference)
        estimates.append(estimate)
        durations.append(duration)
        states.append([new_v, new_w])
        v, w = new_v, new_w
        if v <= c["stop_speed"]:
            break
    states, actions = np.asarray(states), np.asarray(actions)
    refs, est, weights = np.asarray(references), np.asarray(estimates), np.asarray(durations)
    lams = np.clip(1-c["radius"]*states[:-1, 1]/states[:-1, 0], 0, 1)
    after_warmup = np.arange(len(est)) >= c["history_steps"]
    metrics = {"distance_m": distance, "time_s": elapsed, "stopped": bool(v <= c["stop_speed"]),
               "high_slip_s": high_slip,
               "tracking_rmse": float(np.sqrt(np.average((lams-refs)**2, weights=weights))),
               "peak_mae": float(np.mean(np.abs(est[after_warmup]-true_peak))) if after_warmup.any() else None,
               "torque_total_variation": float(np.abs(np.diff(np.r_[0, actions])).sum()),
               "selector_mean_ms": float(np.mean(timings)) if timings else 0.0,
               "selector_p95_ms": float(np.percentile(timings, 95)) if timings else 0.0,
               "selector_updates": len(timings)}
    trace = {"states": states, "actions": actions, "reference": refs, "estimate": est,
             "time": np.r_[0, np.cumsum(weights)], "slip": np.r_[lams, min(1., max(0., 1-c['radius']*w/v))]}
    return metrics, trace


def tune_controller(c, roads, output):
    candidates = []
    for kp in (1000.0, 3000.0, 6000.0):
        for ki in (1000.0, 6000.0, 20000.0):
            scores = []
            for road in roads:
                for speed in c["initial_speeds"]:
                    for oracle in (False, True):
                        result, _ = simulate_closed(road["theta"], speed, c, (kp, ki), oracle=oracle)
                        scores.append(result["distance_m"] + 100*result["high_slip_s"]
                                      + 10000*(not result["stopped"]))
            candidates.append({"kp": kp, "ki": ki, "validation_cost": float(np.mean(scores))})
    best = min(candidates, key=lambda x: x["validation_cost"])
    from .config import write_json
    write_json(output/"controller.json", {"selected": best, "candidates": candidates,
               "tuned_on": "validation roads only; fixed and true-peak references",
               "cost": "distance_m + 100*high_slip_s + 10000*not_stopped"})
    print(f"controller kp={best['kp']} ki={best['ki']}", flush=True)
    return best["kp"], best["ki"]
