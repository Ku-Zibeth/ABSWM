"""Receding-horizon ABS, baseline controllers, and a shared safety supervisor."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .active_information import entropy, information_values
from .active_rollout import candidate_actions, lower_cvar, utilities
from .physics import advance_scalar


class Supervisor:
    def __init__(self, c):
        self.c, self.releasing = c, False

    def apply(self, current, lam, previous, requested, credible=True):
        c = self.c
        if not np.isfinite(current).all() or min(current) < 0:
            reason = "invalid_state"
        elif current[0] <= c["stop_speed"]:
            reason = "low_speed"
        elif current[1] <= .5:
            self.releasing = True
            reason = "locked"
        elif lam >= .8 or (self.releasing and lam > .2):
            self.releasing = True
            reason = "high_slip"
        elif not math.isfinite(requested):
            reason = "unavailable_action"
        else:
            self.releasing = False
            reason = "" if credible else "untrusted_prediction"
        if reason == "untrusted_prediction":
            delta = c["control_dt"]*c["torque_rate"]
            requested = previous+delta*np.clip((c["fixed_reference"]-lam)*4, -1, .25)
        elif reason:
            requested = 0.
        delta = c["control_dt"]*c["torque_rate"]
        applied = min(min(previous+delta, c["torque_max"]), max(max(0., previous-delta), requested))
        return float(applied), reason


class PI:
    def __init__(self, c, gains, predictor=None):
        self.c, self.kp, self.ki = c, *gains
        self.predictor = predictor
        self.integral = self.error = 0.
        self.reference = self.target = c["fixed_reference"]

    def decide(self, history, tick):
        c = self.c
        if self.predictor is not None and tick >= c["history_steps"] and not tick % c["reference_period"]:
            self.predictor.observe(history, tick)
            if hasattr(self.predictor, "reference"):
                self.target = self.predictor.reference
            else:
                from .active_rollout import GRID
                self.target = GRID[np.argmax(self.predictor.curves[0])]
        else:
            if self.predictor is not None:
                self.predictor.timing = {}
        desired = np.clip(self.target, c["reference_min"], c["reference_max"])
        rate = c["control_dt"]*c["reference_rate"]
        self.reference += np.clip(desired-self.reference, -rate, rate)
        self.error = self.reference-history[-1, 2]
        return Decision(float(self.kp*self.error+self.ki*self.integral))

    def applied(self, decision, torque, reason):
        if reason:
            self.integral = 0.
        elif (decision.requested-torque)*self.error <= 1e-10:
            self.integral += self.c["control_dt"]*self.error


class PeakSeeking:
    """Perturb torque and reverse direction when causal mean force decreases."""
    def __init__(self, c):
        self.c = c
        self.direction, self.previous_mu = 1., None
        self.baseline_mu, self.anchor_torque = None, 0.
        self.period = 5

    def propose(self, history, tick):
        previous = history[-1, 3]
        if tick and tick % self.period == 0:
            v = history[-self.period-1:, 0]
            mu = (v[0]-v[-1])/(self.period*self.c["control_dt"]*self.c["gravity"])
            if self.baseline_mu is not None:
                moved = previous-self.anchor_torque
                if abs(moved) > 1. and abs(mu-self.baseline_mu) > .0015:
                    self.direction = math.copysign(1., (mu-self.baseline_mu)*moved)
            self.baseline_mu, self.anchor_torque = mu, previous
        if history[-1, 2] > .45 or previous >= self.c["torque_max"]-1:
            self.direction = -1.
        if previous < 1. and history[-1, 2] < .15:
            self.direction = 1.
        return float(previous+12.5*self.direction)

    def decide(self, history, tick):
        return Decision(self.propose(history, tick))

    def applied(self, decision, torque, reason):
        if reason:
            self.direction = -1.
            self.baseline_mu = None


@dataclass
class Decision:
    requested: float
    credible: bool = True
    candidate: int = -1
    passive: int = -1
    proposal: float = math.nan
    high_probability: float = math.nan
    lock_probability: float = math.nan
    gain: float = 0.
    value: float = 0.
    gamma: float = 0.
    posterior_entropy: float = math.nan
    ess: float = math.nan
    decision_gap: float = math.nan
    information_changed: bool = False
    modification: str = ""


class PredictiveController:
    def __init__(self, c, options, predictor, *, assistant=False, risk=False, information="", gamma=0.,
                 weighting="fixed", information_steps=1):
        self.c, self.options, self.predictor = c, options, predictor
        self.risk, self.information, self.gamma = risk, information, gamma
        self.weighting, self.information_steps = weighting, information_steps
        self.peaks = PeakSeeking(c) if assistant else None
        self.timing = {}

    def evaluate(self, current, actions, previous):
        start = perf_counter()
        forecast = self.predictor.predict(current, actions)
        self.timing["rollout"] = 1000*(perf_counter()-start)
        start = perf_counter()
        weights = self.predictor.weights
        score = utilities(forecast, actions, previous, self.options)
        if not np.isfinite(score).all() or not np.isfinite(weights).all():
            return forecast, Decision(math.nan, False)
        expected = weights@score
        high, lock = weights@forecast.high, weights@forecast.lock
        utility = expected.copy()
        if self.risk:
            level = self.options["cvar_level"]
            mix = self.options["cvar_mix"]
            utility = (1-mix)*utility+mix*lower_cvar(score, weights, level)
        # The risk term remains active for every controller that uses these candidates.
        passive = int(np.argmax(utility))
        posterior_entropy = float(entropy(weights))
        regret = max(0., float(weights@score.max(axis=1)-expected.max()))
        gain, value = np.zeros_like(utility), np.zeros_like(utility)
        gamma = self.gamma
        self.timing["mpc"] = 1000*(perf_counter()-start)
        if gamma and self.information:
            start = perf_counter()
            n = min(self.information_steps, actions.shape[-1])
            observation = forecast.interval_mu[..., :n].mean(axis=-1)
            scale = self.options["observation_scale"]/math.sqrt(n)
            gain, value = information_values(observation, weights, score, scale)
            if self.weighting in {"speed", "dynamic"}:
                gamma *= np.clip((current[0]-self.c["stop_speed"])/15, 0, 1)
            if self.weighting in {"entropy", "dynamic"}:
                gamma *= posterior_entropy/max(math.log(len(weights)), 1e-12)
            if self.weighting in {"disagreement", "dynamic"}:
                gamma *= min(regret/.015, 1.)
            signal = gain/max(math.log(len(weights)), 1e-12) if self.information == "entropy" else value
            explore_risk = (high <= self.options["explore_high_limit"]) & (lock <= self.options["explore_lock_limit"])
            utility += gamma*signal*explore_risk
            self.timing["information"] = 1000*(perf_counter()-start)
        chosen = int(np.argmax(utility))
        changed = not math.isclose(actions[chosen, 0], actions[passive, 0], abs_tol=1e-7)
        return forecast, Decision(float(actions[chosen, 0]), self.predictor.trusted, chosen, passive,
                                  high_probability=float(high[chosen]), lock_probability=float(lock[chosen]),
                                  gain=float(gain[chosen]), value=float(value[chosen]), gamma=float(gamma),
                                  posterior_entropy=posterior_entropy, ess=float(1/(weights@weights)),
                                  decision_gap=regret, information_changed=changed)

    def decide(self, history, tick):
        self.timing = {}
        self.predictor.observe(history, tick)
        self.timing.update(self.predictor.timing)
        prior = self.peaks.propose(history, tick) if self.peaks else None
        previous = history[-1, 3]
        actions = candidate_actions(previous, self.options["horizon"], self.c, proposal=prior)[0]
        if not np.isfinite(history[-1]).all() or not np.isfinite(actions).all():
            return Decision(math.nan, False)
        try:
            forecast, decision = self.evaluate(history[-1, :2], actions, previous)
        except FloatingPointError:
            return Decision(math.nan, False)
        if prior is not None:
            decision.proposal = float(actions[0, 0])
            if decision.candidate >= 0 and not math.isclose(decision.requested, decision.proposal, abs_tol=1e-7):
                if decision.information_changed:
                    decision.modification = "information"
                elif self.predictor.weights@forecast.high[:, 0] > decision.high_probability:
                    decision.modification = "high_slip_risk"
                elif self.predictor.weights@forecast.lock[:, 0] > decision.lock_probability:
                    decision.modification = "lock_risk"
                else:
                    decision.modification = "braking_objective"
        return decision

    def applied(self, decision, torque, reason):
        if self.peaks is not None:
            self.peaks.applied(decision, torque, reason)


def plant(road, c):
    """Scalar simulator for feedback. Only the caller and the Oracle see the road."""
    if road["family"] == "burckhardt":
        return lambda v, w, torque: advance_scalar(v, w, torque, road["parameters"], c)
    r, dt = c["radius"], c["physics_dt"]
    g, force, inertia = c["gravity"], c["gravity"]*c["mass"], c["inertia"]
    p = road["parameters"]
    if road["family"] == "magic":
        b, shape, height, e = p

        def coefficient(lam):
            x = b*lam
            return height*math.sin(shape*math.atan(x-e*(x-math.atan(x))))
    else:
        height, peak, alpha, tail, a, q = p
        end = math.exp(-a)

        def coefficient(lam):
            if lam <= peak:
                x = lam/peak
                return height*(1-((1-x)/(1+alpha*x))**2)
            x = (lam-peak)/(1-peak)
            return height*(tail+(1-tail)*(math.exp(-a*x**q)-end)/(1-end))

    def advance(v, w, torque):
        def derivative(x, y):
            mu = coefficient(max(0., min(1., 1-r*y/max(x, 1e-8))))
            dw = (r*force*mu-torque)/inertia
            return -g*mu, max(0., dw) if y <= 0 else dw

        distance = 0.
        for _ in range(round(c["control_dt"]/dt)):
            f, fw = derivative(v, w)
            b, bw = derivative(v+dt*f/2, w+dt*fw/2)
            d, dw = derivative(v+dt*b/2, w+dt*bw/2)
            e, ew = derivative(v+dt*d, w+dt*dw)
            new_v = max(0., v+dt*(f+2*b+2*d+e)/6)
            w = max(0., w+dt*(fw+2*bw+2*dw+ew)/6)
            distance += dt*(v+new_v)/2
            v = new_v
        return v, w, distance

    return advance


def simulate(road, c, controller, speed, *, initial_slip=0., initial_torque=0., keep_trace=False):
    from .active_rollout import GRID
    from .tire_prior_data import tire_function

    advance, dt, r = plant(road, c), c["control_dt"], c["radius"]
    w, v = speed*(1-initial_slip)/r, speed
    previous = initial_torque
    history = deque([[v, w, initial_slip, previous]]*c["history_steps"], maxlen=c["history_steps"])
    protector = Supervisor(c)
    elapsed = distance = high_time = lock_time = variation = 0.
    triggers = explorations = modified = steps = 0
    failed = ""
    trace, timings = [], {}
    for tick in range(math.ceil(c["max_time"]/dt)):
        if v <= c["stop_speed"]:
            break
        h = np.asarray(history)
        start = perf_counter()
        decision = controller.decide(h, tick)
        lam = float(h[-1, 2])
        applied, reason = protector.apply(h[-1, :2], lam, previous, decision.requested, decision.credible)
        controller.applied(decision, applied, reason)
        total = 1000*(perf_counter()-start)
        component = getattr(controller, "timing", getattr(getattr(controller, "predictor", None), "timing", {}))
        for name, value in (component | {"control": total}).items():
            timings.setdefault(name, []).append(value)
        if reason == "invalid_state":
            failed = reason
            break
        new_v, new_w, increment = advance(v, w, applied)
        duration = dt
        if new_v <= c["stop_speed"]:
            fraction = (v-c["stop_speed"])/max(v-new_v, 1e-12)
            new_w = w+fraction*(new_w-w)
            new_v, duration = c["stop_speed"], dt*fraction
            increment = (v+new_v)*duration/2
        real_explore = decision.information_changed and not reason and math.isclose(applied, decision.requested)
        if keep_trace:
            trace.append({"step": tick, "time_s": elapsed, "v": v, "omega": w, "slip": lam,
                          "force_n": c["mass"]*(v-new_v)/duration, "applied": applied, "safety": reason,
                          "executed_exploration": real_explore, **vars(decision)})
        elapsed += duration
        distance += increment
        high_time += duration*(lam >= .8)
        lock_time += duration*(w <= .5)
        variation += abs(applied-previous)
        triggers += bool(reason)
        explorations += real_explore
        modified += bool(decision.modification)
        steps += 1
        previous, v, w = applied, new_v, new_w
        history.append([v, w, max(0., min(1., 1-r*w/max(v, 1e-8))), applied])
    peak = tire_function([road])(GRID[None]).max()
    impulse = c["mass"]*(speed-v)
    stopped = bool(v <= c["stop_speed"] and not failed)
    result = {"distance_m": distance, "time_s": elapsed, "stopped": stopped,
              "failure": failed or ("" if stopped else "timeout"), "end_speed": v,
              "decel_m_s2": (speed-v)/max(elapsed, 1e-12), "impulse_ns": impulse,
              "friction_utilization": impulse/(c["mass"]*c["gravity"]*max(elapsed*peak, 1e-12)),
              "high_slip_s": high_time, "lock_s": lock_time, "torque_variation_nm": variation,
              "safety_count": triggers, "exploration_count": explorations, "modification_count": modified,
              "steps": steps}
    latency = [{"component": name, "samples": len(values), "p50_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)), "max_ms": float(np.max(values))}
               for name, values in timings.items()]
    return result, trace, latency
