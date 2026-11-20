"""Matched finite-feedback planning and safely interruptible action segments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .active_control import Decision
from .active_fast import static_forecast
from .active_information import entropy, information_values
from .active_rollout import candidate_actions, utilities


@dataclass(frozen=True)
class ObservationBranch:
    mass: float
    state: np.ndarray
    posterior: np.ndarray


def observation_branches(states, weights, c, d, scale, length):
    """Cubature samples grouped by observable (speed, wheel speed) only."""
    effective = length/(1+d["observation_correlation"]*(length-1))
    sigma_v = c["gravity"]*c["control_dt"]*length*scale/np.sqrt(effective)
    sigma_w = .75+2.*scale*np.sqrt(length)
    correlation = -.45  # Faster predicted deceleration and wheel acceleration are coupled.
    spread = np.array([[sigma_v, 0.], [correlation*sigma_w, sigma_w*np.sqrt(1-correlation**2)]])
    nodes = np.array([[0., 0.], [2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
    node_mass = np.array([.5, .125, .125, .125, .125])
    values = np.maximum(states[:, None, :]+nodes@spread.T, 0.)
    quantized = np.rint((values-states.mean(axis=0))/[max(sigma_v, 1e-10), sigma_w]).astype(int)
    quantized = quantized.reshape(-1, 2)
    labels, assignment = np.unique(quantized, axis=0, return_inverse=True)
    # Sort by a deterministic scalar projection, then merge adjacent observation cells.
    # Identical observation cells are never split by the hidden hypothesis index.
    order = np.argsort(labels[:, 0]+.27*labels[:, 1], kind="stable")
    groups = np.array_split(order, min(len(order), d["observation_branches"]))
    k = len(weights)
    masses = (weights[:, None]*node_mass).reshape(-1)
    outcome = []
    for group in groups:
        chosen = np.isin(assignment, group)
        component = np.where(chosen.reshape(k, -1), masses.reshape(k, -1), 0.).sum(axis=1)
        mass = component.sum()
        if mass > 1e-15:
            observed = masses[chosen]@values.reshape(-1, 2)[chosen]/mass
            outcome.append(ObservationBranch(float(mass), observed, component/mass))
    return outcome


def _unique_actions(actions):
    _, order = np.unique(actions, axis=0, return_index=True)
    return actions[np.sort(order)]


def _take_actions(previous, horizon, c, indices, proposal=None):
    actions = candidate_actions(previous, horizon, c, proposal=proposal)[0]
    chosen = indices if proposal is None else range(len(actions))
    return _unique_actions(actions[list(chosen)])


@dataclass
class Plan:
    segments: np.ndarray
    objective: np.ndarray
    frozen: np.ndarray
    selected: int
    matched: int
    high: np.ndarray
    lock: np.ndarray
    impulse: np.ndarray
    gain: np.ndarray
    value: np.ndarray
    branches: list


def feedback_plan(current, previous, bank, c, d, options, length, mode, scale, *, proposal=None, detail=False,
                  risk_transform=None):
    """Score first segments across a common lattice and a common observation partition."""
    h = d["total_horizon"]
    remainder = h-length
    if not 0 < length < h:
        raise ValueError("The commitment must leave time for replanning")
    weights, curves = bank.weights, bank.curves
    if not np.isfinite(curves).all() or not np.isfinite(weights).all():
        raise FloatingPointError("Nonfinite model bank")
    first = _take_actions(previous, length, c, d["first_actions"], proposal)
    head = static_forecast(current, first, curves, c)
    high, lock = weights@head.high, weights@head.lock
    if risk_transform is not None:
        gate_high, gate_lock = risk_transform(high), risk_transform(lock)
    else:
        gate_high, gate_lock = high, lock
    feasible = (gate_high <= options["explore_high_limit"]) & (gate_lock <= options["explore_lock_limit"])
    if not feasible.any():
        feasible[:] = True
    front_options = options | {"terminal_penalty": 0.}
    tail_options = options | {"terminal_penalty": options["terminal_penalty"]*h/remainder}
    prefix = (weights@utilities(head, first, previous, front_options))*length/h
    following = [_take_actions(u, remainder, c, d["second_actions"]) for u in first[:, -1]]
    active, frozen = prefix.copy(), prefix.copy()
    gain, value = np.zeros(len(first)), np.zeros(len(first))
    rows = []
    if mode in {"feedback", "frozen_feedback"}:
        for i, controls in enumerate(following):
            branches = observation_branches(head.state[:, i, -1], weights, c, d, scale, length)
            after_entropy = 0.
            for j, branch in enumerate(branches):
                result = static_forecast(branch.state, controls, curves, c)
                outcomes = utilities(result, controls, first[i, -1], tail_options)*remainder/h
                learned_score = branch.posterior@outcomes
                adaptive = int(np.argmax(learned_score))
                fixed = int(np.argmax(weights@outcomes))
                # Evaluate both decisions under the same joint observation/hypothesis law.
                active[i] += branch.mass*learned_score[adaptive]
                frozen[i] += branch.mass*learned_score[fixed]
                after_entropy += branch.mass*float(entropy(branch.posterior))
                if detail:
                    rows.append({"first": i, "branch": j, "mass": branch.mass,
                                 "velocity": float(branch.state[0]), "wheel": float(branch.state[1]),
                                 "posterior": branch.posterior.tolist(), "learned_next": adaptive,
                                 "frozen_next": fixed, "realization_value": float(learned_score[adaptive]-learned_score[fixed])})
            gain[i] = max(0., float(entropy(weights))-after_entropy)
        value = np.maximum(active-frozen, 0.)
    else:
        full = np.concatenate([np.column_stack((np.broadcast_to(u, (len(a), length)), a))
                               for u, a in zip(first, following)], axis=0)
        complete = static_forecast(current, full, curves, c)
        full_score = utilities(complete, full, previous, options)
        partitions = np.r_[0, np.cumsum([len(a) for a in following])]
        for i, (lo, hi) in enumerate(zip(partitions[:-1], partitions[1:])):
            frozen[i] = np.max(weights@full_score[:, lo:hi])
        active = frozen.copy()
        if mode in {"entropy", "task"}:
            raw = head.interval_mu.mean(axis=-1)
            effective = length/(1+d["observation_correlation"]*(length-1))
            gain, value = information_values(raw, weights, full_score, scale/np.sqrt(effective))
            signal = gain/max(math.log(len(weights)), 1e-10) if mode == "entropy" else value
            active += d["exploration_gamma"]*signal
    objective = frozen if mode in {"passive", "frozen_feedback"} else active
    if not np.isfinite(objective).all():
        raise FloatingPointError("Nonfinite candidate utility")
    best = int(np.argmax(np.where(feasible, objective, -np.inf)))
    matched = int(np.argmax(np.where(feasible, frozen, -np.inf)))
    return Plan(first, objective, frozen, best, matched, high, lock, weights@head.impulse, gain, value, rows)


class StrongPeak:
    """Causal slip-force gradient search with build-up, a dead zone, and recovery."""
    def __init__(self, c, settings):
        self.c, self.settings = c, settings
        self.direction, self.vote = 1, 0
        self.gradient = 0.
        self.stall = 0

    def propose(self, history, tick):
        current = history[-1]
        lam, previous = float(current[2]), float(current[3])
        p = self.settings
        increment = p["increment"]
        if tick < p["build_steps"]:
            return previous+increment if lam < .25 else previous-increment
        self.stall = self.stall+1 if lam < .018 and previous < .12*self.c["torque_max"] else 0
        if self.stall >= p["period"]:
            self.direction, self.vote = 1, 0
            return previous+increment
        if tick % p["period"] == 0:
            rows = history[-2*p["period"]-1:]
            mu = -np.diff(rows[:, 0])/(self.c["gravity"]*self.c["control_dt"])
            slip = .5*(rows[:-1, 2]+rows[1:, 2])
            centered = slip-slip.mean()
            if np.ptp(slip) >= .004:
                gradient = float(centered@(mu-mu.mean())/max(centered@centered, 1e-12))
                self.gradient = .5*self.gradient+.5*np.clip(gradient, -8, 8)
                sign = int(np.sign(self.gradient)) if abs(self.gradient) > p["deadzone"] else 0
                self.vote = self.vote+sign if sign*self.vote >= 0 else sign
                if abs(self.vote) >= p["confirm"]:
                    self.direction = int(np.sign(self.vote))
                    self.vote = 0
        if lam > .55:
            self.direction = -1
            return previous-1.5*increment
        amplitude = increment*float(np.clip(abs(self.gradient)/2.5, .20, 1.))
        if previous >= self.c["torque_max"]-1:
            self.direction = -1
        return previous+amplitude*self.direction

    def decide(self, history, tick):
        return Decision(self.propose(history, tick))

    def applied(self, decision, torque, reason):
        if reason:
            self.gradient, self.vote = 0., 0
            self.direction = -1


class ProtectionOnly:
    """Default action is open-loop moderate build-up under the shared supervisor."""
    def __init__(self, c):
        self.increment = .25*c["torque_rate"]*c["control_dt"]

    def decide(self, history, tick):
        return Decision(float(history[-1, 3]+self.increment))

    def applied(self, decision, torque, reason):
        pass


@dataclass
class SegmentDecision(Decision):
    segment: int = -1
    segment_offset: int = 0
    planned_steps: int = 0
    planned_exploration: bool = False
    matched_action: float = math.nan
    planned_impulse: float = math.nan
    planned_value: float = math.nan
    ranked_candidates: tuple = ()


class CommittedController:
    def __init__(self, c, d, options, observer, length, mode, *, peak=None, risk_transform=None):
        self.c, self.d, self.options, self.predictor = c, d, options, observer
        self.length, self.mode, self.peaks = length, mode, peak
        self.risk_transform = risk_transform
        self.plan, self.segment, self.offset = None, -1, 0
        self.real_evidence = True
        self.timing = {}
        self.last_plan = None
        self.forced_prefix = None

    def decide(self, history, tick):
        self.timing = {}
        if not np.isfinite(history[-1]).all():
            return SegmentDecision(math.nan, False)
        self.predictor.observe(history, tick, evidence=self.real_evidence)
        self.timing.update(self.predictor.timing)
        proposal = None if self.peaks is None else self.peaks.propose(history, tick)
        if self.plan is None:
            mark = perf_counter()
            bank = self.predictor.snapshot()
            try:
                plan = feedback_plan(history[-1, :2], history[-1, 3], bank, self.c, self.d, self.options,
                                     self.length, self.mode, self.predictor.scale, proposal=proposal,
                                     risk_transform=self.risk_transform)
            except FloatingPointError:
                return SegmentDecision(math.nan, False)
            chosen, matched = plan.selected, plan.matched
            if self.forced_prefix is not None:
                chosen = int(np.argmin(np.max(np.abs(plan.segments-self.forced_prefix), axis=1)))
                if not np.allclose(plan.segments[chosen], self.forced_prefix):
                    raise ValueError("The forced segment must belong to the common candidate set")
                self.forced_prefix = None
            self.plan, self.last_plan = (plan, chosen, matched), plan
            self.segment, self.offset = self.segment+1, 0
            self.timing["plan"] = 1000*(perf_counter()-mark)
        plan, chosen, matched = self.plan
        selected, baseline = plan.segments[chosen], plan.segments[matched]
        planned = not np.allclose(selected, baseline, atol=1e-7, rtol=0)
        selected_torque = float(selected[self.offset])
        compared = float(baseline[self.offset])
        modified = "feedback_objective" if proposal is not None and not math.isclose(proposal, selected_torque) else ""
        return SegmentDecision(selected_torque, self.predictor.trusted, chosen, matched,
                               proposal=math.nan if proposal is None else proposal,
                               high_probability=float(plan.high[chosen]), lock_probability=float(plan.lock[chosen]),
                               gain=float(plan.gain[chosen]), value=float(plan.value[chosen]),
                               posterior_entropy=float(entropy(self.predictor.weights)),
                               ess=float(1/(self.predictor.weights@self.predictor.weights)),
                               information_changed=planned and not math.isclose(selected_torque, compared, abs_tol=1e-7),
                               modification=modified, segment=self.segment, segment_offset=self.offset,
                               planned_steps=len(selected), planned_exploration=planned, matched_action=compared,
                               planned_impulse=float(plan.impulse[chosen]), planned_value=float(plan.objective[chosen]),
                               ranked_candidates=tuple(map(int, np.argsort(-plan.objective, kind="stable"))))

    def applied(self, decision, torque, reason):
        if self.peaks is not None:
            self.peaks.applied(decision, torque, reason)
        self.offset += 1
        if reason or self.offset >= self.length or not math.isclose(torque, decision.requested, abs_tol=1e-7):
            self.plan = None
