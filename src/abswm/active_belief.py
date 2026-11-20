"""Finite posterior and information-subspace correction of tire contexts."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from scipy.special import logsumexp

from .active_information import entropy, likelihood_posterior, log_weights
from .active_fast import static_forecast
from .active_rollout import CurvePredictor, GRID
from .active_training import curve_grid
from .physics import TYPICAL_ROADS, friction


def normalized(weights):
    w = np.array(weights, dtype=float, copy=True)
    total = w.sum()
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any() or total <= 0:
        raise ValueError("Belief must contain finite nonnegative probability mass")
    return w/total


def effective_size(weights):
    return 1/np.square(weights).sum()


def information_step(residual, jacobian, scales, sigma, *, mode="subspace", ridge=2., threshold=2.):
    """Gauss–Newton solve in standardized coordinates with an information cutoff."""
    whitened = jacobian*scales/sigma
    information = whitened.T@whitened
    eigenvalues, directions = np.linalg.eigh(information)
    gradient = whitened.T@(residual/sigma)
    keep = eigenvalues >= threshold if mode != "full" else np.ones(len(eigenvalues), dtype=bool)
    projected = np.where(keep, (directions.T@gradient)/(np.maximum(eigenvalues, 0)+ridge), 0.)
    delta = directions@projected
    delta *= min(1., .55/max(np.linalg.norm(delta), 1e-12))
    return delta*scales, eigenvalues, int(keep.sum())


def correction(raw, scales, history, sigma, mode):
    """Update latent coordinates using a held-aside set of causal intervals."""
    evidence = np.asarray(history)
    midpoint = evidence[:, :2].mean(axis=1)
    if len(evidence) < 4 or np.ptp(midpoint) < .008:
        return raw, {"rank": 0, "before": None, "after": None, "movement": 0.}
    if mode == "selected":
        # Thin densely repeated slip regions before computing local information.
        _, idx = np.unique(np.floor(midpoint/.012).astype(int), return_index=True)
        evidence = evidence[np.sort(idx)]
    s0, s1, target = torch.as_tensor(evidence.T, dtype=raw.dtype)
    x = torch.stack((s0, (s0+s1)/2, s1), dim=0)

    def estimate(z):
        return (curve_grid(z, x)*x.new_tensor([1., 4., 1.])[:, None]).sum(dim=0)/6

    improved, ranks, before, after, traces = [], [], [], [], []
    for z in raw:
        with torch.enable_grad():
            prediction = estimate(z).detach()
            jac = torch.func.jacrev(estimate)(z).detach()
        err = (target-prediction).numpy()
        delta, values, rank = information_step(err, jac.numpy(), scales, sigma, mode=mode)
        update = torch.as_tensor(delta, dtype=z.dtype)
        best, value = z, np.mean(err**2)
        original_error = value
        for fraction in (1., .35):
            trial = (z+fraction*update).clamp(-9, 9)
            test = float((estimate(trial)-target).square().mean())
            if np.isfinite(test) and test+1e-12 < value:
                value, best = test, trial
        before.append(original_error)
        after.append(value)
        ranks.append(rank)
        traces.append(float(values[-1]))
        improved.append(best)
    solved = torch.stack(improved).detach()
    return solved, {"rank": float(np.mean(ranks)), "before": float(np.sqrt(np.mean(before))),
                    "after": float(np.sqrt(np.mean(after))), "movement": float((solved-raw).norm()),
                    "max_eigenvalue": float(np.max(traces))}


@dataclass(frozen=True)
class Evidence:
    tick: int
    source: str
    entropy_before: float
    entropy_after: float
    effective_size: float
    surprise: float
    degenerate: bool
    mismatch: bool
    refreshed: bool


class BeliefObserver(CurvePredictor):
    def __init__(self, c, model, sigma, *, reliability=.25, update=True, memory=True,
                 refinement="none", degeneracy_fraction=.29):
        super().__init__(c)
        self.model, self.sigma = model, sigma
        self.scale = sigma/np.sqrt(reliability)
        self.update, self.memory, self.refinement = update, memory, refinement
        self.degeneracy_fraction = degeneracy_fraction
        self.tick = None
        self.low_ess = self.bad_run = self.updates = 0
        self.pending = []
        self.last_event = self.last_refine = None

    def predict(self, current, actions):
        return static_forecast(current, actions, self.curves, self.c)

    @torch.no_grad()
    def encode(self, history):
        if self.model is None:
            bases = list(TYPICAL_ROADS.values())
            theta = [np.asarray(base)*[height, stiffness, tail] for base in bases
                     for height, stiffness, tail in ((.9, .9, 1.1), (1.1, 1.1, .9))]
            self.curves = np.stack([friction(GRID, p) for p in theta])
            self.weights = np.full(len(theta), 1./len(theta))
        else:
            x = torch.as_tensor(history[None], dtype=torch.float32)
            self.raw, logp = self.model(x)
            self.raw = self.raw[0].detach()
            self.weights = normalized(logp[0].exp().numpy())
            self.set_curves()
        self.initial = self.weights.copy()

    @torch.no_grad()
    def set_curves(self):
        self.curves = curve_grid(self.raw, torch.as_tensor(GRID, dtype=self.raw.dtype)).numpy()

    def assimilate(self, predicted, measured, tick):
        before = float(entropy(self.weights))
        residual = measured-predicted
        surprise = float(np.min(np.abs(residual))/self.sigma)
        log_likelihood = -.5*(residual/self.scale)**2
        marginal = -float(logsumexp(log_weights(self.weights)+log_likelihood))
        self.bad_run = self.bad_run+1 if surprise > 3. or marginal > 5. else max(0, self.bad_run-1)
        mismatch = self.bad_run >= 3
        if not mismatch:
            self.weights = normalized(likelihood_posterior(self.weights, residual, self.scale))
        degenerate = effective_size(self.weights) < max(1.05, self.degeneracy_fraction*len(self.weights))
        self.low_ess = self.low_ess+1 if degenerate else 0
        refreshed = mismatch or self.low_ess >= 3
        if refreshed:
            recovery = .35 if mismatch else .025
            self.weights = normalized((1-recovery)*self.weights+recovery*self.initial)
            self.low_ess = 0
        self.trusted = not mismatch
        self.updates += 1
        return Evidence(tick, "bayesian", before, float(entropy(self.weights)),
                        float(effective_size(self.weights)), surprise, degenerate, mismatch, refreshed)

    def observe(self, history, tick):
        latest = np.asarray(history[-1])
        self.timing = {}
        if self.tick == tick:
            if not np.array_equal(latest, self.previous, equal_nan=True):
                raise ValueError("Conflicting measurements for the same observation tick")
            return
        self.last_event = self.last_refine = None
        if not np.isfinite(latest).all():
            self.trusted = False
            return
        if self.tick is None or not self.memory:
            start = perf_counter()
            self.encode(history)
            self.timing["encoder_decoder"] = 1000*(perf_counter()-start)
        else:
            if tick != self.tick+1:
                raise ValueError("Observer requires successive real observations")
            previous = self.previous
            c = self.c
            mu = (previous[0]-latest[0])/(c["control_dt"]*c["gravity"])
            usable = (.002 <= (previous[2]+latest[2])/2 and latest[0] > c["stop_speed"]+.05
                      and -.002 <= mu <= 2.)
            reserved = self.refinement != "none" and self.model is not None and (not self.update or tick % 2 == 0)
            if usable and reserved:
                self.pending.append((previous[2], latest[2], mu))
                if len(self.pending) == 12:
                    start = perf_counter()
                    self.raw, self.last_refine = correction(self.raw, self.model.latent_scale.numpy(),
                                                            self.pending, self.scale, self.refinement)
                    self.pending.clear()
                    self.set_curves()
                    self.timing["refinement"] = 1000*(perf_counter()-start)
            elif usable and self.update:
                start = perf_counter()
                predicted = self.predict(previous[:2], np.array([[latest[3]]]))
                self.last_event = self.assimilate(predicted.interval_mu[:, 0, 0], mu, tick)
                self.timing["belief_update"] = 1000*(perf_counter()-start)
        self.tick, self.previous = tick, latest.copy()


def make_observer(c, model, policy, **options):
    return BeliefObserver(c, model, policy["observation_scale"],
                          reliability=policy["study"]["likelihood_reliability"],
                          degeneracy_fraction=policy["study"]["min_effective_fraction"], **options)
