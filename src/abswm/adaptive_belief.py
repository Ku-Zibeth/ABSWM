"""Causal innovation, particle revision, and a continuous tire proposal."""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from scipy.special import logsumexp
from torch import nn

from .active_belief import correction, effective_size, normalized
from .active_fast import static_forecast
from .active_information import entropy, log_weights
from .active_rollout import GRID
from .active_training import curve_grid
from .physics import TYPICAL_ROADS, friction
from .tire_prior import friction_query


class ContinuousTire(nn.Module):
    """Conditional diagonal Gaussian mixture in the six structured tire logits."""
    def __init__(self, prototype, width=.28):
        super().__init__()
        self.prototype = copy.deepcopy(prototype)
        self.width = width
        self.log_adjustment = nn.Parameter(torch.zeros_like(prototype.centers))

    @property
    def latent_scale(self):
        return self.prototype.latent_scale

    def distribution(self, history):
        mean, p = self.prototype(history)
        sigma = self.width*self.latent_scale*self.log_adjustment.clamp(-2., 1.).exp()
        return mean, sigma, p

    def forward(self, history, deviation=None):
        mean, sigma, p = self.distribution(history)
        if deviation is None:
            if self.training:
                deviation = torch.randn_like(mean)
            else:
                indices = torch.arange(mean.shape[-2]*mean.shape[-1], device=mean.device).reshape(mean.shape[-2:])
                deviation = torch.where((indices^(indices//3)) % 2 == 0, 1., -1.).to(mean)
        offset = deviation*sigma
        raw = torch.cat((mean+offset, mean-offset), dim=1)
        return raw, torch.cat((p, p), dim=1)-np.log(2.)


@dataclass(frozen=True)
class TireBank:
    curves: np.ndarray
    weights: np.ndarray
    raw: torch.Tensor | None
    identities: tuple


class ParticleSource:
    def __init__(self, kind, model=None, seed=0):
        self.kind, self.model, self.seed = kind, model, seed

    @torch.no_grad()
    def make(self, history, generation):
        x = torch.as_tensor(history[None], dtype=torch.float32)
        grid = torch.as_tensor(GRID, dtype=x.dtype)
        raw = None
        if self.kind == "ensemble":
            curves = np.stack([friction_query(net, net.encode(x), grid[None])[0].numpy() for net in self.model])
            weights = np.full(len(curves), 1./len(curves))
        elif self.kind == "library":
            params = [np.asarray(p)*q for p in TYPICAL_ROADS.values()
                      for q in ([.9, .88, 1.1], [1., 1., 1.], [1.13, 1.16, .9])]
            curves = np.stack([friction(GRID, p) for p in params])
            weights = np.full(len(curves), 1./len(curves))
        else:
            if self.kind == "continuous" and generation:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, generation]))
                q = rng.normal(size=self.model.prototype.centers.shape)
                # Use a reproducible antithetic proposal, with bounded tail support.
                deviation = torch.as_tensor(np.clip(q, -2.5, 2.5), dtype=x.dtype)
                raw, logits = self.model(x, deviation)
            else:
                raw, logits = self.model(x)
            raw = raw[0].detach().clone()
            weights = normalized(logits[0].exp().numpy())
            curves = curve_grid(raw, grid).numpy()
        return TireBank(curves, weights, raw, tuple((generation, k) for k in range(len(weights))))


def update_weights(weights, means, measured, scale):
    log_likelihood = -.5*((np.asarray(means)-measured)/scale)**2
    joint = log_weights(weights)+log_likelihood
    log_mass = logsumexp(joint)
    return np.exp(joint-log_mass), float(-log_mass)


class Innovation:
    """Hysteretic accumulation of predictive surprise in normalized log units."""
    def __init__(self, threshold, drift):
        self.threshold, self.drift = threshold, drift
        self.score, self.detected = 0., False

    def push(self, surprise):
        self.score = min(4*self.threshold, max(0., .82*self.score+min(4*self.threshold, surprise)-self.drift))
        was_active = self.detected
        if self.detected:
            self.detected = self.score > .3*self.threshold
        else:
            self.detected = self.score >= self.threshold
        return self.detected and not was_active

    def reset(self):
        self.score, self.detected = 0., False


class AdaptiveObserver:
    """One real interval per tick. All planning uses a detached bank snapshot."""
    def __init__(self, c, d, source, *, revision="R2", refinement="none", scale=None, threshold=None):
        self.c, self.d, self.source = c, d, source
        self.scale = float(d["innovation_scales"][1] if scale is None else scale)
        self.detector = Innovation(d["innovation_thresholds"][1] if threshold is None else threshold,
                                   d["innovation_drift"])
        self.revision, self.refinement = revision, refinement
        self.tick, self.previous, self.prepared = None, None, None
        self.history = deque(maxlen=d["revision_window"])
        self.pending = []
        self.generation = 0
        self.last_revision = -d["revision_cooldown"]
        self.trusted = True
        self.timing, self.last = {}, {}

    def snapshot(self):
        return TireBank(self.curves.copy(), self.weights.copy(), None, self.identities)

    def set_bank(self, bank):
        self.curves, self.weights, self.raw, self.identities = bank.curves, bank.weights, bank.raw, bank.identities
        self.anchor_weights = bank.weights.copy()

    def predict(self, current, actions):
        return static_forecast(current, actions, self.curves, self.c)

    def predict_executed(self, torque, tick):
        values = self.predict(self.previous[:2], np.array([[torque]])).interval_mu[:, 0, 0]
        self.prepared = (tick, torque, values)

    def _regenerate(self, tick):
        screen = list(self.history)[-self.d["revision_screen"]:]
        prefix = list(self.history)[:-len(screen)]
        # An encoded prefix and its held-out screening intervals are disjoint.
        cutoff = np.array([prefix[0][0], *(row[1] for row in prefix)])
        needed = self.c["history_steps"]
        context = np.concatenate((np.repeat(cutoff[:1], max(0, needed-len(cutoff)), axis=0), cutoff))[-needed:]
        new = self.source.make(context, self.generation+1)
        curves, weights = new.curves.copy(), new.weights.copy()
        raw = None if new.raw is None else new.raw.clone()
        ids = list(new.identities)
        if len(weights) > 1:
            # One old anchor may survive. Its old posterior mass is discarded.
            keep, slot = int(np.argmax(self.weights)), int(np.argmin(weights))
            curves[slot], ids[slot] = self.curves[keep], self.identities[keep]
            if raw is not None:
                raw[slot] = self.raw[keep]
            weights[slot] = weights.mean()
            weights = normalized(weights)
        predictions = [static_forecast(a[:2], np.array([[b[3]]]), curves, self.c).interval_mu[:, 0, 0] for a, b, _ in screen]
        residual = np.asarray(predictions)-np.asarray([row[2] for row in screen])[:, None]
        # Temper the block evidence to accommodate correlated interval residuals.
        log_joint = log_weights(weights)-.5*np.mean((residual/self.scale)**2, axis=0)
        screened = np.exp(log_joint-logsumexp(log_joint))
        prior_gap = np.ptp(curves, axis=0).mean()
        self.set_bank(TireBank(curves, screened, raw, tuple(ids)))
        self.anchor_weights = weights
        self.generation += 1
        self.last_revision = tick
        self.pending.clear()
        self.history.clear()
        self.detector.reset()
        after = float(np.min(np.sqrt((residual**2).mean(axis=0)))/self.scale)
        return after, float(prior_gap), len(screen)

    def observe(self, history, tick, *, evidence=True):
        x = np.asarray(history[-1], dtype=float)
        self.timing, self.last = {}, {}
        if not np.isfinite(x).all():
            self.trusted = False
            return
        if tick == self.tick:
            if not np.array_equal(x, self.previous):
                raise ValueError("Conflicting observations at one tick")
            return
        if self.tick is None or self.revision == "window" and evidence:
            start = perf_counter()
            self.set_bank(self.source.make(np.asarray(history), tick if self.revision == "window" else 0))
            self.timing["encode"] = 1000*(perf_counter()-start)
            self.tick, self.previous = tick, x.copy()
            return
        if tick != self.tick+1:
            raise ValueError("Real observation ticks must be consecutive")
        start = perf_counter()
        previous = self.previous
        measured = (previous[0]-x[0])/(self.c["control_dt"]*self.c["gravity"])
        usable = ((x[2]+previous[2])/2 > .002 and x[0] > self.c["stop_speed"]+.05
                  and -.002 <= measured <= 2.1)
        if not evidence:
            self.pending.clear()
            self.history.clear()
        elif usable:
            cached = self.prepared
            if cached is not None and cached[0] == tick and np.isclose(cached[1], x[3]):
                means = cached[2]
            else:
                means = self.predict(previous[:2], np.array([[x[3]]])).interval_mu[:, 0, 0]
            self.history.append((previous.copy(), x.copy(), measured))
            # A measurement is assessed once against the predictive distribution from t-1.
            innovation_scale = self.scale
            if self.revision == "R1" and tick-self.last_revision < self.d["revision_cooldown"]:
                innovation_scale *= 1.5
            updated, surprise = update_weights(self.weights, means, measured, innovation_scale)
            prior_mean = float(means@self.weights)
            trigger = self.detector.push(surprise)
            active = self.detector.detected
            closest = float(np.min(np.abs(measured-means))/innovation_scale)
            before, bank_spread = float(entropy(self.weights)), float(np.std(self.curves, axis=0).mean())
            allowed = tick-self.last_revision >= self.d["revision_cooldown"]
            revise, screen_count = "", 0
            after = closest
            if active and allowed and self.revision == "R2" and len(self.history) > self.d["revision_screen"]+2:
                after, bank_spread, screen_count = self._regenerate(tick)
                revise = "regenerate"
            else:
                reserve = self.refinement != "none" and self.raw is not None and tick % 2 == 0 and not active
                if active and self.revision == "R1" and allowed:
                    fade = self.d["forgetting"]
                    prior = normalized((1-fade)*self.weights+fade*self.anchor_weights)
                    self.weights, _ = update_weights(prior, means, measured, innovation_scale*1.5)
                    self.last_revision = tick
                    self.pending.clear()
                    self.detector.reset()
                    revise = "forget"
                elif reserve:
                    self.pending.append((previous[2], x[2], measured))
                    if len(self.pending) == 12:
                        mark = perf_counter()
                        self.raw, fit = correction(self.raw, self.source.model.latent_scale.numpy(), self.pending,
                                                   self.scale, self.refinement)
                        self.curves = curve_grid(self.raw, torch.as_tensor(GRID, dtype=self.raw.dtype)).numpy()
                        self.pending.clear()
                        self.timing["refinement"] = 1000*(perf_counter()-mark)
                        self.last = {f"refine_{k}": v for k, v in fit.items()}
                else:
                    self.weights = normalized(updated)
                    if active:
                        self.pending.clear()
            self.trusted = not (active and after > self.d["guard_sigma"])
            self.last.update({"innovation": surprise, "cusum": self.detector.score,
                              "detection": trigger, "mismatch": active, "closest_residual": closest,
                              "censored_residual": after, "revision": revise, "generation": self.generation,
                              "screen_intervals": screen_count, "entropy_before": before,
                              "entropy_after": float(entropy(self.weights)), "ess": effective_size(self.weights),
                              "curve_disagreement": bank_spread, "observed_mu": measured,
                              "predictive_mu_mean": prior_mean})
            self.timing["belief_update"] = 1000*(perf_counter()-start)
        else:
            self.history.clear()
        self.tick, self.previous, self.prepared = tick, x.copy(), None
