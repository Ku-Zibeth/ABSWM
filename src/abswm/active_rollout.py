"""Shared counterfactual actions, rollouts and predictive braking objective."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from .controllers import LeastSquaresPeak
from .physics import TYPICAL_ROADS, friction, slip
from .recurrent_model import Posterior
from .tire_prior import friction_query
from .tire_prior_data import tire_function


ACTION_NAMES = ("hold", "fast_release", "slow_release", "slow_increase", "fast_increase",
                "increase_hold", "increase_release", "pulse", "multistage")
GRID = np.linspace(0., 1., 257)


def candidate_actions(previous, horizon, c, *, proposal=None, random=None):
    """Same move blocks for all MPC predictors, with an optional local search."""
    previous = np.atleast_1d(previous).astype(float)
    delta = c["torque_rate"] * c["control_dt"]
    t = np.arange(horizon)
    short = max(1, horizon // 4)
    middle = max(1, horizon // 2)
    rate = np.stack((t*0, t*0-1, t*0-.25, t*0+.25, t*0+1, .8*(t < short),
                     np.where(t < middle, .6, -.85),
                     np.where(t < short, 1., np.where(t < 2*short, -1., 0.)),
                     np.where(t < short, -.5, np.where(t < 2*short, .85, -.25))))
    if proposal is not None:
        impulses = np.clip((proposal-previous[0])/delta + np.array([0., -.5, .5, -1., 1.]), -1, 1)
        rate = np.concatenate((impulses[:, None]*(t < short), rate[:3]), axis=0)
    rates = np.broadcast_to(rate, (len(previous), *rate.shape))
    if random is not None:
        extra = random.uniform(-1, 1, (len(previous), 2, 4))
        rates = np.concatenate((rates, extra[..., np.minimum(t*4//horizon, 3)]), axis=1)
    result = np.empty_like(rates)
    torque = previous[:, None]
    for step, move in enumerate(np.moveaxis(rates, -1, 0)):
        torque = np.clip(torque + delta*move, 0, c["torque_max"])
        result[..., step] = torque
    return result


def grid_coefficient(curves):
    """Vectorized linear queries. Curve rows broadcast over the candidate axes."""
    curves = np.asarray(curves)
    keys = np.arange(len(curves))[:, None]

    def coefficient(lam):
        index = lam*(curves.shape[1]-1)
        lower = np.minimum(index.astype(np.intp), curves.shape[1]-2)
        left, right = curves[keys, lower], curves[keys, lower+1]
        return left + (index-lower)*(right-left)

    return coefficient


@dataclass(frozen=True)
class Forecast:
    # The leading dimensions are hypothesis and action for controller calls.
    state: np.ndarray
    slip: np.ndarray
    mu: np.ndarray
    interval_mu: np.ndarray
    max_slip: np.ndarray
    min_wheel: np.ndarray
    dt: float
    load: float

    @property
    def force(self):
        return self.mu*self.load

    @property
    def impulse(self):
        return self.interval_mu.sum(axis=-1)*self.dt*self.load

    @property
    def high(self):
        return self.max_slip.max(axis=-1) >= .8

    @property
    def lock(self):
        return self.min_wheel.min(axis=-1) <= .5


def integrate(initial, actions, coefficient, c, *, rk4=False):
    """Advance using the prescribed 1 ms step and retain within-step event extrema."""
    actions = np.asarray(actions, dtype=float)
    shape = actions.shape[:-1]
    v = np.broadcast_to(np.asarray(initial)[..., 0], shape).copy()
    w = np.broadcast_to(np.asarray(initial)[..., 1], shape).copy()
    dt, g, r = c["physics_dt"], c["gravity"], c["radius"]
    scale = r*c["mass"]*g/c["inertia"]
    steps = round(c["control_dt"]/dt)
    state = np.empty((*actions.shape, 2))
    mu, interval, lam, max_lam, min_w = (np.empty_like(actions) for _ in range(5))

    def get_slip(vv, ww):
        return np.clip(1-r*ww/np.maximum(vv, 1e-8), 0, 1)

    def derivative(vv, ww, torque):
        value = coefficient(get_slip(vv, ww))
        dw = scale*value-torque
        return -g*value, np.where(ww <= 0, np.maximum(dw, 0), dw)

    slip_now = get_slip(v, w)
    for step, torque in enumerate(np.moveaxis(actions, -1, 0)):
        torque = np.clip(torque, 0, c["torque_max"])/c["inertia"]
        start_v = v.copy()
        high, low = slip_now, w
        for _ in range(steps):
            if rk4:
                a, aw = derivative(v, w, torque)
                b, bw = derivative(v+dt*a/2, w+dt*aw/2, torque)
                d, dw = derivative(v+dt*b/2, w+dt*bw/2, torque)
                e, ew = derivative(v+dt*d, w+dt*dw, torque)
                v = np.maximum(v+dt*(a+2*b+2*d+e)/6, 0)
                w = np.maximum(w+dt*(aw+2*bw+2*dw+ew)/6, 0)
            else:
                value = coefficient(slip_now)
                v = np.maximum(v-dt*g*value, 0)
                w = np.maximum(w+dt*(scale*value-torque), 0)
            slip_now = get_slip(v, w)
            high, low = np.maximum(high, slip_now), np.minimum(low, w)
        state[..., step, 0], state[..., step, 1] = v, w
        lam[..., step] = slip_now
        mu[..., step] = coefficient(slip_now)
        interval[..., step] = (start_v-v)/(g*c["control_dt"])
        max_lam[..., step], min_w[..., step] = high, low
    return Forecast(state, lam, mu, interval, max_lam, min_w, c["control_dt"], c["mass"]*g)


def curve_forecast(current, actions, curves, c):
    curves = np.atleast_2d(curves)
    actions = np.broadcast_to(actions, (len(curves), *actions.shape))
    return integrate(current, actions, grid_coefficient(curves), c)


def true_forecast(roads, initial, actions, c):
    """Offline labels and Oracle only. One road per shared initial history."""
    return integrate(np.asarray(initial)[:, None], actions, tire_function(roads), c, rk4=True)


class CurvePredictor:
    def __init__(self, c):
        self.c = c
        self.weights = np.ones(1)
        self.trusted = True
        self.timing = {}

    def predict(self, current, actions):
        return curve_forecast(current, actions, self.curves, self.c)


class NeuralCurvePredictor(CurvePredictor):
    def __init__(self, c, models):
        super().__init__(c)
        self.models = models
        self.weights = np.full(len(models), 1/len(models))

    @torch.no_grad()
    def observe(self, history, tick):
        start = perf_counter()
        x = torch.as_tensor(history[None], dtype=torch.float32)
        grid = torch.as_tensor(GRID[None], dtype=x.dtype)
        self.curves = np.stack([friction_query(model, model.encode(x), grid)[0].numpy()
                                for model in self.models])
        self.timing = {"encoder_decoder": 1000*(perf_counter()-start)}


class NLSPredictor(CurvePredictor):
    def __init__(self, c):
        super().__init__(c)
        self.estimator = LeastSquaresPeak(c)
        self.curves = friction(GRID, TYPICAL_ROADS["wet"])[None]
        self.reference = c["fixed_reference"]

    def observe(self, history, tick):
        if tick >= self.c["history_steps"] and tick % self.c["reference_period"] == 0:
            start = perf_counter()
            self.reference = self.estimator(history)
            if self.estimator.theta is not None:
                self.curves = friction(GRID, self.estimator.theta)[None]
            self.timing = {"nls": 1000*(perf_counter()-start)}
        else:
            self.timing = {}


class RecurrentPredictor(CurvePredictor):
    """Frozen original recurrent observer and imagined transition."""
    def __init__(self, c, model):
        super().__init__(c)
        self.model, self.memory, self.tick = model, None, None

    @torch.inference_mode()
    def observe(self, history, tick):
        if tick == self.tick:
            return
        start = perf_counter()
        rows = history if self.tick is None else history[-(tick-self.tick):]
        self.memory = self.model.observe(torch.as_tensor(rows[None], dtype=torch.float32), self.memory)
        self.tick = tick
        self.timing = {"encoder_decoder": 1000*(perf_counter()-start)}

    @torch.inference_mode()
    def predict(self, current, actions):
        k = len(actions)
        original = self.memory.last_observation
        latest = original.clone()
        latest[:, :2] = torch.as_tensor(current, dtype=latest.dtype)
        memory = Posterior(self.memory.z.expand(k, -1), latest.expand(k, -1))
        predicted, predicted_mu, _ = self.model.predict(memory, torch.as_tensor(actions, dtype=latest.dtype))
        x = predicted.numpy()[None]
        lams = slip(x, self.c["radius"])
        before = np.concatenate((np.broadcast_to(current[0], (*x.shape[:2], 1)), x[..., :-1, 0]), axis=-1)
        interval = (before-x[..., 0])/(self.c["gravity"]*self.c["control_dt"])
        return Forecast(x, lams, predicted_mu.numpy()[None], interval,
                        np.maximum(lams, np.concatenate((np.full((*x.shape[:2], 1), slip(current, self.c["radius"])),
                                                         lams[..., :-1]), axis=-1)),
                        np.minimum(x[..., 1], np.concatenate((np.full((*x.shape[:2], 1), current[1]),
                                                               x[..., :-1, 1]), axis=-1)),
                        self.c["control_dt"], self.c["mass"]*self.c["gravity"])


class OraclePredictor(CurvePredictor):
    def __init__(self, c, road):
        super().__init__(c)
        self.coefficient = tire_function([road])

    def observe(self, history, tick):
        pass

    def predict(self, current, actions):
        return integrate(current, actions[None], self.coefficient, self.c, rk4=True)


def utilities(result, actions, previous, options):
    """Dimensionless benefit and penalties, scaled by 1 g over the horizon."""
    effort = np.abs(np.diff(actions, axis=-1, prepend=np.full((*actions.shape[:-1], 1), previous)))
    effort = effort.mean(axis=-1)/options["slew_nm"]
    braking = result.interval_mu.mean(axis=-1)
    high = (result.max_slip >= .8).mean(axis=-1)
    lock = (result.min_wheel <= .5).mean(axis=-1)
    terminal = ((result.slip[..., -1]-.45).clip(0)/.55)**2
    score = (braking - options["high_penalty"]*high - options["lock_penalty"]*lock
             - options["smooth_penalty"]*effort - options["terminal_penalty"]*terminal)
    return score


def lower_cvar(scores, weights, level):
    order = np.argsort(scores, axis=0)
    values = np.take_along_axis(scores, order, axis=0)
    probability = np.asarray(weights)[order]
    before = np.cumsum(probability, axis=0)-probability
    mass = np.minimum(np.maximum(level-before, 0), probability)
    return (values*mass).sum(axis=0)/level
