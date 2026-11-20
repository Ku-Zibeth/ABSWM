"""Optional compiled integration of the same single-candidate dynamics."""
from __future__ import annotations

import math

import numpy as np

from .active_rollout import Forecast, curve_forecast

try:
    from numba import njit
except ImportError:
    njit = None

FAMILIES = {"burckhardt": 1, "magic": 2, "rational": 3}


if njit is not None:
    @njit(cache=True, inline="always")
    def _coefficient(lam, profile, kind):
        if kind == 0:
            index = lam*(len(profile)-1)
            lower = min(int(index), len(profile)-2)
            return profile[lower]+(index-lower)*(profile[lower+1]-profile[lower])
        if kind == 1:
            return max(0., -profile[0]*math.expm1(-profile[1]*lam)-profile[2]*lam)
        if kind == 2:
            b, c, h, e = profile[:4]
            s = b*lam
            return h*math.sin(c*math.atan(s-e*(s-math.atan(s))))
        h, peak, alpha, tail, a, q = profile
        if lam <= peak:
            x = lam/peak
            return h*(1-((1-x)/(1+alpha*x))**2)
        x = (lam-peak)/(1-peak)
        return h*(tail+(1-tail)*(math.exp(-a*x**q)-math.exp(-a))/-math.expm1(-a))

    @njit(cache=True, inline="always")
    def _slip(v, w, r):
        return max(0., min(1., 1-r*w/max(v, 1e-8)))

    @njit(cache=True, inline="always")
    def _derivative(v, w, r, g, scale, torque, profile, kind):
        value = _coefficient(_slip(v, w, r), profile, kind)
        dw = scale*value-torque
        return -g*value, max(0., dw) if w <= 0. else dw

    @njit(cache=True, nogil=True)
    def _run(initial, actions, profiles, kind, mass, g, r, inertia, dt, control_dt, torque_max):
        k, a, horizon = len(profiles), len(actions), actions.shape[-1]
        shape = (k, a, horizon)
        state = np.empty((k, a, horizon, 2))
        slips, mus, means = np.empty(shape), np.empty(shape), np.empty(shape)
        high, lock = np.empty(shape), np.empty(shape)
        steps = round(control_dt/dt)
        scale = r*mass*g/inertia
        for i in range(k):
            profile = profiles[i]
            for j in range(a):
                v, w = initial
                lam = _slip(v, w, r)
                for t in range(horizon):
                    torque = min(torque_max, max(0., actions[j, t]))/inertia
                    before, max_slip, min_wheel = v, lam, w
                    for _ in range(steps):
                        if kind:
                            dv, dw = _derivative(v, w, r, g, scale, torque, profile, kind)
                            bv, bw = _derivative(v+dt*dv/2, w+dt*dw/2, r, g, scale, torque, profile, kind)
                            cv, cw = _derivative(v+dt*bv/2, w+dt*bw/2, r, g, scale, torque, profile, kind)
                            ev, ew = _derivative(v+dt*cv, w+dt*cw, r, g, scale, torque, profile, kind)
                            v = max(0., v+dt*(dv+2*bv+2*cv+ev)/6)
                            w = max(0., w+dt*(dw+2*bw+2*cw+ew)/6)
                        else:
                            value = _coefficient(lam, profile, kind)
                            v = max(0., v-dt*g*value)
                            w = max(0., w+dt*(scale*value-torque))
                        lam = _slip(v, w, r)
                        max_slip, min_wheel = max(max_slip, lam), min(min_wheel, w)
                    state[i, j, t, 0], state[i, j, t, 1] = v, w
                    slips[i, j, t] = lam
                    mus[i, j, t] = _coefficient(lam, profile, kind)
                    means[i, j, t] = (before-v)/(g*control_dt)
                    high[i, j, t], lock[i, j, t] = max_slip, min_wheel
        return state, slips, mus, means, high, lock


def _forecast(current, actions, profiles, kind, c):
    arrays = _run(np.ascontiguousarray(current, dtype=np.float64),
                  np.ascontiguousarray(actions, dtype=np.float64),
                  np.ascontiguousarray(profiles, dtype=np.float64), kind,
                  c["mass"], c["gravity"], c["radius"], c["inertia"], c["physics_dt"],
                  c["control_dt"], c["torque_max"])
    return Forecast(*arrays, c["control_dt"], c["mass"]*c["gravity"])


def static_forecast(current, actions, curves, c):
    if njit is None:
        return curve_forecast(current, actions, curves, c)
    return _forecast(current, actions, np.atleast_2d(curves), 0, c)


def oracle_forecast(predictor, current, actions, road):
    if njit is None:
        return predictor.predict(current, actions)
    params = np.zeros((1, 6))
    params[0, :len(road["parameters"])] = road["parameters"]
    return _forecast(current, actions, params, FAMILIES[road["family"]], predictor.c)
