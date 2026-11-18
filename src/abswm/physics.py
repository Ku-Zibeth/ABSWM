from __future__ import annotations

import math

import numpy as np

# Typical Burckhardt road parameters; sources and limitations are in docs/method.md.
TYPICAL_ROADS = {
    "dry": (1.2801, 23.99, 0.52),
    "wet": (0.857, 33.822, 0.347),
    "snow": (0.1946, 94.129, 0.0646),
}


def slip(state, radius: float):
    x = np.asarray(state)
    return np.clip(1 - radius * x[..., 1] / np.maximum(x[..., 0], 1e-8), 0, 1)


def friction(lam, theta):
    theta = np.asarray(theta)
    lam = np.clip(lam, 0, 1)
    return np.maximum(theta[..., 0] * (-np.expm1(-theta[..., 1] * lam))
                      - theta[..., 2] * lam, 0)


def peak_slip(theta):
    t = np.asarray(theta)
    if np.any(t <= 0):
        raise ValueError("Burckhardt parameters must be positive")
    return np.clip(np.log(t[..., 0] * t[..., 1] / t[..., 2]) / t[..., 1], 0, 1)


def valid_road(theta) -> bool:
    t = np.asarray(theta)
    if np.any(t <= 0):
        return False
    peak = float(peak_slip(t))
    return (0.03 <= peak <= 0.30 and 0.10 <= friction(peak, t) <= 1.5
            and t[0] * (1 - np.exp(-t[1])) - t[2] > 0)


def derivative(state, torque, theta, c):
    x = np.asarray(state)
    force = c["mass"] * c["gravity"] * friction(slip(x, c["radius"]), theta)
    dv = -force / c["mass"]
    dw = (c["radius"] * force - np.clip(torque, 0, c["torque_max"])) / c["inertia"]
    dw = np.where((x[..., 1] <= 0) & (dw < 0), 0, dw)
    return np.stack((dv, dw), axis=-1)


def advance(state, torque, theta, c):
    """Projected RK4 at physics_dt; torque is held for one control interval."""
    x = np.array(state, dtype=np.float64, copy=True)
    distance = np.zeros(x.shape[:-1])
    dt = c["physics_dt"]
    for _ in range(round(c["control_dt"] / dt)):
        before = x[..., 0].copy()
        k1 = derivative(x, torque, theta, c)
        k2 = derivative(x + dt * k1 / 2, torque, theta, c)
        k3 = derivative(x + dt * k2 / 2, torque, theta, c)
        k4 = derivative(x + dt * k3, torque, theta, c)
        x = np.maximum(x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6, 0)
        distance += dt * (before + x[..., 0]) / 2
    return x, distance


def advance_scalar(v, w, torque, theta, c):
    """Scalar equivalent of advance, avoiding NumPy overhead in feedback runs."""
    m, r, j, g = c["mass"], c["radius"], c["inertia"], c["gravity"]
    a, b, d = theta
    dt = c["physics_dt"]
    torque = min(c["torque_max"], max(0.0, torque))

    def rhs(vv, ww):
        lam = min(1.0, max(0.0, 1-r*ww/max(vv, 1e-8)))
        mu = max(0.0, a * (-math.expm1(-b*lam)) - d*lam)
        force = m*g*mu
        dw = (r*force-torque)/j
        return -g*mu, max(0.0, dw) if ww <= 0 else dw

    distance = 0.0
    for _ in range(round(c["control_dt"] / dt)):
        old_v = v
        av, aw = rhs(v, w)
        bv, bw = rhs(v+dt*av/2, w+dt*aw/2)
        cv, cw = rhs(v+dt*bv/2, w+dt*bw/2)
        dv, dw = rhs(v+dt*cv, w+dt*cw)
        v = max(0.0, v+dt*(av+2*bv+2*cv+dv)/6)
        w = max(0.0, w+dt*(aw+2*bw+2*cw+dw)/6)
        distance += dt*(old_v+v)/2
    return v, w, distance
