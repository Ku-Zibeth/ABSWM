"""Evaluator-only trajectories with a shifted post-peak friction curve."""

from __future__ import annotations

import numpy as np

from .data import make_history
from .physics import friction, peak_slip, slip


def tail_friction(lam, theta, shift):
    """Keep the peak and the first quarter of the slip range unchanged."""
    lam = np.clip(lam, 0, 1)
    baseline = friction(lam, theta)
    peak = peak_slip(theta)
    onset = np.maximum(0.25, peak+0.01)
    alpha = np.clip((lam-onset)/(1-onset), 0, 1)
    smooth = alpha**2*(3-2*alpha)
    ceiling = friction(peak, theta)
    offset = np.where(shift >= 0, ceiling-baseline, baseline)
    return baseline+0.55*shift*smooth*offset


def advance_tail(state, torque, theta, shift, c):
    """Projected RK4 with the same quarter-car equations as the base simulator."""
    dt = c["physics_dt"]
    radius, gravity = c["radius"], c["gravity"]
    torque = np.clip(torque, 0, c["torque_max"])

    def derivative(x):
        mu = tail_friction(slip(x, radius), theta, shift)
        dw = (radius*c["mass"]*gravity*mu-torque)/c["inertia"]
        return np.column_stack((-gravity*mu, np.where(x[:, 1] <= 0, np.maximum(dw, 0), dw)))

    for _ in range(round(c["control_dt"]/dt)):
        a = derivative(state)
        b = derivative(state+dt*a/2)
        d = derivative(state+dt*b/2)
        e = derivative(state+dt*d)
        state = np.maximum(state+dt*(a+2*b+2*d+e)/6, 0)
    return state


def make_stress_cases(c, roads, *, caps=(0.20, 0.65), rate=1800):
    specs = [(r, speed, shift) for r in roads for speed in c["initial_speeds"] for shift in (0, -1, 1)]
    theta = np.array([road["theta"] for road, _, _ in specs])
    shifts = np.array([shift for _, _, shift in specs])
    v = np.array([speed for _, speed, _ in specs])
    x = np.column_stack((v, v/c["radius"]))
    states, actions = [x], []
    active = np.ones(len(specs), dtype=bool)
    history_ends = np.full((len(caps), len(specs)), -1)
    dt = c["control_dt"]
    for t in range(round(c["data_time"]/dt)):
        u = min(c["torque_max"], min(rate, c["torque_rate"])*(t+1)*dt)
        new_state = advance_tail(x, u, theta, shifts, c)
        x = np.where(active[:, None], new_state, x)
        states.append(x)
        actions.append(u)
        lam = slip(x, c["radius"])
        for k, cap in enumerate(caps):
            crossed = active & (lam > cap) & (history_ends[k] < 0)
            history_ends[k, crossed] = t
        active &= (lam <= max(caps)) & (x[:, 0] > c["stop_speed"])
        if not active.any():
            break
    states, actions = np.stack(states, axis=1), np.array(actions)
    histories, labels = [], []
    for j, cap in enumerate(caps):
        for i, (road, speed, shift) in enumerate(specs):
            t = history_ends[j, i]
            if t < 0:
                continue
            history = make_history(states[i], actions, t, c["history_steps"], c)
            histories.append(history)
            labels.append({"road_id": road["id"], "family": road["family"], "speed": speed, "slip_cap": cap,
                           "shift": shift, "theta": road["theta"], "peak": road["peak"]})
    return np.asarray(histories), labels
