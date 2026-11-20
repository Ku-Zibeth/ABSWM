"""Position-indexed quarter-car plant and independent counterfactual truth."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .active_fast import njit
from .active_rollout import Forecast, GRID
from .config import fingerprint
from .data import sample_roads
from .tire_prior_data import burckhardt, generate_family, row, tire_function


FAMILIES = ("burckhardt", "magic", "rational")
SCENARIOS = ("high_low", "low_high", "patch", "gradual", "shape")


def _compile(function):
    return njit(cache=True, nogil=True)(function) if njit is not None else function


@_compile
def _region(position, starts, widths):
    profile, blend = 0, 0.
    for i in range(len(starts)):
        if position < starts[i]:
            break
        if widths[i] > 0 and position < starts[i]+widths[i]:
            blend = (position-starts[i])/widths[i]
            break
        profile += 1
    return profile, blend


@_compile
def _mu(v, wheel, s, r, profiles, starts, widths):
    lam = max(0., min(1., 1-r*wheel/max(v, 1e-8)))
    i, f = _region(s, starts, widths)
    p = lam*(profiles.shape[1]-1)
    j = min(int(p), profiles.shape[1]-2)
    q = p-j
    value = profiles[i, j]*(1-q)+profiles[i, j+1]*q
    if f > 0:
        other = profiles[i+1, j]*(1-q)+profiles[i+1, j+1]*q
        value += f*(other-value)
    return value


@_compile
def _rhs(v, w, s, torque, profiles, starts, widths, r, g, eta):
    mu = _mu(v, w, s, r, profiles, starts, widths)
    dw = eta*mu-torque
    return -g*mu, max(0., dw) if w <= 0. else dw, max(0., v)


@_compile
def _rollout(initial, actions, profiles, starts, widths, constants, rk4, stop):
    mass, gravity, radius, inertia, dt, control_dt, torque_max = constants
    eta = radius*mass*gravity/inertia
    n, h = actions.shape
    state = np.empty((n, h, 2))
    records = np.zeros((10, n, h))
    peaks = np.empty(len(profiles))
    for i in range(len(profiles)):
        peaks[i] = np.max(profiles[i])
    for row in range(n):
        v, w, s = initial
        for tick in range(h):
            start_v = v
            hit, locked, elapsed, potential = 0., 0., 0., 0.
            highest, lowest = max(0., min(1., 1-radius*w/max(v, 1e-8))), w
            torque = min(torque_max, max(0., actions[row, tick]))/inertia
            for _ in range(round(control_dt/dt)):
                if v <= stop:
                    break
                old_v, old_w, old_s = v, w, s
                a, aw, av = _rhs(v, w, s, torque, profiles, starts, widths, radius, gravity, eta)
                if rk4:
                    b, bw, bv = _rhs(v+dt*a/2, w+dt*aw/2, s+dt*av/2, torque,
                                     profiles, starts, widths, radius, gravity, eta)
                    d, dw, dv = _rhs(v+dt*b/2, w+dt*bw/2, s+dt*bv/2, torque,
                                     profiles, starts, widths, radius, gravity, eta)
                    e, ew, ev = _rhs(v+dt*d, w+dt*dw, s+dt*dv, torque, profiles,
                                     starts, widths, radius, gravity, eta)
                    v = max(0., v+dt*(a+2*b+2*d+e)/6)
                    w = max(0., w+dt*(aw+2*bw+2*dw+ew)/6)
                    s += dt*(av+2*bv+2*dv+ev)/6
                else:
                    v = max(0., v+dt*a)
                    w = max(0., w+dt*aw)
                    s += dt*av
                duration = dt
                if v < stop:
                    fraction = (old_v-stop)/max(old_v-v, 1e-12)
                    w = old_w+fraction*(w-old_w)
                    v, duration = stop, dt*fraction
                    s = old_s+duration*(old_v+v)/2
                lam = max(0., min(1., 1-radius*w/max(v, 1e-8)))
                highest, lowest = max(highest, lam), min(lowest, w)
                hit += (lam >= .8)*duration
                locked += (w <= .5)*duration
                elapsed += duration
                i, blend = _region((old_s+s)/2, starts, widths)
                if blend > 0:
                    peak = np.max((1-blend)*profiles[i]+blend*profiles[i+1])
                else:
                    peak = peaks[i]
                potential += duration*peak
            state[row, tick, 0], state[row, tick, 1] = v, w
            records[0, row, tick] = max(0., min(1., 1-radius*w/max(v, 1e-8)))
            records[1, row, tick] = _mu(v, w, s, radius, profiles, starts, widths)
            records[2, row, tick] = (start_v-v)/(gravity*control_dt)
            records[3, row, tick] = highest
            records[4, row, tick] = lowest
            records[5, row, tick], records[6, row, tick] = s, elapsed
            records[7, row, tick], records[8, row, tick] = hit, locked
            records[9, row, tick] = potential
    return state, records


@dataclass(frozen=True)
class SpatialRoad:
    spec: dict
    c: dict
    profiles: np.ndarray
    starts: np.ndarray
    widths: np.ndarray
    constants: tuple

    @classmethod
    def build(cls, spec, c):
        roads = spec["segments"]
        values = np.broadcast_to(GRID, (len(roads), len(GRID)))
        profiles = tire_function(roads)(values)
        starts = np.asarray(spec["positions"], dtype=float)
        widths = np.asarray(spec["widths"], dtype=float)
        if (len(roads) != len(starts)+1 or starts.shape != widths.shape or (widths < 0).any()
                or (starts <= 0).any() or (starts[1:] <= starts[:-1]+widths[:-1]).any()
                or not np.isfinite(profiles).all() or (profiles < 0).any() or (profiles > 2).any()):
            raise ValueError("Invalid spatial road")
        constants = tuple(float(c[x]) for x in ("mass", "gravity", "radius", "inertia", "physics_dt",
                                               "control_dt", "torque_max"))
        return cls(spec, c, profiles, starts, widths, constants)

    def profile_at(self, position):
        index, blend = _region(float(position), self.starts, self.widths)
        first = self.profiles[index]
        return first if not blend else first+blend*(self.profiles[index+1]-first)

    def forecast(self, state, actions, *, preview=True, rk4=True, stop=None):
        profiles, starts, widths = self.profiles, self.starts, self.widths
        if not preview:
            profiles = self.profile_at(state[2])[None]
            starts, widths = starts[:0], widths[:0]
        terminal = self.c["stop_speed"] if stop is None else stop
        final, data = _rollout(np.asarray(state, dtype=float), np.asarray(actions, dtype=float),
                               profiles, starts, widths, self.constants, rk4, terminal)
        x = Forecast(final[None], *(row[None] for row in data[:5]), self.c["control_dt"],
                     self.c["mass"]*self.c["gravity"])
        return x, data[5:]


def scene(kind, roads, positions=(), widths=()):
    profiles = [r["id"] for r in roads]
    identity = fingerprint([kind, profiles, positions, widths])[:16]
    if kind == "constant":
        identity = profiles[0]
    return {"id": identity, "kind": kind, "family": roads[0]["family"], "segments": roads,
            "positions": list(positions), "widths": list(widths)}


def sample_family(c, family, count, seed):
    if family == "burckhardt":
        profile = {**c, "data_seed": seed, "roads": {"sample": count}}
        return [burckhardt(r) for r in sample_roads(profile)["sample"]]
    return generate_family(family, count, seed)


def make_partitions(c, d, previously_seen=()):
    """Freeze all component road identities, including shared patch endpoints."""
    partitions, seen = {}, set(previously_seen)
    for split_id, split in enumerate(("train", "validation", "test")):
        offset = d["seed"]+1009*split_id
        sizes = d["constant_roads"][split]
        constant = [scene("constant", [road]) for i, (family, n) in enumerate(zip(FAMILIES, sizes))
                    if n for road in sample_family(c, family, n, offset+i)]
        dynamic = []
        repeats = d["dynamic_repeats"][split]
        for i, kind in enumerate(SCENARIOS):
            # Each constituent draw has its own seed and split. Some test shapes use the held-out family.
            for j in range(repeats):
                seed = offset+100+13*i+j
                pool = sample_family(c, "burckhardt", 9, seed)
                high, low = pool[3], pool[2]
                family = FAMILIES[2 if split == "test" and j % 2 else 1]
                mixed = sample_family(c, family, 8, seed+513)
                at = (3.0, 8.0, 14.0, 5.5)[j % 4]
                if kind == "high_low":
                    pair = [high, low if j % 3 != 2 else pool[1]]
                elif kind == "low_high":
                    pair = [low if j % 3 != 2 else pool[1], high]
                elif kind == "patch":
                    pair = [high, low, high]
                elif kind == "gradual":
                    pair = [high, mixed[0]] if j % 2 == 0 else [mixed[0], low]
                else:
                    # Match peak height and retain distinct native shapes.
                    candidates = mixed[:4]
                    curves = tire_function(candidates)(np.broadcast_to(GRID, (len(candidates), len(GRID))))
                    other = min(range(1, len(candidates)), key=lambda k: abs(curves[k].max()-curves[0].max()))
                    parameters = candidates[other]["parameters"].copy()
                    height = 0 if family == "rational" else 2
                    parameters[height] *= curves[0].max()/curves[other].max()*(1+.012*(-1)**j)
                    pair = [candidates[0], row(family, parameters, "matched_peak_distinct_shape")]
                length = (1.5, 3., 5., 2.5)[j % 4]
                positions = (at, at+length) if kind == "patch" else (at,)
                widths = (0., 0.) if kind == "patch" else (length if kind == "gradual" else 0.,)
                dynamic.append(scene(kind, pair, positions, widths))
        for entry in (*constant, *dynamic):
            ids = {road["id"] for road in entry["segments"]}
            if ids & seen:
                raise ValueError(f"Reused constituent road parameters in {split}: {entry['id']}")
            seen.update(ids)
        partitions[split] = {"constant": constant, "dynamic": dynamic}
    return partitions
