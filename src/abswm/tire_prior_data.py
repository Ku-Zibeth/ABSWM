"""Independent steady-state tire families and causal evaluation histories."""

from __future__ import annotations

import hashlib

import numpy as np

from .data import make_history, sample_roads
from .physics import friction, slip
from .recurrent_study import slew_limited


def burckhardt(road):
    return {"id": road["id"], "family": "burckhardt", "surface": "low" if road["family"] == "snow" else "hard",
            "subfamily": road["family"], "parameters": road["theta"]}


def row(kind, params, label=""):
    identifier = hashlib.sha256(np.asarray(params, dtype="<f8").tobytes()+kind.encode()).hexdigest()[:16]
    return {"id": identifier, "family": kind, "surface": "hard", "subfamily": label or kind, "parameters": list(params)}


def generate_family(kind, count, seed):
    rng = np.random.default_rng(seed)
    height = rng.uniform(.60, 1.22, count)
    peak = rng.uniform(.095, .225, count)
    if kind == "magic":
        c = rng.uniform(1.43, 1.86, count)
        e = rng.uniform(-.15, .35, count)
        shape_target = np.tan(np.pi/(2*c))
        low, high = np.zeros(count), np.full(count, 12.)
        for _ in range(45):
            middle = (low+high)/2
            under = middle-e*(middle-np.arctan(middle)) < shape_target
            low, high = np.where(under, middle, low), np.where(under, high, middle)
        parameters = np.column_stack(((low+high)/(2*peak), c, height, e))
    else:
        parameters = np.column_stack((height, peak, rng.uniform(.3, 2.25, count), rng.uniform(.48, .89, count),
                                      rng.uniform(.75, 3.0, count), rng.uniform(1.25, 3.0, count)))
    return [row(kind, params) for params in parameters]


def stress_pairs():
    common = [.83, .15, 1.25, .63, 1.4, 1.65]
    variants = {
        "same_height": [(1, .11), (1, .21)],
        "same_peak": [(0, .62), (0, 1.13)],
        "same_initial_slope": [None, None],
        "same_peak_height": [(2, .35), (2, 2.50)],
        "same_rise": [(3, .49), (3, .87)],
        "weak_identifiability": [None, None],
    }
    pairs = []
    for name, changes in variants.items():
        for j, change in enumerate(changes):
            params = common.copy()
            if change is None:
                params[1] = ((.11, .21) if name == "same_initial_slope" else (.12, .183))[j]
                slope = 25. if name == "same_initial_slope" else 32.
                params[2] = slope*params[1]/(2*params[0])-1
                if name == "weak_identifiability":
                    params[4] = 1.8
            else:
                params[change[0]] = change[1]
            pairs.append(row("rational", params, f"{name}_{'ab'[j]}"))
    return pairs


def manifests(c, d, original):
    profile = {**c, "data_seed": d["independent_burckhardt_seed"], "roads": {"test": d["independent_burckhardt_roads"]}}
    independent = [burckhardt(r) for r in sample_roads(profile)["test"]]
    ab = {r["id"]: r for r in original["test"]}
    result = {"independent_burckhardt": independent,
              "magic_train": generate_family("magic", d["magic_train_roads"], d["magic_train_seed"]),
              "independent_magic": generate_family("magic", d["magic_test_roads"], d["magic_test_seed"]),
              "withheld_rational": generate_family("rational", d["rational_test_roads"], d["rational_test_seed"]),
              "decoupled": stress_pairs(), "ab": [burckhardt(ab[name]) for name in d["ab_ids"]]}
    ids = {r["id"] for split in original.values() for r in split}
    for name, roads in result.items():
        if name == "ab":
            continue
        new_ids = {r["id"] for r in roads}
        if ids & new_ids or len(new_ids) != len(roads):
            raise ValueError("Tire family partitions overlap")
        ids.update(new_ids)
    return result


def tire_function(roads):
    groups = []
    for family in sorted({road["family"] for road in roads}):
        indices = np.array([i for i, r in enumerate(roads) if r["family"] == family])
        groups.append((family, indices, np.array([roads[i]["parameters"] for i in indices])))

    def evaluate(values):
        lam = np.clip(values, 0, 1)
        output = np.empty_like(lam, dtype=float)
        for family, ids, values in groups:
            x = lam[ids]
            shape = (len(ids),)+(1,)*(x.ndim-1)
            p = [v.reshape(shape) for v in values.T]
            if family == "burckhardt":
                a, b, d = p
                mu = np.maximum(a*(-np.expm1(-b*x))-d*x, 0)
            elif family == "magic":
                b, c, h, e = p
                s = b*x
                mu = h*np.sin(c*np.arctan(s-e*(s-np.arctan(s))))
            else:
                h, peak, alpha, tail, a, q = p
                front = (x/peak).clip(0, 1)
                back = ((x-peak)/(1-peak)).clip(0, 1)
                rise = 1-((1-front)/(1+alpha*front))**2
                fall = tail+(1-tail)*(np.exp(-a*back**q)-np.exp(-a))/(-np.expm1(-a))
                mu = h*np.where(x <= peak, rise, fall)
            output[ids] = mu
        return output

    return evaluate


def advance(states, torque, coefficient, c):
    def derivative(x):
        mu = coefficient(slip(x, c["radius"]))
        dw = (c["radius"]*c["mass"]*c["gravity"]*mu-torque)/c["inertia"]
        return np.column_stack((-c["gravity"]*mu, np.where(x[:, 1] <= 0, np.maximum(dw, 0), dw)))

    dt = c["physics_dt"]
    for _ in range(round(c["control_dt"]/dt)):
        a = derivative(states)
        b = derivative(states+dt*a/2)
        d = derivative(states+dt*b/2)
        e = derivative(states+dt*d)
        states = np.maximum(states+dt*(a+2*b+2*d+e)/6, 0)
    return states


def trajectory(c, roads, speeds, rates, seconds=1.6, plateau=False):
    coefficient = tire_function(roads)
    x = np.column_stack((speeds, speeds/c["radius"]))
    xs, actions = [x], []
    dt = c["control_dt"]
    y = np.zeros(len(roads))
    for step in range(round(seconds/dt)):
        if plateau and (step+1)*dt > .85:
            target = y+900*dt*(.12-slip(x, c["radius"]))
        else:
            target = rates*(step+1)*dt
        y = np.clip(y+np.clip(target-y, -dt*c["torque_rate"], dt*c["torque_rate"]), 0, c["torque_max"])
        x = advance(x, y, coefficient, c)
        xs.append(x)
        actions.append(y)
    return np.stack(xs, axis=1).astype(np.float32), np.stack(actions, axis=1).astype(np.float32)


def pack(c, states, actions, roads, anchors, caps, trajectory_ids=None):
    history = np.stack([make_history(states[i], actions[i], t, c["history_steps"], c) for i, t in anchors])
    return {"history": history, "roads": [roads[i] for i, _ in anchors],
            "locators": np.asarray(anchors), "caps": np.asarray(caps), "raw": {"states": states, "actions": actions},
            "trajectory_ids": np.asarray(trajectory_ids if trajectory_ids is not None else np.arange(len(roads)))}


def stress_suite(c, d, roads, *, ab=False, memory=False):
    if ab:
        inputs = [(r, s, rate) for r in roads for s in c["initial_speeds"] for rate in (500, 1000, 1800, 3000)]
    else:
        inputs = [(r, 30., (1600, 2000, 2400)[i % 3]) for i, r in enumerate(roads)]
    traces, speeds, rates = zip(*inputs)
    x, u = trajectory(c, traces, np.array(speeds), np.array(rates), seconds=2.65 if memory else 2.5,
                      plateau=memory)
    anchors, caps = [], []
    forecast = c["horizon_steps"]
    if memory:
        for i in range(len(traces)):
            for time in (.8, 1.2, 1.7, 2.25):
                t = round(time/c["control_dt"])
                if x[i, t, 0] > c["stop_speed"]+.05:
                    anchors.append((i, t))
                    caps.append(time)
    else:
        observed = slip(x, c["radius"])
        for i in range(len(traces)):
            for cap in d["coverage_caps"]:
                crossed = np.flatnonzero(observed[i] > cap)
                if len(crossed) and 0 < crossed[0] <= len(u[i])-forecast and x[i, crossed[0]-1, 0] > c["stop_speed"]:
                    anchors.append((i, int(crossed[0])-1))
                    caps.append(cap)
    return pack(c, x, u, traces, anchors, caps)


def mixed_windows(c, roads, count, seed):
    rng = np.random.default_rng(seed)
    x, u = trajectory(c, roads, rng.choice(c["initial_speeds"], len(roads)),
                      rng.uniform(1400, 2600, len(roads)), seconds=1.5)
    anchors = []
    for i in rng.integers(len(roads), size=count):
        valid = np.flatnonzero(x[i, :-c["horizon_steps"], 0] > c["stop_speed"]+.05)
        high = min(len(valid), 60) if rng.random() < .4 else len(valid)
        anchors.append((int(i), int(valid[rng.integers(high)])))
    return pack(c, x, u, roads, anchors, np.zeros(count))


ACTIONS = ("hold", "release", "increase", "pulse", "multistage")


def action_suite(c, d, suite):
    hist = suite["history"]
    h = c["horizon_steps"]
    start = hist[:, -1, 3]
    hold = np.broadcast_to(start[:, None], (len(hist), h)).copy()
    offset = d["action_offset"]
    pulse = hold.copy()
    pulse[:, :h//2] += offset
    stage = hold+np.repeat([-offset/2, offset, -offset], [h//3, h//3, h-2*(h//3)])
    candidates = np.stack([slew_limited(start, targets, c) for targets in (hold, np.zeros_like(hold), hold+offset, pulse, stage)],
                          axis=1)
    expanded = [road for road in suite["roads"] for _ in ACTIONS]
    fn = tire_function(expanded)
    states, mu = [], []
    y = np.repeat(hist[:, -1, :2].astype(float), len(ACTIONS), axis=0)
    for torques in candidates.reshape(-1, h).T:
        y = advance(y, torques, fn, c)
        states.append(y)
        mu.append(fn(slip(y, c["radius"])))
    return candidates, np.stack(states, axis=1).reshape(len(hist), len(ACTIONS), h, 2), np.stack(
        mu, axis=1).reshape(len(hist), len(ACTIONS), h)
