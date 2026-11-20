"""Matched-size action-conditioned fine-tuning on open and closed histories."""
from __future__ import annotations

import copy
import itertools
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .active_experiment import environment
from .active_rollout import candidate_actions
from .active_training import torch_forecast
from .adaptive_belief import ContinuousTire
from .config import write_json
from .evaluation import write_csv
from .models import seed_everything
from .spatial_road import SpatialRoad
from .tire_prior import friction_query


DISTRIBUTIONS = ("open", "closed", "mixed", "hard", "dynamic")


def gather_windows(episodes, partition, d):
    scenes = {s["id"]: s for branch in partition.values() for s in branch}
    windows = []
    for episode in episodes:
        identifier = episode["metrics"]["road_id"]
        if identifier not in scenes:
            raise ValueError("An episode does not belong to the declared data partition")
        trace = episode["trace"]
        for i in range(0, len(trace), 7):
            row = trace[i]
            active = any(x.get("actual_exploration") for x in trace[max(0, i-4):i+4])
            hard = (any(x["safety"] for x in trace[max(0, i-4):i+4]) or
                    row["slip"] >= .38 or (row.get("innovation") or 0.) >= 3.)
            dynamic = any(abs(row["position_m"]-pos) <= 3.5 for pos in scenes[identifier]["positions"])
            windows.append({"episode": episode, "index": i, "scene": scenes[identifier], "active": active,
                            "hard": hard, "dynamic": dynamic})
    return windows


def window_history(trace, index, count):
    actual = trace[max(0, index-count+1):index+1]
    rows = np.array([[row["v"], row["omega"], row["slip"], row["previous"]] for row in actual])
    return np.concatenate((np.repeat(rows[:1], count-len(rows), axis=0), rows))


def labels_from_windows(c, d, windows, seed):
    rng = np.random.default_rng(seed)
    roads, data = {}, []
    for window in windows:
        scene = window["scene"]
        road = roads.get(scene["id"])
        if road is None:
            road = roads[scene["id"]] = SpatialRoad.build(scene, c)
        rows, index = window["episode"]["trace"], window["index"]
        row = rows[index]
        h = window_history(rows, index, c["history_steps"])
        actions = candidate_actions(h[-1, 3], d["training_steps"], c, random=rng)[0]
        y, _ = road.forecast([*h[-1, :2], row["position_m"]], actions, stop=0.)
        data.append({"history": h, "actions": actions, "state": y.state[0], "slip": y.slip[0], "mu": y.mu[0],
                     "reward": y.interval_mu[0].mean(axis=-1), "high": y.high[0], "lock": y.lock[0],
                     "curve": road.profile_at(row["position_m"])[::4]})
    return {name: torch.as_tensor(np.stack([row[name] for row in data]), dtype=torch.float32) for name in data[0]}


def _choose(rng, data, size):
    return [data[i] for i in rng.choice(len(data), size, replace=len(data) < size)]


def materialize(c, d, output, key, train_episodes, validation_episodes, partitions, legacy_path):
    folder = output/"data"
    folder.mkdir(exist_ok=True)
    train = gather_windows(train_episodes, partitions["train"], d)
    validation = gather_windows(validation_episodes, partitions["validation"], d)
    constant = [r for r in train if r["scene"]["kind"] == "constant"]
    hard = [r for r in constant if r["hard"] or r["active"]]
    dynamic = [r for r in train if r["dynamic"]]
    if not constant or not validation:
        raise ValueError("No eligible closed-loop training or validation histories")
    rng = np.random.default_rng(d["seed"]+602)
    n = d["training_cases_per_distribution"]
    old = torch.load(legacy_path, weights_only=True)
    fields = {"history", "actions", "state", "slip", "mu", "reward", "high", "lock", "curve"}
    old = {name: old[name] for name in sorted(fields)}
    if old["actions"].shape[-1] != d["training_steps"]:
        raise ValueError("Training horizon must match the cached open-loop event labels")
    old_idx = rng.choice(len(old["history"]), n, replace=n > len(old["history"]))
    open_data = {name: x[old_idx].clone() for name, x in old.items()}
    if open_data["actions"].shape[1] != 11:
        raise ValueError("Expected 11 open-loop candidate branches")
    closed_pick = _choose(rng, constant, n)
    hard_pick = closed_pick[:n//2]+_choose(rng, hard or constant, n-n//2)
    dynamic_pick = closed_pick[:n//2]+_choose(rng, dynamic or constant, n-n//2)
    plans = {"closed": closed_pick, "mixed": closed_pick[:n//2], "hard": hard_pick, "dynamic": dynamic_pick}
    manifest = {"key": key, "training_only": True, "open_from": str(legacy_path), "datasets": {}}
    data = {"open": open_data}
    for j, (name, chosen) in enumerate(plans.items()):
        new = labels_from_windows(c, d, chosen, d["seed"]+700+j)
        if name == "mixed":
            new = {field: torch.cat((value, open_data[field][len(chosen):])) for field, value in new.items()}
        data[name] = new
        manifest["datasets"][name] = {"windows": len(new["history"]), "closed_windows": len(chosen),
                                      "unique_windows": len({(r["scene"]["id"], r["episode"]["metrics"]["method"],
                                                              r["episode"]["metrics"]["initial_speed"], r["index"]) for r in chosen}),
                                      "hard_windows": sum(x["hard"] for x in chosen),
                                      "active_windows": sum(x["active"] for x in chosen),
                                      "dynamic_windows": sum(x["dynamic"] for x in chosen),
                                      "roads": sorted({r["scene"]["id"] for r in chosen})}
    val = _choose(rng, validation, d["training_validation_cases"])
    data["validation"] = labels_from_windows(c, d, val, d["seed"]+760)
    manifest["datasets"]["validation"] = {"windows": len(val), "roads": sorted({r["scene"]["id"] for r in val})}
    for name, dataset in data.items():
        target = folder/f"{name}.pt"
        tmp = target.with_suffix(".tmp")
        torch.save(dataset, tmp)
        tmp.replace(target)
    write_json(output/"closed_loop_data_manifest.json", manifest)


def free_forecast(model, initial, history, actions, c):
    """E4 A keeps its original free decoder and fixed history context per case."""
    latent = model.encode(history)
    grid = torch.linspace(0, 1, 129, device=history.device).expand(len(history), -1)
    curve = friction_query(model, latent, grid)
    v = initial[:, 0, None].expand(-1, actions.shape[1])
    w = initial[:, 1, None].expand_as(v)
    r, g, dt = c["radius"], c["gravity"], c["physics_dt"]
    factor = r*c["mass"]*g/c["inertia"]

    def coeff(lam):
        index = lam*128
        lo = index.long().clamp(max=127)
        left, right = curve.gather(1, lo), curve.gather(1, lo+1)
        return left+(index-lo)*(right-left)

    state, slips, force = [], [], []
    high = (1-r*w/v.clamp_min(1e-8)).clamp(0, 1)
    minimum = w
    for torque in actions.unbind(-1):
        for _ in range(round(c["control_dt"]/dt)):
            mu = coeff((1-r*w/v.clamp_min(1e-8)).clamp(0, 1))
            v = (v-dt*g*mu).clamp_min(0)
            w = (w+dt*(factor*mu-torque/c["inertia"])).clamp_min(0)
            lam = (1-r*w/v.clamp_min(1e-8)).clamp(0, 1)
            high, minimum = torch.maximum(high, lam), torch.minimum(minimum, w)
        state.append(torch.stack((v, w), dim=-1))
        slips.append(lam)
        force.append(coeff(lam))
    reward = (initial[:, :1]-v)/(g*c["control_dt"]*actions.shape[-1])
    return (torch.stack(state, dim=-2)[:, None], torch.stack(slips, dim=-1)[:, None],
            torch.stack(force, dim=-1)[:, None], reward[:, None], high[:, None], minimum[:, None])


def objective(model, kind, batch, c, d):
    history, actions = batch["history"], batch["actions"]
    if kind == "e4":
        logp = history.new_zeros((len(history), 1))
        state, slip, force, reward, high, wheel = free_forecast(model, history[:, -1, :2], history, actions, c)
    else:
        raw, logp = model(history)
        state, slip, force, reward, high, wheel = torch_forecast(history[:, -1, :2], actions, raw, c)
    weight = logp.exp()
    target = batch["state"][:, None]
    err = ((state-target)/state.new_tensor([.35, 15])).square().mean(dim=(-1, -2, -3))
    err = err+.35*((slip-batch["slip"][:, None])/.18).square().mean(dim=(-1, -2))
    err = err+.45*((force-batch["mu"][:, None])/.2).square().mean(dim=(-1, -2))
    rollout = -.5*torch.logsumexp(logp-2*err, dim=1).mean()
    y = torch.cat((state[..., -1, :], force[..., -1, None]), dim=-1)
    x = torch.cat((batch["state"][..., -1, :], batch["mu"][..., -1, None]), dim=-1)
    expected = torch.einsum("bk,bkad->bad", weight, y)
    action = (((expected-expected[:, :1])-(x-x[:, :1]))[:, 1:]/y.new_tensor([.25, 12, .2])).square().mean()
    q = torch.einsum("bk,bka->ba", weight, reward)
    label = batch["reward"][:, :, None]-batch["reward"][:, None, :]
    valid = label.abs() > d["tie_margin_mu"]
    rank = (F.softplus(-label.sign()*(q[:, :, None]-q[:, None, :])/.025)*valid).sum()/valid.sum().clamp_min(1)
    high_prob = torch.einsum("bk,bka->ba", weight, ((high-.8)/.04).sigmoid())
    lock_prob = torch.einsum("bk,bka->ba", weight, ((.5-wheel)/.65).sigmoid())
    risk = ((high_prob-batch["high"]).square()+(lock_prob-batch["lock"]).square()).mean()
    # Predict the scalar causal force innovation using the first five intervals.
    n = min(5, state.shape[-2])
    proxy = (history[:, None, None, -1, 0]-state[..., n-1, 0])/(n*c["control_dt"]*c["gravity"])
    true_proxy = (history[:, None, -1, 0]-batch["state"][..., n-1, 0])/(n*c["control_dt"]*c["gravity"])
    residual = ((proxy-true_proxy[:, None])/.08).square().mean(dim=-1)
    observation = -.5*torch.logsumexp(logp-residual, dim=1).mean()
    total = rollout+.4*action+.14*rank+.24*risk+.15*observation
    return {"loss": total, "rollout": rollout, "action": action, "rank": rank, "risk": risk, "observation": observation}


def make_model(kind, donor, d):
    if kind == "continuous":
        return ContinuousTire(donor, d["continuous_scale"])
    return copy.deepcopy(donor)


def train_job(args):
    c, d, folder, key, kind, distribution, seed, donor = args
    name = f"{kind}_{distribution}_{seed}"
    target = folder/"models"/f"{name}.pt"
    if target.exists() and target.with_suffix(".csv").exists():
        prior = torch.load(target, weights_only=True)
        if prior["key"] != key:
            raise ValueError(f"Stale checkpoint: {name}")
        return prior["summary"]
    seed_everything(seed, c)
    model = make_model(kind, donor, d).requires_grad_(True)
    optim = torch.optim.AdamW(model.parameters(), lr=d["learning_rate"])
    train = torch.load(folder/"data"/f"{distribution}.pt", weights_only=True)
    validation = torch.load(folder/"data"/"validation.pt", weights_only=True)
    rng = torch.Generator().manual_seed(seed)
    best, log = float("inf"), []
    start = perf_counter()
    for epoch in range(1, d["training_epochs"]+1):
        model.train()
        for ix in torch.randperm(len(train["history"]), generator=rng).split(d["training_batch_size"]):
            values = objective(model, kind, {name: x[ix] for name, x in train.items()}, c, d)
            optim.zero_grad()
            values["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optim.step()
        model.eval()
        vals = {}
        with torch.no_grad():
            for ix in torch.arange(len(validation["history"])).split(d["training_batch_size"]):
                values = objective(model, kind, {name: x[ix] for name, x in validation.items()}, c, d)
                for metric, value in values.items():
                    vals[metric] = vals.get(metric, 0.)+len(ix)*value.item()/len(validation["history"])
        row = {"epoch": epoch, **{f"val_{metric}": value for metric, value in vals.items()}}
        log.append(row)
        if vals["loss"] < best:
            best, record = vals["loss"], row
            state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    if not np.isfinite(best):
        raise FloatingPointError(f"No finite validation model for {name}")
    summary = {"method": name, "kind": kind, "distribution": distribution, "seed": seed,
               "windows": len(train["history"]), "seconds": perf_counter()-start, **record}
    write_csv(target.with_suffix(".csv"), log)
    temporary = target.with_suffix(".tmp")
    torch.save({"key": key, "summary": summary, "state": state}, temporary)
    temporary.replace(target)
    print(f"P4 trained {name}: {best:.5f}", flush=True)
    return summary


def train_variants(c, d, output, key, kind, legacy, seeds, workers, smoke):
    (output/"models").mkdir(exist_ok=True)
    distributions = ("open", "closed") if smoke else DISTRIBUTIONS
    variants = [("e4", "closed"), *( (kind, data) for data in distributions)]
    jobs = [(c, d, output, key, method, distribution, seed,
             legacy["e4" if method == "e4" else "prototype", seed])
            for (method, distribution), seed in itertools.product(variants, seeds)]
    with ProcessPoolExecutor(max_workers=workers, initializer=environment, initargs=(c["torch_threads"],)) as pool:
        results = list(pool.map(train_job, jobs))
    write_csv(output/"training_comparison.csv", results)
    return results
