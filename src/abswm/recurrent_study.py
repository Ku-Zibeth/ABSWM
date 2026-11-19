"""Train and evaluate causal recurrent updates against the registered E4 A."""

from __future__ import annotations

import copy
from time import perf_counter

import numpy as np
import torch

from .data import load_split, make_history
from .decoder_study import load_windows, read_json, validate_partitions
from .diagnostic_experiments import prefix_before_cap
from .diagnostics import chunks, parametric_forecast, rmse, tensor
from .evaluation import write_csv
from .models import seed_everything
from .physics import friction, slip
from .recurrent_model import Posterior, RecurrentTireModel


CANDIDATES = ("factual", "hold", "release", "increase", "pulse")
OBSERVED = "recurrent_observed"
IMAGINED = "recurrent_transition"


def slew_limited(current, targets, c):
    previous = np.asarray(current, dtype=float)
    actions = []
    rate = c["torque_rate"]*c["control_dt"]
    for target in np.asarray(targets).T:
        bounded = np.clip(target, 0, c["torque_max"])
        previous = previous+np.clip(bounded-previous, -rate, rate)
        actions.append(previous)
    return np.stack(actions, axis=1).astype(np.float32)


def make_cases(c, d, raw, locators, roads, factual=None):
    fixed, training = [], []
    history_steps, horizon = c["history_steps"], c["horizon_steps"]
    for i, t in locators:
        x, u = raw["states"][i], raw["actions"][i]
        fixed.append(make_history(x, u, t, history_steps, c))
        # Cap backpropagation length while maintaining the same initial padding rule.
        length = min(history_steps+int(t), d["train_history_steps"])
        long = make_history(x, u, t, length, c)
        if len(long) < d["train_history_steps"]:
            long = np.pad(long, ((0, d["train_history_steps"]-len(long)), (0, 0)), mode="edge")
        training.append((long, length-1))
    h = np.asarray(fixed)
    current = h[:, -1, 3]
    bound = d["torque_offset_nm"]
    target = np.broadcast_to(current[:, None], (len(h), horizon)).copy()
    pulse = target.copy()
    pulse[:, :horizon//2] += bound
    hold, release, increase, pulse = (slew_limited(current, requested, c) for requested in (
        target, np.zeros_like(target), target+bound, pulse))
    actual_actions = raw["actions"]
    logged = np.stack([actual_actions[i, t:t+horizon] for i, t in locators])
    actions = np.stack((logged, hold, release, increase, pulse), axis=1)
    road_index = raw["road_index"][locators[:, 0]]
    theta = np.asarray([roads[i]["theta"] for i in road_index])
    synthetic = parametric_forecast(np.repeat(h[:, -1, :2], len(CANDIDATES)-1, axis=0),
                                    actions[:, 1:].reshape(-1, horizon), np.repeat(theta, len(CANDIDATES)-1, axis=0), c)
    synthetic = synthetic.reshape(len(h), len(CANDIDATES)-1, horizon, 2)
    if factual is None:
        factual = np.stack([raw["states"][i, t+1:t+horizon+1] for i, t in locators])
    states = np.concatenate((factual[:, None], synthetic), axis=1).astype(np.float32)
    mu = friction(slip(states, c["radius"]), theta[:, None, None]).astype(np.float32)
    return {"raw": raw, "locators": locators, "road_id": np.array([roads[i]["id"] for i in road_index]),
            "history": h, "training": np.asarray([row for row, _ in training]),
            "last_index": np.asarray([idx for _, idx in training]), "actions": actions, "states": states, "mu": mu}


def prefix_cases(c, d, raw, roads):
    locators, tags = [], []
    for road_idx, _ in enumerate(roads):
        for speed in c["initial_speeds"]:
            indices = np.flatnonzero((raw["road_index"] == road_idx) & (raw["initial_speed"] == speed)
                                     & (raw["kind"] == c["excitation_kinds"][0]))
            i = int(indices[0])
            # Keep only prefixes that leave the full factual horizon available.
            count = int(raw["lengths"][i])-c["horizon_steps"]
            observed = slip(raw["states"][i, :count], c["radius"])
            for cap in d["prefix_slip_caps"]:
                t, crossed = prefix_before_cap(observed, cap)
                if crossed:
                    locators.append((i, t))
                    tags.append(cap)
    cases = make_cases(c, d, raw, np.asarray(locators), roads)
    return {**cases, "caps": np.asarray(tags)}


@torch.inference_mode()
def stream_posteriors(model, cases, c):
    """Start each trace once; read out causal hidden states at all requested anchors."""
    raw, locators = cases["raw"], cases["locators"]
    traces = np.unique(locators[:, 0])
    contexts = torch.empty((len(locators), c["hidden_dim"]), device=c["device"])
    for trace_batch in chunks(traces, {**c, "batch_size": 24}):
        histories, groups = [], []
        width = max(int(locators[locators[:, 0] == i, 1].max()) for i in trace_batch)+c["history_steps"]
        for i in trace_batch:
            ids = np.flatnonzero(locators[:, 0] == i)
            t = int(locators[ids, 1].max())
            h = make_history(raw["states"][i], raw["actions"][i], t, t+c["history_steps"], c)
            histories.append(np.pad(h, ((0, width-len(h)), (0, 0)), mode="edge"))
            groups.append(ids)
        _, sequence = model.observe(tensor(np.stack(histories), c), return_sequence=True)
        for i, ids in enumerate(groups):
            time = locators[ids, 1]+c["history_steps"]-1
            contexts[ids] = sequence[i, time]
    return Posterior(contexts, tensor(cases["history"][:, -1], c))


def expand_memory(memory, branches):
    return Posterior(memory.z.repeat_interleave(branches, dim=0),
                     memory.last_observation.repeat_interleave(branches, dim=0))


def predict_branches(model, memory, actions):
    n, k, horizon = actions.shape
    x, mu, z = model.predict(expand_memory(memory, k), actions.reshape(-1, horizon))
    return x.reshape(n, k, horizon, 2), mu.reshape(n, k, horizon), z.reshape(n, k, horizon, -1)


def batch_objective(model, memory, actions, truth, true_mu, d):
    pred, mu, context = predict_branches(model, memory, actions)
    scale = truth.new_tensor([1, 10])
    state = ((pred-truth)/scale).square().mean()
    point = ((mu-true_mu)/1.5).square().mean()
    delta = (((pred[:, 1:]-pred[:, :1])-(truth[:, 1:]-truth[:, :1]))/scale).square().mean()
    drift = (context-memory.z[:, None, None, :]).square().mean()
    loss = d["state_weight"]*state+d["force_weight"]*point+d["action_delta_weight"]*delta+d["latent_drift_weight"]*drift
    return loss, {"state_mse": state, "mu_mse": point, "action_delta_mse": delta, "drift_mse": drift}


@torch.inference_mode()
def infer(c, d, model, cases, *, persistent):
    if persistent:
        posterior = stream_posteriors(model, cases, c)
    else:
        z = torch.cat([model.tire.encode(tensor(h, c)) for h in chunks(cases["history"], c)])
        posterior = Posterior(z, tensor(cases["history"][:, -1], c))
    states, mu = [], []
    for batch in chunks(np.arange(len(cases["history"])), d):
        start = Posterior(posterior.z[batch], posterior.last_observation[batch])
        prediction, force, _ = predict_branches(model, start, tensor(cases["actions"][batch], c))
        states.append(prediction.cpu().numpy())
        mu.append(force.cpu().numpy())
    return np.concatenate(states), np.concatenate(mu), posterior


def quality(c, cases, predicted, predicted_mu):
    truth, mu = cases["states"], cases["mu"]
    error, slip_error = predicted-truth, slip(predicted, c["radius"])-slip(truth, c["radius"])
    contrast = (predicted[:, 2:]-predicted[:, 1:2])-(truth[:, 2:]-truth[:, 1:2])
    contrast_mu = (predicted_mu[:, 2:]-predicted_mu[:, 1:2])-(mu[:, 2:]-mu[:, 1:2])
    return error, slip_error, predicted_mu-mu, contrast, contrast_mu


def validation(c, d, model, cases, *, persistent=True):
    predictions, force, _ = infer(c, d, model, cases, persistent=persistent)
    error, lam, mu, delta, _ = quality(c, cases, predictions, force)
    scale = np.array([1, 10])
    score = rmse(error/scale)+0.35*rmse(delta/scale)+0.15*rmse(mu/1.5)
    return {"val_score": score, "val_factual_endpoint_w": rmse(error[:, 0, -1, 1]),
            "val_factual_endpoint_slip": rmse(lam[:, 0, -1]), "val_factual_endpoint_fx": c["mass"]*c["gravity"]*rmse(mu[:, 0, -1]),
            "val_action_delta_endpoint_w": rmse(delta[..., -1, 1])}


def fit(c, d, model, train, val, seed):
    groups = [{"params": model.tire.encoder.parameters(), "lr": d["learning_rate_observer"]}]
    if model.transition is not None:
        groups.append({"params": model.transition.parameters(), "lr": d["learning_rate_transition"]})
    optimizer = torch.optim.Adam(groups)
    rng = np.random.default_rng(seed+d["seed"])
    parts = {k: tensor(train[k], c) for k in ("training", "actions", "states", "mu")}
    log, best_state, best_score = [], copy.deepcopy(model.state_dict()), np.inf
    for epoch in range(1, d["epochs"]+1):
        model.train()
        start = perf_counter()
        totals = {}
        for indices in chunks(rng.permutation(len(train["history"])), d):
            # Compare a single varied torque sequence (or the factual sequence) to hold.
            branches = np.column_stack((np.ones(len(indices), dtype=int), rng.choice([0, 2, 3, 4], len(indices))))
            keys = indices[:, None], branches
            h = parts["training"][indices]
            _, sequence = model.observe(h, return_sequence=True)
            last = train["last_index"][indices]
            memory = Posterior(sequence[np.arange(len(h)), last], h[np.arange(len(h)), last])
            optimizer.zero_grad(set_to_none=True)
            loss, components = batch_objective(model, memory, parts["actions"][keys], parts["states"][keys],
                                               parts["mu"][keys], d)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite recurrent training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            for name, value in {"loss": loss, **components}.items():
                totals[name] = totals.get(name, 0)+len(indices)*float(value.detach())
        metrics = validation(c, d, model.eval(), val)
        if metrics["val_score"] < best_score:
            best_score = metrics["val_score"]
            best_state = copy.deepcopy(model.state_dict())
        log.append({"epoch": epoch, "seconds": perf_counter()-start, **metrics,
                    **{f"train_{k}": v/len(train["history"]) for k, v in totals.items()}})
    model.load_state_dict(best_state)
    return log


def prepare(c, d, base, split, roads):
    raw = load_split(base/"data", split)
    windows, _ = load_windows(c, base, split)
    indices = np.arange(len(windows["locators"]))
    if split == "train":
        indices = np.random.default_rng(d["seed"]).choice(indices, d["train_windows"], replace=False)
    return make_cases(c, d, raw, windows["locators"][indices], roads, windows["states"][indices, 1:])


def inputs(c, d, base):
    manifest = read_json(base/"data"/"manifest.json")
    validate_partitions(c, manifest)
    return manifest["roads"], [prepare(c, d, base, split, manifest["roads"][split]) for split in ("train", "val")]


def train_recurrent(c, d, references, train, val, output, key):
    folder = output/"models"
    folder.mkdir(exist_ok=True)
    rows, models = [], {}
    for seed, reference in references.items():
        seed_everything(seed, c)
        stream = RecurrentTireModel(reference, d).eval().to(c["device"])
        for method in ("e4_a", "e4_a_stream"):
            models[method, seed] = stream
            rows.append({"method": method, "seed": seed, "best_epoch": 0, "training_seconds": 0,
                         **validation(c, d, stream, val, persistent=method != "e4_a")})
        for method in (OBSERVED, IMAGINED):
            seed_everything(seed, c)
            model = RecurrentTireModel(reference, d, imagine=method == IMAGINED).to(c["device"])
            path = folder/f"{method}_{seed}.pt"
            log = folder/f"{method}_{seed}_history.csv"
            if path.exists() and log.exists():
                saved = torch.load(path, weights_only=True, map_location=c["device"])
                if saved["experiment_key"] != key:
                    raise ValueError(f"Stale checkpoint: {path.name}")
                model.load_state_dict(saved["state_dict"])
                metadata = saved["metadata"]
            else:
                print(f"train {method} seed={seed}", flush=True)
                start = perf_counter()
                history = fit(c, d, model, train, val, seed)
                best = min(history, key=lambda row: row["val_score"])
                metadata = {"method": method, "seed": seed, "best_epoch": best["epoch"],
                            "training_seconds": perf_counter()-start,
                            **{k: v for k, v in best.items() if k.startswith("val_")}}
                write_csv(log, history)
                temporary = path.with_suffix(".tmp")
                torch.save({"experiment_key": key, "state_dict": model.state_dict(), "metadata": metadata}, temporary)
                temporary.replace(path)
                print(f"{method} seed={seed} best epoch {best['epoch']} val {best['val_score']:.5f}", flush=True)
            rows.append(metadata)
            models[method, seed] = model.eval()
    write_csv(output/"validation.csv", rows)
    return models, rows
