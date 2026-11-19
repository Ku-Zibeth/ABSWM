"""Matched P1–P5 training and validation, with independent tests sealed off."""

from __future__ import annotations

import copy
from time import perf_counter

import numpy as np
import torch

from .data import load_split, make_history
from .decoder_study import load_windows
from .diagnostics import chunks, tensor
from .evaluation import write_csv
from .models import seed_everything
from .tire_diagnosis import road_stats
from .tire_prior import (PhysicalPrior, StructuredTire, differentiable_descriptors, friction_query, history_evidence,
                         history_residual, range_loss, soft_shape)
from .tire_prior_data import burckhardt, mixed_windows, tire_function


MATRIX = {
    "free_matched": {}, "soft_range": {"range": True}, "prior_peak": {"prior": 2},
    "prior_stiffness": {"prior": 3}, "prior_tail": {"prior": 4},
    "soft_unimodal": {"shape": True}, "soft_smooth": {"smooth": True}, "soft_shape_combined": {"shape": True, "smooth": True},
    "structured_frozen": {"structured": True, "frozen": True}, "structured_joint": {"structured": True},
    "structured_prior": {"structured": True, "prior": 4},
    "history_midpoint": {"structured": True, "history": "midpoint"},
    "history_integral": {"structured": True, "history": "integral"},
    "history_wheel": {"structured": True, "history": "wheel"},
    "history_joint": {"structured": True, "history": "joint"},
    "structured_mixed": {"structured": True, "history": "joint", "mixed": True},
    "structured_recurrent": {"structured": True, "history": "joint", "long": True},
}


def existing_suite(c, d, base, split, roads, count=None):
    sample, road_idx = load_windows(c, base, split)
    raw = load_split(base/"data", split)
    pick = np.arange(len(road_idx))
    if count is not None and count < len(pick):
        pick = np.random.default_rng(d["seed"]+{"train": 1, "val": 2, "test": 3}[split]).choice(pick, count, replace=False)
    return {"history": sample["history"][pick], "roads": [burckhardt(roads[road_idx[i]]) for i in pick],
            "locators": sample["locators"][pick], "raw": raw, "caps": np.zeros(len(pick))}


def histories(suite, steps, c):
    raw = suite["raw"]
    return np.stack([make_history(raw["states"][i], raw["actions"][i], t, steps, c) for i, t in suite["locators"]])


def tensors(c, d, suite, grid):
    fn = tire_function(suite["roads"])
    h = suite["history"]
    long = histories(suite, max(d["refinement_histories"]), c)
    return {"history": tensor(h, c), "long": tensor(long, c),
            "target": tensor(fn(np.broadcast_to(grid, (len(h), len(grid)))), c),
            "point": tensor(fn(h[:, -1, 2]), c)}


def prior_from_training(partitions, d, c):
    train_roads = partitions["train"]
    if any("theta" not in road for road in train_roads):
        raise ValueError("Use the original Burckhardt training-road manifest")
    features = road_stats(train_roads, np.linspace(0, 1, d["eval_grid"]))[:, :4]
    return PhysicalPrior(tensor(features, c), d["prior_bandwidth"]), features


def model_for(ref, variant):
    model = StructuredTire(ref) if variant.get("structured") else copy.deepcopy(ref)
    model.requires_grad_(True)
    model.encoder.requires_grad_(not variant.get("frozen", False))
    return model


def losses(model, variant, data, grid, c, d, prior):
    history = data["long"] if variant.get("long") else data["history"]
    z = model.encode(history)
    values = friction_query(model, z, grid.expand(len(z), -1))
    curve = (values-data["target"]).square().mean()
    extra = {}
    if variant.get("prior") or variant.get("range"):
        physical = model.physical(model.parameters_for(z)) if isinstance(model, StructuredTire) else differentiable_descriptors(values, grid)
        if variant.get("prior"):
            extra["prior"] = (prior.nll(physical, variant["prior"]).mean(), variant["prior_weight"])
        if variant.get("range"):
            extra["range"] = (range_loss(physical, d), d["range_weight"])
    if variant.get("shape") or variant.get("smooth"):
        monotone, smooth = soft_shape(values, grid)
        if variant.get("shape"):
            extra["shape"] = (monotone, d["shape_weight"])
        if variant.get("smooth"):
            extra["smooth"] = (smooth, d["smooth_weight"])
    if variant.get("history"):
        evidence = history_evidence(history, c)
        extra["history"] = (history_residual(model, z, evidence, variant["history"]).mean(), d["history_weight"])
    loss = curve+sum(value*weight for value, weight in extra.values())
    return loss, {"curve": curve, **{name: value for name, (value, _) in extra.items()}}


@torch.inference_mode()
def validation(c, d, model, data, grid, variant):
    h = data["long"] if variant.get("long") else data["history"]
    z = model.encode(h)
    mu = friction_query(model, z, grid.expand(len(z), -1))
    truth = data["target"]
    choice = mu.argmax(dim=1, keepdim=True)
    maximum = truth.amax(dim=1)
    regret = ((maximum-truth.gather(1, choice)[:, 0])/maximum.clamp_min(1e-6)).mean()
    point = (friction_query(model, z, h[:, -1:, 2])[:, 0]-data["point"]).square().mean().sqrt()
    error = (mu-truth).square().mean().sqrt()
    score = error+.35*point+.30*regret
    return {"val_score": float(score), "val_curve_rmse": float(error), "val_point_mu_rmse": float(point),
            "val_regret": float(regret)}


def fit(c, d, model, variant, train, val, grid, prior, seed, epochs):
    groups = [{"params": list((model.head if isinstance(model, StructuredTire) else model.decoder).parameters()),
               "lr": d["learning_rate"]}]
    if not variant.get("frozen"):
        groups.append({"params": model.encoder.parameters(), "lr": d["encoder_learning_rate"]})
    optimizer = torch.optim.Adam(groups)
    rng = np.random.default_rng(seed+d["seed"])
    best, logs = np.inf, []
    for epoch in range(1, epochs+1):
        start = perf_counter()
        model.train()
        totals = {}
        for idx in chunks(rng.permutation(len(train["history"])), d):
            data = {name: value[idx] for name, value in train.items()}
            loss, parts = losses(model, variant, data, grid, c, d, prior)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            for name, value in parts.items():
                totals[name] = totals.get(name, 0)+len(idx)*float(value.detach())
        record = validation(c, d, model.eval(), val, grid, variant)
        logs.append({"epoch": epoch, "seconds": perf_counter()-start, **record,
                     **{f"train_{k}": v/len(train["history"]) for k, v in totals.items()}})
        if record["val_score"] < best:
            best = record["val_score"]
            checkpoint = copy.deepcopy(model.state_dict())
    model.load_state_dict(checkpoint)
    return model.requires_grad_(False), logs


def train(c, d, base, output, references, original, partitions, experiment_key):
    directory = output/"models"
    directory.mkdir(exist_ok=True)
    grid_array = np.linspace(0, 1, d["train_grid"])
    grid = tensor(grid_array, c)
    prior, _ = prior_from_training(original, d, c)
    train_suite = existing_suite(c, d, base, "train", original["train"], d["train_windows"])
    val_suite = existing_suite(c, d, base, "val", original["val"], d["fit_validation_windows"])
    ordinary = tensors(c, d, train_suite, grid_array)
    val = tensors(c, d, val_suite, grid_array)
    mixed = tensors(c, d, mixed_windows(c, partitions["magic_train"], d["train_windows"]//2, d["seed"]), grid_array)
    mixed = {k: torch.cat((v[:d["train_windows"]-len(mixed[k])], mixed[k])) for k, v in ordinary.items()}
    methods, summaries = {}, []

    def run(name, seed, variant, train_data, epochs):
        seed_everything(seed, c)
        model = model_for(references[seed], variant).to(c["device"])
        path = directory/f"{name}_{seed}.pt"
        if path.exists() and path.with_suffix(".csv").exists():
            saved = torch.load(path, weights_only=True, map_location=c["device"])
            if saved["key"] != experiment_key:
                raise ValueError(f"Stale training checkpoint {path.name}")
            model.load_state_dict(saved["state"])
            return model.eval().requires_grad_(False), saved["summary"]
        print(f"train {name} seed={seed}", flush=True)
        start = perf_counter()
        model, log = fit(c, d, model, variant, train_data, val, grid, prior, seed, epochs)
        best = min(log, key=lambda r: r["val_score"])
        summary = {"method": name, "seed": seed, "best_epoch": best["epoch"], "seconds": perf_counter()-start,
                   "prior_weight": variant.get("prior_weight"), **{k: v for k, v in best.items() if k.startswith("val_")}}
        write_csv(path.with_suffix(".csv"), log)
        temp = path.with_suffix(".tmp")
        torch.save({"key": experiment_key, "state": model.state_dict(), "summary": summary}, temp)
        temp.replace(path)
        print(f"done {name} seed={seed} val={best['val_score']:.5f} seconds={summary['seconds']:.1f}", flush=True)
        return model, summary

    pilot = {k: v[:d["pilot_windows"]] for k, v in ordinary.items()}
    first = c["seeds"][0]
    trials = []
    for weight in d["prior_weights"]:
        _, result = run(f"pilot_prior_{weight:g}", first, {"prior": 4, "prior_weight": weight}, pilot, d["pilot_epochs"])
        trials.append(result)
    selected = min(trials, key=lambda r: r["val_score"])["prior_weight"]
    write_csv(output/"prior_pilot.csv", trials)
    for name, variant in MATRIX.items():
        options = {**variant}
        if options.get("prior"):
            options["prior_weight"] = selected
        for seed in c["seeds"]:
            model, row = run(name, seed, options, mixed if options.get("mixed") else ordinary, d["epochs"])
            methods[name, seed] = model
            summaries.append(row)
    write_csv(output/"training_summary.csv", summaries)
    return methods, prior, val_suite, summaries
