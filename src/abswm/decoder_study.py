"""Models and training for the frozen and jointly trained decoder experiments."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import fingerprint, write_json
from .data import load_split, sample_roads, windows
from .diagnostics import chunks, encode, ridge_predictions, tensor
from .evaluation import load_models, write_csv
from .models import WorldModel, seed_everything
from .physics import friction


class FrozenPeak(nn.Module):
    def __init__(self, c, pretrained):
        super().__init__()
        self.encoder = copy.deepcopy(pretrained.encoder).requires_grad_(False)
        width = c["mlp_dim"]
        self.head = nn.Sequential(nn.Linear(c["hidden_dim"], width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.lo, self.hi = c["reference_min"], c["reference_max"]

    def encode(self, history):
        return self.encoder(history)

    def peak_from_context(self, z):
        return self.lo+(self.hi-self.lo)*torch.sigmoid(self.head(z).squeeze(-1))

    def peak(self, history):
        return self.peak_from_context(self.encode(history))


class CurveDecoder(WorldModel):
    """Use the original fixed-context dynamics with a new, bounded tire head."""

    def __init__(self, c, study, pretrained, original_shape=False):
        super().__init__(c)
        self.encoder.load_state_dict(pretrained.encoder.state_dict())
        self.original_shape = original_shape
        self.slip_scale = study["slip_feature_scale"]
        self.bound = study["curve_bound"]
        if not original_shape:
            width = c["mlp_dim"]
            self.decoder = nn.Sequential(nn.Linear(c["hidden_dim"]+2, width), nn.SiLU(),
                                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def friction(self, lam, z):
        if self.original_shape:
            return super().friction(lam, z)
        lam = lam.clamp(0, 1)
        resolved = lam/(lam+self.slip_scale)
        query = torch.stack((lam, resolved), dim=-1)
        raw = self.decoder(torch.cat((query, z), dim=-1)).squeeze(-1)
        return self.bound*torch.tanh(resolved*F.softplus(raw))

    def curve(self, z, grid):
        query = grid.expand(len(z), -1)
        return self.friction(query, z[:, None, :].expand(-1, len(grid), -1))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_windows(c, base, split):
    data = load_split(base/"data", split)
    offset = {"train": 10, "val": 20, "test": 30}[split]
    batch = windows(data, c, c[f"{split}_windows"], c["data_seed"]+offset)
    if split == "test":
        with np.load(base/"open_loop_windows.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(batch["locators"], archive["locators"])
            np.testing.assert_array_equal(batch["coverage"], archive["coverage"])
    else:
        np.testing.assert_array_equal(batch["locators"], read_json(base/"models"/"sampling.json")[f"{split}_locators"])
    return batch, data["road_index"][batch["locators"][:, 0]]


def validate_partitions(c, manifest):
    if manifest["config_hash"] != fingerprint(c) or manifest["config"] != c:
        raise ValueError("The base configuration and data manifest must match")
    roads = manifest["roads"]
    ids = [road["id"] for split in roads.values() for road in split]
    if len(ids) != len(set(ids)) or roads != sample_roads(c):
        raise ValueError("The original disjoint train/validation/test road partition is required")


def make_training_data(c, sample, road_indices, roads, grid):
    """Training/validation labels only. Road parameters are discarded after label generation."""
    theta = np.array([r["theta"] for r in roads])
    curves = friction(grid[None], theta[:, None])
    values = {k: tensor(sample[k], c) for k in ("history", "actions", "states", "mu", "peak")}
    values["target_curve"] = tensor(curves[road_indices], c)
    return values


def predict_from_history(model, histories, grid):
    return model.curve(model.encode(histories), grid)


def curve_losses(model, batch, grid, study, weights, weighted, *, validation=False):
    z = batch.get("z")
    if z is None:
        z = model.encode(batch["history"])
    predicted = model.curve(z, grid)
    error = (predicted-batch["target_curve"]).square()
    curve = error.mean()
    if weighted:
        near_peak = (grid-batch["peak"][:, None])/study["peak_region_sigma"]
        weight = 1+study["peak_region_gain"]*torch.exp(-near_peak.square()/2)
        objective_curve = ((weight*error).sum(dim=1)/weight.sum(dim=1)).mean()
    else:
        objective_curve = curve
    mu_pred = model.friction(batch["history"][:, -1, 2], z)
    point = ((mu_pred-batch["mu"])/1.5).square().mean()
    winner = grid[predicted.argmax(dim=1)]
    peak_mae = (winner-batch["peak"]).abs().mean()
    alignment = (predicted/study["peak_temperature"]).softmax(dim=1)
    peak = (alignment*((grid-batch["peak"][:, None])/0.1).square()).sum(dim=1).mean()
    state = None
    if weights.get("state", 0) or validation:
        future = model.rollout(batch["history"], batch["actions"])
        state = (((future-batch["states"][:, 1:])/future.new_tensor([1, 10])).square()).mean()
    total = objective_curve
    for name, value in (("state", state), ("point", point), ("peak", peak)):
        if weights.get(name, 0):
            total = total+weights[name]*value
    return {"total": total, "curve": curve, "curve_objective": objective_curve, "state": state,
            "point": point, "peak": peak, "peak_mae": peak_mae}


def peak_losses(model, batch):
    error = model.peak_from_context(batch["z"])-batch["peak"]
    return {"total": (error/0.1).square().mean(), "peak_mae": error.abs().mean()}


def metrics_for_batch(losses, sums, n):
    for name, loss in losses.items():
        if loss is not None:
            sums[name] = sums.get(name, 0)+n*float(loss.detach())


def validation_score(metrics, study, peak_only):
    if peak_only:
        return metrics["peak_mae"]
    components = {"curve_rmse": np.sqrt(metrics["curve"]), "state_rmse": np.sqrt(metrics["state"]),
                  "peak_mae": metrics["peak_mae"]}
    return float(sum(study["validation_score"][name]*value for name, value in components.items()))


def fit(c, study, model, train, val, *, epochs, seed, weights=None, weighted=False, frozen=False, peak_only=False):
    model.encoder.requires_grad_(not frozen)
    grid = torch.linspace(0, 1, study["train_grid_points"], device=c["device"])
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    groups = [{"params": [p for p in model.parameters() if id(p) not in encoder_ids],
               "lr": study["learning_rate_heads" if frozen else "learning_rate_joint"]}]
    if not frozen:
        groups.append({"params": model.encoder.parameters(), "lr": study["learning_rate_encoder"]})
    optimizer = torch.optim.Adam(groups)
    if frozen:
        model.encoder.eval()
        with torch.no_grad():
            train = {**train, "z": encode(model, train["history"], c)}
            val = {**val, "z": encode(model, val["history"], c)}
    keys = ("total", "peak_mae") if peak_only else ("total", "curve", "curve_objective", "state", "point",
                                                   "peak", "peak_mae")
    best_score, best_weights, log = np.inf, None, []
    rng = np.random.default_rng(seed)

    def losses(batch, validation=False):
        if peak_only:
            return peak_losses(model, batch)
        return curve_losses(model, batch, grid, study, weights or {}, weighted, validation=validation)

    for epoch in range(1, epochs+1):
        start = perf_counter()
        model.train()
        if frozen:
            model.encoder.eval()
        order = rng.permutation(len(train["history"]))
        train_totals, val_totals = {}, {}
        for indices in chunks(order, c):
            batch = {k: value[indices] for k, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            parts = losses(batch)
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            metrics_for_batch(parts, train_totals, len(indices))
        model.eval()
        with torch.inference_mode():
            for indices in chunks(np.arange(len(val["history"])), c):
                batch = {k: value[indices] for k, value in val.items()}
                metrics_for_batch(losses(batch, validation=True), val_totals, len(indices))
        n_train, n_val = len(train["history"]), len(val["history"])
        val_metrics = {k: v/n_val for k, v in val_totals.items()}
        score = validation_score(val_metrics, study, peak_only)
        if not np.isfinite(score):
            raise FloatingPointError(f"Nonfinite validation score at epoch {epoch}")
        log.append({"epoch": epoch, "train_windows": n_train, "val_windows": n_val,
                    "val_score": score, "seconds": perf_counter()-start,
                    **{f"train_{k}": train_totals[k]/n_train if k in train_totals else None for k in keys},
                    **{f"val_{k}": val_metrics.get(k) for k in keys}})
        if score < best_score:
            best_score = score
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    return log, best_score


def train_study(c, study, manifest, base, output, experiment_key):
    models = load_models(c, base)
    samples = {split: load_windows(c, base, split) for split in ("train", "val")}
    grid = np.linspace(0, 1, study["train_grid_points"])
    train, val = (make_training_data(c, *samples[split], manifest["roads"][split], grid)
                  for split in ("train", "val"))
    path = output/"models"
    path.mkdir(exist_ok=True)
    for source in (base/"models").glob("*_history.csv"):
        shutil.copyfile(source, path/source.name)
    metadata = []

    def run(name, seed, model, training=train, *, epochs, weights=None, weighted=False, frozen=False, peak=False):
        checkpoint = path/f"{name}_{seed}.pt"
        log_path = path/f"{name}_{seed}_history.csv"
        meta_path = path/f"{name}_{seed}.json"
        if all(p.exists() for p in (checkpoint, log_path, meta_path)):
            saved = torch.load(checkpoint, map_location=c["device"], weights_only=True)
            if saved["experiment_key"] != experiment_key:
                raise ValueError(f"Stale decoder weights in {checkpoint}")
            model.load_state_dict(saved["state_dict"])
            model.encoder.requires_grad_(not frozen)
            info = read_json(meta_path)
        else:
            print(f"train {name} seed={seed} epochs={epochs}", flush=True)
            start = perf_counter()
            log, score = fit(c, study, model, training, val, epochs=epochs, seed=seed, weights=weights,
                             weighted=weighted, frozen=frozen, peak_only=peak)
            info = {"method": name, "seed": seed, "val_score": score, "best_epoch": min(log, key=lambda r: r["val_score"])["epoch"],
                    "epochs": epochs, "training_seconds": perf_counter()-start, "weights": weights or {},
                    "peak_weighted_curve": weighted, "frozen_encoder": frozen}
            write_csv(log_path, log)
            write_json(meta_path, info)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({"experiment_key": experiment_key, "state_dict": model.state_dict()}, temporary)
            temporary.replace(checkpoint)
            print(f"done {name} seed={seed} best={info['best_epoch']} score={score:.5f} "
                  f"seconds={info['training_seconds']:.1f}", flush=True)
        metadata.append(info)
        return model.eval(), info["val_score"]

    def new_curve(seed, original_shape=False):
        seed_everything(seed, c)
        return CurveDecoder(c, study, models["world", seed], original_shape=original_shape).to(c["device"])

    for seed in c["seeds"]:
        seed_everything(seed, c)
        pretrained = models["world", seed]
        models["e2", seed], _ = run("e2", seed, FrozenPeak(c, pretrained).to(c["device"]),
                                    epochs=study["epochs_peak"], frozen=True, peak=True)
        models["e3_original", seed], _ = run("e3_original", seed, new_curve(seed, original_shape=True),
                                             epochs=study["epochs_curve"], frozen=True)
        models["e3", seed], _ = run("e3", seed, new_curve(seed), epochs=study["epochs_curve"], frozen=True)
    seed = study["tuning_seed"]
    indices = np.random.default_rng(seed).permutation(len(train["history"]))[:study["tuning_windows"]]
    pilot = {k: value[indices] for k, value in train.items()}
    selected, weights_by_variant, tuning_rows = {}, {"e4_a": {}}, []
    for variant, name in (("e4_b", "state"), ("e4_c", "point"), ("e4_d", "peak")):
        candidates = []
        for amount in study["weight_candidates"][name]:
            weights = {**selected, name: amount}
            _, score = run(f"pilot_{name}_{amount:g}", seed, copy.deepcopy(models["e3", seed]), pilot,
                           epochs=study["tuning_epochs"], weights=weights)
            candidates.append((score, amount))
        choice = min(candidates)[1]
        tuning_rows.extend({"loss": name, "weight": amount, "val_score": score, "selected": amount == choice}
                           for score, amount in candidates)
        selected = {**selected, name: choice}
        weights_by_variant[variant] = selected
    weights_by_variant["e4_d_weighted"] = selected
    write_csv(output/"weight_tuning.csv", tuning_rows)
    write_json(output/"selected_weights.json", weights_by_variant)
    for seed in c["seeds"]:
        for variant, weights in {"e3_refine": {}, **weights_by_variant}.items():
            study_model = copy.deepcopy(models["e3", seed])
            models[variant, seed], _ = run(variant, seed, study_model, epochs=study["epochs_joint"], weights=weights,
                                           frozen=variant == "e3_refine", weighted=variant == "e4_d_weighted")
    write_json(output/"training_summary.json", metadata)
    return models, samples["train"][0]


class LinearProbe:
    """Same standardized ridge fit as the phase-one linear probe."""

    def __init__(self, model, history, peak, c, penalty):
        self.model = model
        with torch.inference_mode():
            z = encode(model, history, c).cpu().numpy().astype(float)
        self.center, self.scale = z.mean(axis=0), z.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        basis = self.center+self.scale*np.vstack((np.zeros(z.shape[1]), np.eye(z.shape[1])))
        intercept, *columns = ridge_predictions(z, basis, peak, penalty)
        self.intercept = intercept
        self.coefficients = np.asarray(columns)-intercept

    @torch.inference_mode()
    def peak(self, history):
        z = self.model.encode(history).cpu().numpy().astype(float)
        prediction = (z-self.center)/self.scale@self.coefficients+self.intercept
        return torch.as_tensor(prediction, device=history.device)
