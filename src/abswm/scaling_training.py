"""Resume the joint E4 A objective at exact optimizer-step budgets."""
from __future__ import annotations

from copy import deepcopy
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .active_experiment import environment
from .decoder_study import curve_losses, metrics_for_batch, validation_score
from .models import seed_everything
from .scaling_data import atomic_torch


def batches(windows, c):
    return ceil(windows/c["batch_size"])


def validation_losses(model, data, c, study, grid):
    totals = {}
    n = len(data["history"])
    model.eval()
    with torch.inference_mode():
        for start in range(0, n, c["batch_size"]):
            sample = {k: v[start:start+c["batch_size"]] for k, v in data.items()}
            losses = curve_losses(model, sample, grid, study, {}, False, validation=True)
            metrics_for_batch(losses, totals, len(sample["history"]))
    metrics = {k: v/n for k, v in totals.items()}
    return {"val_score": validation_score(metrics, study, False),
            **{f"val_{key}": value for key, value in metrics.items()}}


def train_budget(job):
    c, study, seed, windows = (job[k] for k in ("c", "study", "seed", "windows"))
    base_path = Path(job["path"])
    model = deepcopy(job["model"])
    targets = sorted(set(job["updates"]))
    outputs = {step: base_path/f"step_{step}.pt" for step in targets}
    if all(p.exists() for p in outputs.values()):
        return {step: str(p) for step, p in outputs.items()}
    resume = base_path/"resume.pt"
    base_path.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, c)
    model.requires_grad_(True)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": study["learning_rate_joint"]},
        {"params": model.encoder.parameters(), "lr": study["learning_rate_encoder"]},
    ])
    rng = np.random.default_rng(seed)
    grid = torch.linspace(0, 1, study["train_grid_points"], device=c["device"])
    train = torch.load(job["train"], weights_only=True, map_location=c["device"])
    val = torch.load(job["val"], weights_only=True, map_location=c["device"])
    n_batches = batches(windows, c)
    done, count, training_s, validation_s = 0, 0, 0., 0.
    totals, history = {}, []
    order = None
    if resume.exists():
        saved = torch.load(resume, weights_only=True, map_location=c["device"])
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
        done, count, training_s, validation_s = (saved[k] for k in ("updates", "samples", "training_s", "validation_s"))
        totals, history, order = (saved[k] for k in ("totals", "history", "order"))
    if done in outputs and not outputs[done].exists():
        atomic_torch(outputs[done], saved)
    if any(step < done and not p.exists() for step, p in outputs.items()):
        raise ValueError(f"A requested past milestone is missing in {base_path}")
    print(f"{job['arm']} seed={seed} windows={windows} steps={done}->{targets[-1]}", flush=True)
    while done < targets[-1]:
        mark = perf_counter()
        number = done % n_batches
        if number == 0:
            order = torch.from_numpy(rng.permutation(windows)).to(c["device"])
            totals = {}
            model.train()
        elif not model.training:
            model.train()
        start = number*c["batch_size"]
        ii = order[start:start+c["batch_size"]]
        sample = {k: v[ii] for k, v in train.items()}
        optimizer.zero_grad(set_to_none=True)
        losses = curve_losses(model, sample, grid, study, {}, False)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Nonfinite loss at step {done+1} in {base_path}")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        metrics_for_batch(losses, totals, len(ii))
        count += len(ii)
        done += 1
        training_s += perf_counter()-mark
        if done % n_batches == 0 or done in outputs:
            epoch = done/n_batches
            seen_epoch = min((number+1)*c["batch_size"], windows)
            mark = perf_counter()
            metrics = validation_losses(model, val, c, study, grid)
            validation_s += perf_counter()-mark
            if not np.isfinite(metrics["val_score"]):
                raise FloatingPointError(f"Nonfinite validation score at step {done} in {base_path}")
            history.append({"epoch": epoch, "updates": done, "samples_seen": count, "training_s": training_s,
                            "validation_s": validation_s, "train_windows": windows, "val_windows": len(val["history"]),
                            **{f"train_{k}": v/seen_epoch for k, v in totals.items()}, **metrics})
            payload = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": rng.bit_generator.state, "order": order, "totals": totals, "history": history,
                       "updates": done, "samples": count, "training_s": training_s, "validation_s": validation_s}
            if done in outputs:
                atomic_torch(outputs[done], payload)
                print(f"{job['arm']} seed={seed} milestone={done} val={metrics['val_score']:.6f}", flush=True)
            atomic_torch(resume, payload)
    return {step: str(p) for step, p in outputs.items()}
