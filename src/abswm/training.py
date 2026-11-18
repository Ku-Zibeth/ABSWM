from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .config import fingerprint, write_json
from .data import load_split, windows
from .models import build_model, seed_everything

KINDS = ("world", "direct", "current")


def tensor_windows(values, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in values.items() if k in ("history", "actions", "states", "mu", "peak")}


def loss_for(model, batch, kind, horizon):
    if kind == "direct":
        return ((model.peak(batch["history"])-batch["peak"])/0.1).square().mean()
    prediction = model.rollout(batch["history"], batch["actions"][:, :horizon])
    scale = prediction.new_tensor([1.0, 10.0])
    state_loss = ((prediction-batch["states"][:, 1:horizon+1])/scale).square().mean()
    z = model.encode(batch["history"])
    mu = model.friction(batch["history"][:, -1, 2], z)
    force_loss = ((mu-batch["mu"])/1.5).square().mean()
    return state_loss + 0.5*force_loss


def train_one(c, output: Path, kind: str, seed: int, train_data, val_data):
    device = c["device"]
    seed_everything(seed, c)
    model = build_model(kind, c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
    model_path = output / "models" / f"{kind}_{seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        saved = torch.load(model_path, weights_only=True, map_location="cpu")
        if saved["config_hash"] != fingerprint(c):
            raise ValueError("Checkpoint config mismatch")
        if saved.get("completed_epochs") == c["epochs"]:
            print(f"reuse {model_path.name}", flush=True)
            return
    train = tensor_windows(train_data, device)
    val = tensor_windows(val_data, device)
    generator = np.random.default_rng(seed)
    best, history = float("inf"), []
    for epoch in range(c["epochs"]):
        model.train()
        horizon = 1 if epoch < c["warmup_epochs"] else c["horizon_steps"]
        total = 0.0
        order = generator.permutation(len(train_data["history"]))
        for start in range(0, len(order), c["batch_size"]):
            ids = torch.as_tensor(order[start:start+c["batch_size"]], device=device)
            batch = {k: v[ids] for k, v in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, batch, kind, horizon)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {kind} seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        model.eval()
        with torch.inference_mode():
            validation = float(loss_for(model, val, kind, c["horizon_steps"]))
        if not np.isfinite(validation):
            raise FloatingPointError("Non-finite validation loss")
        row = {"epoch": epoch+1, "train_horizon": horizon,
               "train_loss": total/len(order), "val_loss": validation}
        history.append(row)
        if validation < best:
            best = validation
            torch.save({"state_dict": model.state_dict(), "config": c, "config_hash": fingerprint(c),
                        "kind": kind, "seed": seed, "best_epoch": epoch+1,
                        "val_loss": best, "completed_epochs": 0}, model_path)
        print(f"{kind} seed={seed} epoch={epoch+1}/{c['epochs']} "
              f"train={row['train_loss']:.5f} val={validation:.5f}", flush=True)
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    saved["completed_epochs"] = c["epochs"]
    torch.save(saved, model_path)
    log_path = output / "models" / f"{kind}_{seed}_history.csv"
    with log_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_all(c, output: Path):
    train = windows(load_split(output/"data", "train"), c, c["train_windows"], c["data_seed"]+10)
    val = windows(load_split(output/"data", "val"), c, c["val_windows"], c["data_seed"]+20)
    write_json(output/"models"/"sampling.json", {"train_seed": c["data_seed"]+10,
               "val_seed": c["data_seed"]+20, "shared_across_models_and_seeds": True,
               "train_locators": train["locators"].tolist(), "val_locators": val["locators"].tolist()})
    for kind in KINDS:
        for seed in c["seeds"]:
            train_one(c, output, kind, seed, train, val)
