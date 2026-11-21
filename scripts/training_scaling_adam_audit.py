"""Measure the effect of an Adam restart from an S1 checkpoint with known state."""

import argparse
from pathlib import Path

import torch

from abswm.active_experiment import environment
from abswm.adaptive_experiment import table
from abswm.decoder_study import read_json
from abswm.recurrent_model import registered_reference
from abswm.scaling_data import atomic_torch
from abswm.scaling_training import batches, train_budget
from abswm.training_scaling import REPO


def audit(output, epochs):
    manifest = read_json(output/"experiment.json")
    if manifest["smoke"]:
        raise ValueError("The audit requires the full S1 train and validation windows")
    registry_path = REPO/"configs"/"abswm_2.json"
    registry = read_json(registry_path)
    c, models = registered_reference(registry_path, REPO/registry["base_run"], REPO/registry["reference_run"])
    environment(c["torch_threads"])
    study, windows = manifest["decoder_config"], c["train_windows"]
    steps = batches(windows, c)
    restart_step = study["epochs_joint"]*steps
    targets = [restart_step+extra*steps for extra in epochs]
    rows = []
    for seed, model in models.items():
        s1 = output/"models"/f"S1_{seed}"
        uninterrupted = torch.load(s1/"resume.pt", weights_only=True)
        history = {r["updates"]: r for r in uninterrupted["history"]}
        if not all(step in history for step in targets):
            raise ValueError("Run the matching S1 optimizer continuation first")
        original = torch.load(s1/f"step_{restart_step}.pt", weights_only=True)
        path = output/"models"/f"adam_reset_audit_{seed}"
        if not (path/"resume.pt").exists():
            original["optimizer"]["state"] = {}
            atomic_torch(path/"resume.pt", original)
        job = {"path": path, "arm": "adam_reset_audit", "seed": seed, "windows": windows,
               "updates": targets, "model": model, "c": c, "study": study,
               "train": output/"cache"/"windows_1.pt", "val": output/"cache"/"validation.pt"}
        branch = train_budget(job)
        for step, extra in zip(targets, epochs):
            reset = torch.load(branch[step], weights_only=True)["history"][-1]
            retained = history[step]
            rows.append({"seed": seed, "epochs_after_reset": extra, "updates": step,
                         **{f"{key}_{arm}": state[key] for arm, state in (("reset", reset), ("retained", retained))
                            for key in ("train_total", "val_total", "val_score")}})
    table(output, "adam_reset_audit", rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=REPO/"outputs"/"training_scaling")
    cli.add_argument("--epochs", type=int, nargs="+", default=[1, 4])
    args = cli.parse_args()
    if min(args.epochs) < 1:
        cli.error("Audit epochs must be positive")
    audit(args.output, sorted(set(args.epochs)))
