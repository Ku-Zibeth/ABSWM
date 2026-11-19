"""Run E2, E3 and E4 on the unchanged ABSWM base experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from abswm.config import fingerprint, load_config, write_json
from abswm.decoder_study import fit, load_windows, make_training_data, read_json, train_study, validate_partitions
from abswm.evaluation import write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_frozen_control(c, study, manifest, base, output, models, key):
    """Continue the E3 checkpoint with the same decoder LR and epochs as E4 A."""
    weights_path = output/"models"
    local = {**study, "learning_rate_heads": study["learning_rate_joint"]}
    grid = np.linspace(0, 1, study["train_grid_points"])
    data = [make_training_data(c, *load_windows(c, base, split), manifest["roads"][split], grid)
            for split in ("train", "val")]
    rows = []
    control_key = fingerprint({"base": key, "options": local, "script": digest(Path(__file__))})
    for seed in c["seeds"]:
        model = copy.deepcopy(models["e3", seed])
        model.encoder.requires_grad_(False)
        stem = weights_path/f"e3_matched_{seed}"
        path, log_path, meta_path = stem.with_suffix(".pt"), stem.with_name(stem.name+"_history.csv"), stem.with_suffix(".json")
        saved = torch.load(path, map_location=c["device"], weights_only=True) if path.exists() else None
        if saved and saved["experiment_key"] == control_key and log_path.exists() and meta_path.exists():
            model.load_state_dict(saved["state_dict"])
            metadata = read_json(meta_path)
        else:
            print(f"train matched frozen control seed={seed}", flush=True)
            start = perf_counter()
            log, score = fit(c, local, model, *data, epochs=study["epochs_joint"], seed=seed, frozen=True)
            metadata = {"method": "e3_matched", "seed": seed, "val_score": score, "decoder_learning_rate": local["learning_rate_heads"],
                        "best_epoch": min(log, key=lambda r: r["val_score"])["epoch"], "training_seconds": perf_counter()-start,
                        "epochs": study["epochs_joint"], "frozen_encoder": True}
            write_csv(log_path, log)
            write_json(meta_path, metadata)
            temp = path.with_suffix(".tmp")
            torch.save({"experiment_key": control_key, "state_dict": model.state_dict()}, temp)
            temp.replace(path)
        rows.append(metadata)
        models["e3_matched", seed] = model.eval()
    write_json(output/"matched_frozen_training.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/diagnosis_reproduced"))
    parser.add_argument("--config", type=Path, default=Path("configs/decoder_study.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/decoder_study"))
    parser.add_argument("--stage", choices=("train", "run"), default="run")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if output == base or base in output.parents:
        parser.error("Use an independent output directory")
    metadata = output/"experiment.json"
    if output.exists() and not metadata.exists() and any(output.iterdir()):
        parser.error("The output directory already contains other results")
    c, study = load_config(base/"config.json"), read_json(args.config)
    manifest = read_json(base/"data"/"manifest.json")
    validate_partitions(c, manifest)
    source = Path(__file__).resolve().parents[1]/"src"/"abswm"
    provenance = {"base_config": c, "study_config": study,
                  "input_sha256": {p.relative_to(base).as_posix(): digest(p)
                                   for p in [base/"data"/"manifest.json", base/"models"/"sampling.json",
                                             base/"data"/"train.npz", base/"data"/"val.npz",
                                             *sorted((base/"models").glob("*.pt"))]},
                  "training_source_sha256": {f: digest(source/f) for f in
                                             ("decoder_study.py", "models.py", "physics.py", "data.py", "training.py")}}
    key = fingerprint(provenance)
    if metadata.exists() and read_json(metadata)["experiment_key"] != key:
        parser.error("The output belongs to different inputs, training code or options")
    write_json(metadata, {**provenance, "experiment_key": key, "base": str(base)})
    models, train = train_study(c, study, manifest, base, output, key)
    if args.stage == "run":
        from abswm.decoder_evaluation import evaluate_study
        matched_frozen_control(c, study, manifest, base, output, models, key)
        evaluate_study(c, study, manifest, base, output, models, train)


if __name__ == "__main__":
    main()
