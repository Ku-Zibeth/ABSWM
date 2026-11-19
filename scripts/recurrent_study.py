"""Run the ABSWM 2.0 deterministic recurrent forecasting study."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from abswm.config import fingerprint, write_json
from abswm.decoder_study import read_json
from abswm.recurrent_model import registered_reference
from abswm.recurrent_study import inputs, train_recurrent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/abswm_2.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/recurrent_study.json"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/recurrent_study"))
    parser.add_argument("--stage", choices=("train", "run"), default="run")
    args = parser.parse_args()
    registry = read_json(args.registry)
    base = (args.base or Path(registry["base_run"])).resolve()
    reference = (args.reference or Path(registry["reference_run"])).resolve()
    output = args.output.resolve()
    if any(output == source or source in output.parents for source in (base, reference)):
        parser.error("Use a separate output directory")
    archive = output/"experiment.json"
    if output.exists() and not archive.exists() and any(output.iterdir()):
        parser.error("The output directory already contains other results")
    c, references = registered_reference(args.registry, base, reference)
    d = read_json(args.config)
    if not c["history_steps"] <= d["train_history_steps"] or not 0 < d["train_windows"] <= c["train_windows"]:
        parser.error("Invalid training history or window count")
    if (c["horizon_steps"] not in d["forecast_steps"] or not all(1 <= k <= c["horizon_steps"] for k in d["forecast_steps"])
        or not 0 < d["reobserve_step"] < c["horizon_steps"]):
        parser.error("Invalid prediction horizons")
    source = Path(__file__).resolve().parents[1]/"src"/"abswm"
    provenance = {"registry": registry, "base_config": c, "options": d,
                  "reference_run_key": read_json(reference/"experiment.json")["experiment_key"],
                  "reference_sha256": {str(seed): hashlib.sha256(
                      (reference/"models"/f"{registry['reference_model']}_{seed}.pt").read_bytes()).hexdigest()
                      for seed in references},
                  "training_source_sha256": {filename: hashlib.sha256((source/filename).read_bytes()).hexdigest()
                                             for filename in ("recurrent_model.py", "recurrent_study.py")}}
    key = fingerprint(provenance)
    if archive.exists() and read_json(archive)["experiment_key"] != key:
        parser.error("The output directory belongs to other inputs or options")
    write_json(archive, {**provenance, "experiment_key": key})
    roads, (train, val) = inputs(c, d, base)
    models, scores = train_recurrent(c, d, references, train, val, output, key)
    if args.stage == "run":
        from abswm.recurrent_evaluation import evaluate
        evaluate(c, d, base, roads["test"], models, scores, output)


if __name__ == "__main__":
    main()
