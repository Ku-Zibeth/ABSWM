"""Run P0–P7 in an independent output directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from abswm.config import fingerprint, write_json
from abswm.decoder_study import read_json, validate_partitions
from abswm.recurrent_model import registered_reference
from abswm.tire_diagnosis import diagnose
from abswm.tire_prior_data import manifests
from abswm.tire_prior_training import train


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/abswm_2.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/physics_prior.json"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--recurrent", type=Path, default=Path("outputs/recurrent_study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/physics_prior"))
    parser.add_argument("--stage", choices=("train", "audit", "run"), default="run")
    args = parser.parse_args()
    registry = read_json(args.registry)
    base = (args.base or Path(registry["base_run"])).resolve()
    reference = (args.reference or Path(registry["reference_run"])).resolve()
    recurrent, output = args.recurrent.resolve(), args.output.resolve()
    if any(output == path or path in output.parents for path in (base, reference, recurrent)):
        parser.error("Use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    c, models = registered_reference(args.registry, base, reference)
    d = read_json(args.config)
    physical_manifest = read_json(base/"data"/"manifest.json")
    validate_partitions(c, physical_manifest)
    original = physical_manifest["roads"]
    partitions = manifests(c, d, original)
    source = Path(__file__).resolve().parents[1]/"src"/"abswm"
    provenance = {"config": d, "registry": registry, "base_manifest_sha256": hash_file(base/"data"/"manifest.json"),
                  "reference_sha256": {str(s): hash_file(reference/"models"/f"e4_a_{s}.pt") for s in models},
                  "recurrent_sha256": {str(s): hash_file(recurrent/"models"/f"recurrent_transition_{s}.pt") for s in models},
                  "source": {f: hash_file(source/f) for f in ("tire_prior.py", "tire_prior_training.py", "tire_prior_data.py",
                                                              "tire_diagnosis.py")}, "partition_key": fingerprint(partitions)}
    key = fingerprint(provenance)
    lock = output/"experiment.json"
    if lock.exists() and read_json(lock)["experiment_key"] != key:
        parser.error("Existing results use different inputs, source or experiment settings")
    write_json(lock, {**provenance, "experiment_key": key})
    write_json(output/"frozen_partitions.json", partitions)
    if not (output/"baseline_bias.csv").exists() or not (output/"figures"/"baseline_bias.png").exists():
        diagnose(args.registry, base, reference, recurrent, output)
    fitted, prior, validation, rows = train(c, d, base, output, models, original, partitions, key)
    if args.stage == "run":
        from abswm.tire_prior_evaluation import evaluate
        evaluate(c, d, base, reference, recurrent, output, models, fitted, prior, validation, rows, original, partitions)
    if args.stage in ("run", "audit"):
        from abswm.tire_prior_audit import audit
        audit(c, d, base, reference, recurrent, output, models, fitted, prior, original, partitions)


if __name__ == "__main__":
    main()
