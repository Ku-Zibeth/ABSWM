"""Operator-run ABSWM 3.1 validation, training and independent evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from abswm.active_experiment import environment
from abswm.adaptive_experiment import Paths, seal, setup
from abswm.adaptive_study import STAGES, Study
from abswm.decoder_study import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adaptive_belief_feedback.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--legacy", type=Path, default=Path("outputs/active_probabilistic"))
    parser.add_argument("--registry", type=Path, default=Path("configs/abswm_2.json"))
    parser.add_argument("--recurrent", type=Path, default=Path("outputs/recurrent_study"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/physics_prior"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=("all", *STAGES, "p4", "p5"), default="all")
    args = parser.parse_args()
    stage = {"p4": "train", "p5": "test"}.get(args.stage, args.stage)
    output = args.output or Path("outputs/adaptive_belief_feedback"+("_smoke" if args.smoke else ""))
    registry = read_json(args.registry)
    paths = Paths(args.registry, Path(registry["base_run"]), Path(registry["reference_run"]),
                  args.recurrent, args.prior, args.legacy, output)
    d = read_json(args.config)
    if args.smoke:
        d |= {"constant_roads": {"train": [6, 3, 0], "validation": [3, 1, 0], "test": [3, 2, 1]},
              "closed_constant_test": [1, 1, 1], "dynamic_repeats": {"train": 1, "validation": 1, "test": 1},
              "training_cases_per_distribution": 16, "training_validation_cases": 12, "training_epochs": 1}
    if any(not 0 < n < d["total_horizon"] for n in d["commit_steps"]) or d["revision_screen"]+2 >= d["revision_window"]:
        parser.error("All commitments must leave a continuation, and the revision window must leave an encoder prefix")
    c, old, partitions, key = setup(paths, d, args.smoke)
    environment(c["torch_threads"])
    if args.smoke:
        c = c | {"max_time": 2.5}
    try:
        result = Study(paths, c, d, old, partitions, key, args.workers, args.smoke).run(stage)
    except Exception as exc:
        seal(output/"last_error.json", {"type": type(exc).__name__, "message": str(exc), "stage_requested": stage})
        raise
    print(f"Finished through {stage}. Outputs: {output}", flush=True)
    if "acceptance_passed" in result:
        print(f"Research gate passed: {result['acceptance_passed']}", flush=True)


if __name__ == "__main__":
    main()
