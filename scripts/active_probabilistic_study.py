"""Run ABSWM 3.0 experiments against the registered ABSWM 2.0 weights."""
from __future__ import annotations

import argparse
from pathlib import Path

from abswm.active_experiment import environment, p0_validate, prepare_experiment
from abswm.decoder_study import read_json


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--config", type=Path, default=Path("configs/active_probabilistic.json"))
    arg.add_argument("--output", type=Path, default=Path("outputs/active_probabilistic"))
    arg.add_argument("--registry", type=Path, default=Path("configs/abswm_2.json"))
    arg.add_argument("--recurrent", type=Path, default=Path("outputs/recurrent_study"))
    arg.add_argument("--prior", type=Path, default=Path("outputs/physics_prior"))
    arg.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    arg.add_argument("--smoke", action="store_true")
    arg.add_argument("--stage", default="all", choices=("p0", "train", "validation", "closed", "open", "audit", "test", "all"))
    args = arg.parse_args()
    if args.smoke and args.output == Path("outputs/active_probabilistic"):
        args.output = Path("outputs/active_probabilistic_smoke")
    registry = read_json(args.registry)
    paths = args.registry, Path(registry["base_run"]), Path(registry["reference_run"]), args.recurrent, args.prior
    c, d, partitions, original = prepare_experiment(args.output, paths, args.config)
    environment(c["torch_threads"])
    if args.stage in {"p0", "all"}:
        p0 = p0_validate(paths, args.output, c, d, partitions, args.workers, smoke=args.smoke)
    else:
        p0 = read_json(args.output/"p0_policy.json")
    if args.stage != "p0":
        if not p0["oracle_sanity_passed"]:
            raise RuntimeError("Oracle-MPC failed the registered P0 checks")
        from abswm.active_training import run_training
        if args.stage in {"train", "all"}:
            run_training(c, d, paths, partitions, args.output, args.workers, smoke=args.smoke)
        if args.stage in {"validation", "closed", "open", "audit", "all", "test"}:
            from abswm.active_evaluation import run_evaluation
            run_evaluation(c, d, paths, partitions, original, p0, args)


if __name__ == "__main__":
    main()
