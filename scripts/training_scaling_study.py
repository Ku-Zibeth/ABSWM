"""Run the E4 A scaling study through S2 B3, or extend a qualifying run to B4."""

import argparse
from pathlib import Path

from abswm.training_scaling import REPO, Study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO/"configs"/"training_scaling.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--b4", action="store_true", help="Run B4 if the completed B3 decision qualifies.")
    args = parser.parse_args()
    output = (args.output or REPO/"outputs"/("training_scaling_smoke" if args.smoke else "training_scaling")).resolve()
    if output.exists() and not (output/"experiment.json").is_file() and any(output.iterdir()):
        parser.error("Use an empty output directory or a resumable scaling run")
    Study(args.config.resolve(), output, args.workers, args.smoke).run(b4=args.b4)


if __name__ == "__main__":
    main()
