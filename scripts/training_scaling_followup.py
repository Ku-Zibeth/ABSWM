"""Run S3 road diversity and the registered S4 evaluation from completed S1/S2."""

import argparse
from pathlib import Path

from abswm.scaling_followup import run_s3
from abswm.training_scaling import REPO, Study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO/"configs"/"training_scaling.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--stage", choices=("s3", "s4", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = (args.output or REPO/"outputs"/("training_scaling_smoke" if args.smoke else "training_scaling")).resolve()
    study = Study(args.config.resolve(), root, args.workers, args.smoke, read_only=True)
    folder = root/"followup"
    if args.stage == "s4":
        from abswm.decoder_study import read_json

        selection = read_json(folder/"s3_status.json")
    else:
        selection = run_s3(study, folder)
    if args.stage != "s3":
        from abswm.scaling_final import run_s4

        run_s4(study, folder, selection)


if __name__ == "__main__":
    main()
