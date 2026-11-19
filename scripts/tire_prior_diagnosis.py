"""Run the unchanged-model physical descriptor and peak-bias audit."""

import argparse
from pathlib import Path

from abswm.tire_diagnosis import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/physics_prior"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    diagnose(Path("configs/abswm_2.json"), Path("outputs/diagnosis_reproduced"), Path("outputs/decoder_study"),
             Path("outputs/recurrent_study"), args.output)


if __name__ == "__main__":
    main()
