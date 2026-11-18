from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from .config import fingerprint, load_config, write_json


def environment():
    import torch
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("abswm", "numpy", "scipy", "torch", "matplotlib")},
            "cuda_available": torch.cuda.is_available(),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob("*.py"))}}


def main():
    parser = argparse.ArgumentParser(description="ABS and physics-enhanced world-model experiments")
    parser.add_argument("command", choices=("data", "train", "evaluate", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"config.json").exists():
        previous = json.loads((output/"config.json").read_text(encoding="utf-8"))
        if fingerprint(config) != fingerprint(previous):
            parser.error("Output directory belongs to another configuration; select a new --output")
    write_json(output/"config.json", config)
    write_json(output/"environment.json", environment())
    from .data import generate_data
    generate_data(config, output/"data")
    stages = {}
    if args.command in ("train", "run"):
        from .training import train_all
        start = time.perf_counter()
        train_all(config, output)
        stages["training_seconds"] = time.perf_counter()-start
    if args.command in ("evaluate", "run"):
        from .evaluation import evaluate_all
        start = time.perf_counter()
        evaluate_all(config, output)
        stages["evaluation_seconds"] = time.perf_counter()-start
    write_json(output/f"{args.command}_timing.json", stages)


if __name__ == "__main__":
    main()
