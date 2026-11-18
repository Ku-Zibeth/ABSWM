import json
import subprocess
import sys


def test_cli_trains_all_models_and_writes_auditable_report(tmp_path, config):
    small = {**config, "name": "integration-test", "roads": {"train": 3, "val": 1, "test": 1},
             "initial_speeds": [8.], "excitation_kinds": ["ramp"], "data_time": .6,
             "history_steps": 10, "horizon_steps": 4, "seeds": [7], "epochs": 2,
             "warmup_epochs": 1, "train_windows": 16, "val_windows": 8,
             "test_windows": 8, "batch_size": 8, "max_time": 10.}
    path = tmp_path/"config.json"
    path.write_text(json.dumps(small), encoding="utf-8")
    output = tmp_path/"run"
    result = subprocess.run([sys.executable, "-m", "abswm.cli", "run", "--config", str(path),
                             "--output", str(output)], text=True, encoding="utf-8",
                            capture_output=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    for file in ("report.md", "closed_loop.csv", "open_loop.csv", "history_study.csv", "action_study.csv",
                 "paired_comparisons.json", "environment.json", "figures/action_branches.png"):
        assert (output/file).stat().st_size > 0
    assert len(list((output/"models").glob("*.pt"))) == 3
    # Different configuration must never silently reuse an existing run.
    small["data_seed"] += 1
    path.write_text(json.dumps(small), encoding="utf-8")
    refused = subprocess.run([sys.executable, "-m", "abswm.cli", "data", "--config", str(path),
                              "--output", str(output)], text=True, encoding="utf-8", capture_output=True, timeout=30, check=False)
    assert refused.returncode != 0
    assert "another configuration" in refused.stderr
