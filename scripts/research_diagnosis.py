"""Run the phase-one diagnostics from a completed ABSWM run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from abswm.config import fingerprint, load_config, write_json
from abswm.data import load_split, road_id, sample_roads, windows
from abswm.diagnostic_experiments import control_study, distance_pairs, pair_study, pair_uncertainty
from abswm.diagnostics import forecasts, latent_study, model_sizes, sample_roads_for
from abswm.evaluation import load_models, write_csv
from abswm.models import seed_everything
from abswm.physics import friction


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(c, d, run, manifest, output):
    if run == output:
        raise ValueError("Use a separate directory for the diagnostics")
    if manifest["config_hash"] != fingerprint(c) or manifest["config"] != c:
        raise ValueError("Data and run configs must match")
    actual = manifest["roads"]
    if actual != sample_roads(c):
        raise ValueError("The road manifest does not reproduce the configured splits")
    ids = [r["id"] for roads in actual.values() for r in roads]
    if len(ids) != len(set(ids)) or any(r["id"] != road_id(r["theta"]) for roads in actual.values() for r in roads):
        raise ValueError("Road IDs must identify unique parameter combinations across splits")
    pair = set(d["pair_roads"].values())
    if len(pair) != 2 or not pair <= {r["id"] for r in actual["test"]}:
        raise ValueError("Select two distinct pair_roads from the test split")
    if not d["forecast_steps"] or any(step < 1 or step > c["horizon_steps"] for step in d["forecast_steps"]):
        raise ValueError("Forecast steps must lie within the trained horizon")
    if any(not 0 < f <= 1 for f in d["history_fractions"]) or 1 not in d["history_fractions"]:
        raise ValueError("History fractions must be in (0, 1] and include 1")
    if not d["pair_caps"] or any(not 0 < cap < 1 for cap in d["pair_caps"]):
        raise ValueError("Pair caps must be between zero and one")
    if not d["pair_torque_rates"] or min(d["pair_torque_rates"]) <= 0:
        raise ValueError("Pair torque rates must be positive")
    if not d["control_probe_seconds"] or d["grid_steps"] < 2 or d["bootstrap_samples"] < 1 or d["probe_ridge"] <= 0:
        raise ValueError("Control probe times, grid, bootstrap and ridge settings must be nonempty and positive")
    for seconds in d["control_probe_seconds"]:
        step = round(seconds/c["control_dt"])
        if abs(step*c["control_dt"]-seconds) > 1e-9 or step < c["history_steps"] or step % c["reference_period"]:
            raise ValueError("Control probes must occur at estimator update times after warmup")


def provenance(c, d, run, models, replayed_speeds):
    recorded = read_json(run/"environment.json")
    source = Path(__file__).resolve().parents[1]/"src"/"abswm"
    current = {f.name: digest(f) for f in sorted(source.glob("*.py"))}
    differences = [name for name, sha in recorded["source_sha256"].items() if current.get(name) != sha]
    run_paths = ["config.json", "controller.json", "environment.json", "closed_loop.csv",
                 "data/manifest.json", "data/train.npz", "data/val.npz", "data/test.npz", "models/sampling.json",
                 "open_loop_windows.npz"]
    run_paths += [f"models/{method}_{seed}.pt" for method, seed in models]
    timings = {p.name: read_json(p) for p in sorted(run.glob("*_timing.json"))}
    return {"base_run": str(run), "run_config_hash": fingerprint(c), "diagnostic_config": d,
            "source_sha256": current, "script_sha256": digest(Path(__file__)),
            "source_differences_vs_base": differences, "base_timings": timings,
            "base_sha256": {name: digest(run/name) for name in run_paths},
            "world_replay_validated_speeds_m_s": replayed_speeds}


def sampling_audit(splits):
    rows = []
    for name, s in splits.items():
        lo, hi = s["coverage"].T
        rows.append({"split": name, "windows": len(lo), "unique_locators": len(np.unique(s["locators"], axis=0)),
                     "windows_max_slip_lt_0_05": int(np.sum(hi < 0.05)),
                     "windows_max_slip_ge_0_99": int(np.sum(hi >= 0.99)),
                     "windows_current_within_0_02_of_peak": int(np.sum(np.abs(s["history"][:, -1, 2]-s["peak"]) <= 0.02)),
                     "windows_interval_contains_peak": int(np.sum((lo <= s["peak"]) & (s["peak"] <= hi)))})
    return rows


def figs(output, c, d, forecast_rows, grid, curves, pca):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = output/"figures"
    path.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    fields = [("v_rmse_m_s", "Speed RMSE [m/s]"), ("omega_rmse_rad_s", "Wheel speed RMSE [rad/s]"),
              ("slip_rmse", "Slip RMSE"), ("fx_rmse_n", "Tire force RMSE [N]")]
    labels = {"world": "World", "current": "Current", "nls": "NLS", "fixed": "Fixed",
              "oracle_euler": "True friction + Euler"}
    for method, label in labels.items():
        for ax, (field, y_label) in zip(axes.flat, fields):
            rows = [r for r in forecast_rows if r["method"] == method and r["region"] == "all"
                    and r[field] is not None]
            times = sorted({r["time_ms"] for r in rows})
            ax.plot(times, [np.mean([r[field] for r in rows if r["time_ms"] == t]) for t in times],
                    marker=".", label=label)
            ax.set(xlabel="Forecast time [ms]", ylabel=y_label)
    axes[0, 0].legend()
    fig.savefig(path/"rollout_errors.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(d["pair_roads"]), len(d["pair_caps"]), squeeze=False,
                             figsize=(13, 6), layout="constrained", sharex=True, sharey=True)
    names = list(d["pair_roads"])
    for i, j in np.ndindex(axes.shape):
        ax, name, cap = axes[i, j], names[i], d["pair_caps"][j]
        selected = [r for r in curves if r["road"] == name and r["cap"] == cap]
        ax.plot(grid, selected[0]["truth"], "k", label="Plant")
        ax.axvspan(*selected[0]["coverage"], color="0.9", label="Observed interval")
        world_label = f"World ({len(c['seeds'])} seeds)"
        for method, label in (("nls", "NLS"), ("pchip_flat", "PCHIP"), ("world", world_label)):
            group = [r["mu"] for r in selected if r["method"] == method]
            if group:
                h, = ax.plot(grid, np.mean(group, axis=0), label=label)
                if len(group) > 1:
                    ax.fill_between(grid, np.min(group, axis=0), np.max(group, axis=0),
                                    color=h.get_color(), alpha=0.2)
        ax.set(title=f"{name}, cap {cap:.0%}", xlabel="Slip", ylabel="Friction coefficient")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path/"pair_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for name in sorted(set(pca["family"])):
        values = pca["scores"][pca["family"] == name]
        ax.scatter(*values.T, s=9, alpha=0.5, label=name)
    pc1, pc2 = pca["explained_ratio"]
    ax.set(xlabel=f"PC1 ({pc1:.1%} train variance)", ylabel=f"PC2 ({pc2:.1%} train variance)",
           title=f"Test windows, World seed {c['seeds'][0]}")
    ax.legend()
    fig.savefig(path/"latent_pca.png", dpi=160)
    plt.close(fig)


def execute(base_run, diagnostic_config, output):
    start = perf_counter()
    run, output = base_run.resolve(), output.resolve()
    c, d = load_config(run/"config.json"), read_json(diagnostic_config)
    manifest = read_json(run/"data"/"manifest.json")
    validate_inputs(c, d, run, manifest, output)
    seed_everything(c["seeds"][0], c)
    models = load_models(c, run)
    data = {s: load_split(run/"data", s) for s in ("train", "val", "test")}
    samples = {s: windows(data[s], c, c[f"{s}_windows"], c["data_seed"]+offset)
               for s, offset in (("train", 10), ("val", 20), ("test", 30))}
    with np.load(run/"open_loop_windows.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(samples["test"]["locators"], archive["locators"])
        np.testing.assert_array_equal(samples["test"]["coverage"], archive["coverage"])
    sampling = read_json(run/"models"/"sampling.json")
    for split in ("train", "val"):
        np.testing.assert_array_equal(samples[split]["locators"], sampling[f"{split}_locators"])
    labels = {s: sample_roads_for(data[s], manifest["roads"][s], samples[s]) for s in ("train", "test")}
    grid = np.linspace(0, 1, d["grid_steps"])
    roads = {r["id"]: r for r in manifest["roads"]["test"]}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"windows_audit.csv", sampling_audit(samples))
    write_csv(output/"model_sizes.csv", model_sizes(models, run))
    print("Forecasts on base evaluation windows", flush=True)
    metrics, curves, peaks = forecasts(c, d, models, samples["test"], labels["test"][2], grid)
    write_csv(output/"forecast_errors.csv", metrics)
    write_csv(output/"curve_errors.csv", curves)
    write_csv(output/"peak_errors.csv", peaks)
    print("Matched low-slip excitation", flush=True)
    pair_rows, plot_curves = pair_study(c, d, roads, models, grid)
    write_csv(output/"pair_estimates.csv", pair_rows)
    write_csv(output/"pair_seed_spread.csv", pair_uncertainty(pair_rows))
    pair_roads = [roads[road_id] for road_id in d["pair_roads"].values()]
    pair_summary = []
    for cap in d["pair_caps"]:
        query = np.linspace(0, cap, d["grid_steps"])
        a, b = (friction(query, r["theta"]) for r in pair_roads)
        pair_summary.append({"cap": cap, "max_mu_difference": float(np.max(np.abs(a-b)))})
    write_json(output/"pair_truth.json", {"roads": d["pair_roads"], "peaks": [r["peak"] for r in pair_roads],
                                          "overlap": pair_summary})
    print("Latent probes and interventions", flush=True)
    probes, separation, ablations, pca = latent_study(c, d, models, samples["train"], samples["test"],
                                                     labels["train"], labels["test"], grid)
    write_csv(output/"latent_probes.csv", probes)
    write_csv(output/"latent_separation.csv", separation)
    write_csv(output/"latent_ablations.csv", ablations)
    np.savez_compressed(output/"latent_pca.npz", **pca)
    print("PI tracking, delayed oracle and NLS memory", flush=True)
    gains = read_json(run/"controller.json")["selected"]
    control, memory, replay = control_study(c, d, run, roads, (gains["kp"], gains["ki"]))
    write_csv(output/"control_diagnostics.csv", control)
    write_csv(output/"nls_memory.csv", memory)
    write_csv(output/"paired_distance.csv", distance_pairs(c, d, control))
    figs(output, c, d, metrics, grid, plot_curves, pca)
    metadata = provenance(c, d, run, models, replay)
    metadata["diagnostic_seconds"] = perf_counter()-start
    write_json(output/"diagnostic_manifest.json", metadata)
    print(f"Diagnostics: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed original experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/research_diagnosis.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.run, args.config, args.output)


if __name__ == "__main__":
    main()
