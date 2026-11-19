"""Physical descriptors and the diagnostic-only P0 experiment."""

from __future__ import annotations

import numpy as np
import torch

from .decoder_evaluation import predictions
from .decoder_study import CurveDecoder, load_windows, read_json
from .diagnostics import rmse, tensor
from .evaluation import load_models, write_csv
from .models import seed_everything
from .physics import friction, peak_slip
from .recurrent_model import RecurrentTireModel, registered_reference
from .recurrent_study import stream_posteriors
from .data import load_split


DESCRIPTORS = ("mu_peak", "peak", "stiffness", "tail_ratio", "width95")


def descriptors(values, grid, peak=None, height=None, stiffness=None):
    peak_index = values.argmax(axis=-1)
    mu = np.take_along_axis(values, peak_index[..., None], axis=-1)[..., 0] if height is None else height
    location = grid[peak_index] if peak is None else peak
    slope = (values[..., 1]-values[..., 0])/(grid[1]-grid[0]) if stiffness is None else stiffness
    width = (values >= 0.95*mu[..., None]).sum(axis=-1)*(grid[1]-grid[0])
    return np.stack((mu, location, slope, values[..., -1]/np.maximum(mu, 1e-8), width), axis=-1)


def road_stats(roads, grid):
    theta = np.array([r["theta"] for r in roads])
    peak = peak_slip(theta)
    return descriptors(friction(grid[None], theta[:, None]), grid, peak=peak, height=friction(peak, theta),
                       stiffness=theta[:, 0]*theta[:, 1]-theta[:, 2])


def bias_row(error):
    positive, negative = error > 1e-6, error < -1e-6
    return {"mae": float(np.abs(error).mean()), "rmse": rmse(error), "mean_bias": float(error.mean()),
            "median_bias": float(np.median(error)), "over_fraction": float(positive.mean()),
            "under_fraction": float(negative.mean()), "over_mean": float(error[positive].mean()) if positive.any() else None,
            "under_mean_abs": float(-error[negative].mean()) if negative.any() else None}


def distribution_plot(output, values, families, scatter):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Peak friction", "Peak slip", "Initial slip stiffness", "Tail ratio", "95% width"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    names = sorted(set(families))
    for j, label in enumerate(labels):
        for family in names:
            axes.flat[j].hist(values[families == family, j], bins=14, alpha=.45, label=family)
        axes.flat[j].set(xlabel=label, ylabel="Training roads")
    axes.flat[0].legend(fontsize=8)
    mat = np.corrcoef(values.T)
    image = axes.flat[-1].imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    short = ["Height", "Slip", "Slope", "Tail", "Width"]
    axes.flat[-1].set(xticks=range(5), yticks=range(5), xticklabels=short, yticklabels=short)
    fig.colorbar(image, ax=axes.flat[-1], shrink=.8)
    fig.savefig(output/"training_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, (i, j) in zip(axes, ((0, 1), (2, 1), (2, 0))):
        for family in names:
            selected = families == family
            ax.scatter(values[selected, i], values[selected, j], s=9, label=family, alpha=.55)
        ax.set(xlabel=labels[i], ylabel=labels[j])
    axes[0].legend(fontsize=7)
    fig.savefig(output/"joint_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, (label, j) in zip(axes, (("Peak slip bias", 1), ("Peak friction bias", 0))):
        ax.boxplot([errors[:, j] for errors in scatter.values()], tick_labels=list(scatter), showfliers=False)
        ax.axhline(0, color="0.6", linewidth=1)
        ax.set(ylabel=label)
        ax.tick_params(axis="x", rotation=18)
    fig.savefig(output/"baseline_bias.png", dpi=150)
    plt.close(fig)


@torch.inference_mode()
def diagnose(registry_path, base, reference, recurrent, output):
    c, references = registered_reference(registry_path, base, reference)
    existing = load_models(c, base)
    data = read_json(base/"data"/"manifest.json")["roads"]
    grid = np.linspace(0, 1, 1001)
    train = road_stats(data["train"], grid)
    families = np.array([r["family"] for r in data["train"]])
    limits = np.array([train.min(axis=0), train.max(axis=0)])
    write_csv(output/"physical_training_roads.csv",
              [{"road": r["id"], "family": r["family"], **dict(zip(DESCRIPTORS, row))}
               for r, row in zip(data["train"], train)])
    groups = {"all": np.ones(len(train), dtype=bool), "hard": families != "snow", "low": families == "snow",
              **{name: families == name for name in set(families)}}
    write_csv(output/"physical_distribution.csv",
              [{"group": name, "feature": feature, "count": int(mask.sum()), "min": train[mask, j].min(),
                "q05": np.quantile(train[mask, j], .05), "median": np.median(train[mask, j]),
                "q95": np.quantile(train[mask, j], .95), "max": train[mask, j].max()}
               for name, mask in groups.items() for j, feature in enumerate(DESCRIPTORS)])
    rho = []
    for name, mask in groups.items():
        corr = np.corrcoef(train[mask].T)
        for i, left in enumerate(DESCRIPTORS):
            for j, right in enumerate(DESCRIPTORS):
                rho.append({"group": name, "feature_1": left, "feature_2": right, "pearson": corr[i, j]})
    write_csv(output/"physical_correlations.csv", rho)
    sample, idx = load_windows(c, base, "test")
    test_roads = [data["test"][i] for i in idx]
    truth = road_stats(test_roads, grid)
    test_family = np.array([r["family"] for r in test_roads])
    observed = sample["coverage"][:, 1]
    groups = {"all": np.ones(len(observed), dtype=bool), "hard": test_family != "snow", "low": test_family == "snow",
              **{name: test_family == name for name in set(test_family)}}
    edges = (0, .02, .05, .1, .2, .5, 1.00001)
    groups.update({f"max_slip_{lo:g}_{hi:g}": (observed >= lo) & (observed < hi) for lo, hi in zip(edges, edges[1:])})
    payload = {"raw": load_split(base/"data", "test"), "locators": sample["locators"], "history": sample["history"]}
    rows, scatters = [], {}
    options = read_json(recurrent/"experiment.json")["options"]
    decoder_options = read_json(reference/"experiment.json")["study_config"]
    history = sample["history"]
    for seed, ref in references.items():
        seed_everything(seed, c)
        world = existing["world", seed]
        e3 = CurveDecoder(c, decoder_options, world).to(c["device"]).eval()
        e3.load_state_dict(torch.load(reference/"models"/f"e3_{seed}.pt", weights_only=True)["state_dict"])
        rec = RecurrentTireModel(ref, options, imagine=True).eval()
        rec.load_state_dict(torch.load(recurrent/"models"/f"recurrent_transition_{seed}.pt", weights_only=True)["state_dict"])
        from .diagnostics import neural_curve
        for name, model in (("World", world), ("E3", e3), ("E4 A", ref), ("Recurrent", rec)):
            if name == "Recurrent":
                z = stream_posteriors(rec, payload, c).z
                values = neural_curve(rec.tire, np.broadcast_to(grid, (len(z), len(grid))), z, c)
                peak = grid[values.argmax(axis=1)]
            else:
                peak, values, _, _ = predictions(c, model, "world" if name == "World" else "e3", history, grid)
            stats = descriptors(values, grid, peak=peak)
            error = stats-truth
            scatters.setdefault(name, []).append(error)
            for group, mask in groups.items():
                if not mask.any():
                    continue
                for j in (0, 1):
                    outside = (stats[mask, j] < limits[0, j]) | (stats[mask, j] > limits[1, j])
                    rows.append({"method": name, "seed": seed, "group": group, "feature": DESCRIPTORS[j],
                                 "windows": int(mask.sum()), "outside_training_min_max": float(outside.mean()),
                                 **bias_row(error[mask, j])})
    write_csv(output/"baseline_bias.csv", rows)
    figs = output/"figures"
    figs.mkdir(exist_ok=True)
    distribution_plot(figs, train, families, {k: np.concatenate(v) for k, v in scatters.items()})
