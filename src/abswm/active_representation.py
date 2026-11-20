"""Oracle curve fits and history-only fits used solely for representation diagnosis."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares

from .active_rollout import GRID
from .active_training import curve_grid, fit_parameters
from .active_data import historical_stress
from .decoder_study import read_json
from .evaluation import write_csv
from .tire_prior import friction_query, history_evidence
from .tire_prior_data import tire_function


def fit_latent(model, initial, objective):
    """Orient a frozen decoder by optimizing its input in isolation."""
    raw = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([raw], lr=.7, max_iter=24, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        guess, target = objective(raw)
        loss = (guess-target).square().mean()+.00012*(raw-initial).square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return raw.detach() if torch.isfinite(raw).all() else initial


def audit_representation(c, models, original, output):
    from pathlib import Path

    name, suite = next(historical_stress(c, original, read_json(Path("configs/physics_prior.json"))))
    chosen = {}
    for i, (road, cap) in enumerate(zip(suite["roads"], suite["caps"])):
        if cap in {.02, .2}:
            chosen.setdefault((road["id"], cap), i)
    rows = []
    grid = torch.as_tensor(GRID[None], dtype=torch.float32)
    for (identifier, cap), i in chosen.items():
        h = torch.as_tensor(suite["history"][i:i+1], dtype=grid.dtype)
        actual = tire_function([suite["roads"][i]])(GRID[None])[0]
        target = torch.tensor(actual[None], dtype=grid.dtype)
        evidence = history_evidence(h, c)
        left = evidence.left[0, evidence.velocity_mask[0]]
        right = evidence.right[0, evidence.velocity_mask[0]]
        points = torch.stack((left, (left+right)/2, right))
        labels = evidence.velocity_mu[0, evidence.velocity_mask[0]]
        weights = torch.tensor([1, 4, 1], dtype=grid.dtype)[:, None]/6

        def interval(estimated):
            return (estimated*weights).sum(dim=0)

        def metrics(mode, family, seed, current, interval_values):
            peak = GRID[int(np.argmax(current))]
            past = np.asarray(interval_values)-labels.numpy()
            return {"road_id": identifier, "cap": cap, "decoder": family, "mode": mode, "seed": seed,
                    "peak": peak, "true_peak": GRID[np.argmax(actual)], "curve_rmse": np.sqrt(np.mean((current-actual)**2)),
                    "tail_rmse": np.sqrt(np.mean((current[GRID > cap]-actual[GRID > cap])**2)),
                    "history_rmse": np.sqrt(np.mean(past**2)) if len(past) else None}

        x0 = np.array([0., -1., -1., .5, -1.4, -.5])
        fitted = fit_parameters(GRID, actual)

        def history_loss(x):
            curve = curve_grid(torch.tensor(x, dtype=points.dtype), points)
            return np.r_[(interval(curve)-labels).numpy(), .006*(x-x0)]

        history_fit = least_squares(history_loss, x0, bounds=(-9, 9)).x if len(labels) else x0
        for mode, raw in (("oracle_shape_parameters", fitted), ("history_shape_parameters", history_fit)):
            z = torch.tensor(raw, dtype=grid.dtype)
            rows.append(metrics(mode, "structured_shape", 0, curve_grid(z, grid).numpy()[0],
                                interval(curve_grid(z, points)).numpy()))
        for family in ("e4", "structured"):
            for seed in c["seeds"]:
                model = models[family, seed]
                with torch.no_grad():
                    z = model.encode(h)

                def estimate_observations(z):
                    return interval(friction_query(model, z, points.reshape(1, -1)).reshape(points.shape))

                hypotheses = [("encoder", z), ("oracle_frozen_latent", fit_latent(
                    model, z, lambda raw: (friction_query(model, raw, grid), target)))]
                if len(labels):
                    hfit = fit_latent(model, z, lambda raw: (estimate_observations(raw), labels))
                    hypotheses.append(("history_frozen_latent", hfit))
                for mode, current in hypotheses:
                    with torch.no_grad():
                        curve = friction_query(model, current, grid)[0].numpy()
                        observations = estimate_observations(current).numpy()
                    rows.append(metrics(mode, family, seed, curve, observations))
    write_csv(output/"representation_diagnostics.csv", rows)
