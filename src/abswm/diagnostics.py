"""Evaluation-only diagnostics for the existing fixed-context models."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import pdist

from .controllers import LeastSquaresPeak
from .physics import TYPICAL_ROADS, advance, friction, slip


def rmse(error):
    return float(np.sqrt(np.mean(np.square(error)))) if np.size(error) else None


def tensor(x, c):
    if isinstance(x, np.ndarray) and not x.flags.writeable:
        x = x.copy()
    return torch.as_tensor(x, device=c["device"], dtype=torch.float32)


def encode(model, history, c):
    return torch.cat([model.encode(tensor(h, c)) for h in chunks(history, c)])


def chunks(array, c):
    size = c["batch_size"]
    return (array[i:i+size] for i in range(0, len(array), size))


def neural_curve(model, queries, z, c):
    values = []
    for query, context in zip(chunks(queries, c), chunks(z, c)):
        lam = tensor(query, c)
        contexts = context[:, None, :].expand(-1, lam.shape[1], -1)
        values.append(model.friction(lam, contexts).cpu().numpy())
    return np.concatenate(values)


def neural_rollout(model, history, actions, z, c):
    predictions = []
    for h, a, context in zip(chunks(history, c), chunks(actions, c), chunks(z, c)):
        # Bypass only the encoder for the latent intervention. The original rollout
        # processes all actions and still takes the true initial state from h.
        with model.encoder.register_forward_hook(lambda *_: context):
            predictions.append(model.rollout(tensor(h, c), tensor(a, c)).cpu().numpy())
    return np.concatenate(predictions)


def parametric_forecast(initial, actions, theta, c, euler=False):
    x = initial.astype(float)
    result = []
    dt = c["physics_dt"]
    g, radius = c["gravity"], c["radius"]
    for u in actions.T:
        if euler:
            for _ in range(round(c["control_dt"]/dt)):
                mu = friction(slip(x, radius), theta)
                dw = (radius*c["mass"]*g*mu-np.clip(u, 0, c["torque_max"]))/c["inertia"]
                x = np.maximum(x + dt*np.column_stack((-g*mu, dw)), 0)
        else:
            x, _ = advance(x, u, theta, c)
        result.append(x)
    return np.stack(result, axis=1)


def coverage_masks(queries, coverage):
    low, high = coverage[:, 0, None], coverage[:, 1, None]
    inside = (queries >= low) & (queries <= high)
    return {"all": np.ones_like(inside), "inside": inside, "outside": ~inside,
            "below": queries < low, "above": queries > high}


def sample_roads_for(data, roads, samples):
    indices = data["road_index"][samples["locators"][:, 0]]
    labels = [roads[i] for i in indices]
    return np.array([r["id"] for r in labels]), np.array([r["family"] for r in labels]), np.array([
        r["theta"] for r in labels
    ])


@torch.inference_mode()
def forecasts(c, d, models, samples, theta, grid):
    h, actions = samples["history"], samples["actions"]
    theta_grid = theta[:, None, :]
    n = len(h)
    current_slip = h[:, -1:, 2]
    actual = samples["states"][:, 1:].astype(float)
    actual_slip = slip(actual, c["radius"])
    factual_mu = friction(actual_slip, theta_grid)
    curve_queries = np.broadcast_to(grid, (n, len(grid)))
    actual_curve = friction(curve_queries, theta_grid)
    nls_peaks, nls_params = [], []
    for history in h:
        nls = LeastSquaresPeak(c)
        nls_peaks.append(nls(history))
        nls_params.append(nls.theta if nls.theta is not None else TYPICAL_ROADS["wet"])
    force_scale = c["mass"]*c["gravity"]
    fixed_params = np.tile(TYPICAL_ROADS["wet"], (n, 1))
    fixed_peaks = np.full(n, c["fixed_reference"])
    fitted_params = {"fixed": fixed_params, "nls": np.asarray(nls_params)}
    fitted_peaks = {"fixed": fixed_peaks, "nls": np.asarray(nls_peaks)}
    curve_rows, forecast_rows, peak_rows = [], [], []
    families = [("fixed", -1), ("nls", -1), ("oracle_rk4", -1), ("oracle_euler", -1)]
    for method, seed in [*families, *models]:
        keys = {"method": method, "seed": seed}
        model = models.get((method, seed))
        if method == "direct":
            peak = np.concatenate([model.peak(tensor(batch, c)).cpu().numpy() for batch in chunks(h, c)])
        elif model is not None:
            peak = np.concatenate([model.peak(tensor(batch, c), grid_size=len(grid)).cpu().numpy()
                                   for batch in chunks(h, c)])
        else:
            peak = fitted_peaks.get(method, samples["peak"])
        if model is None:
            fitted = fitted_params.get(method, theta)
            predicted = parametric_forecast(h[:, -1, :2], actions, fitted, c, euler=method == "oracle_euler")

            def query(q):
                return friction(q, fitted[:, None, :])
        elif method != "direct":
            z = encode(model, h, c)
            predicted = neural_rollout(model, h, actions, z, c)

            def query(q):
                return neural_curve(model, q, z, c)
        for group, mask in {"all": np.ones(n, dtype=bool), "max_slip_lt_0.05": samples["coverage"][:, 1] < 0.05}.items():
            err = peak[mask]-samples["peak"][mask]
            outside = (peak[mask] < c["reference_min"]) | (peak[mask] > c["reference_max"])
            peak_rows.append({**keys, "group": group, "n": int(mask.sum()),
                              "peak_mae": float(np.abs(err).mean()) if err.size else None,
                              "peak_bias": float(err.mean()) if err.size else None,
                              "clipped_peak_mae": float(np.abs(np.clip(peak[mask], c["reference_min"], c["reference_max"])
                                                            -samples["peak"][mask]).mean()) if err.size else None,
                              "outside_reference_fraction": float(outside.mean()) if err.size else None,
                              "grid_ends_at_one_fraction": float((peak[mask] == 1).mean()) if err.size else None})
        if method == "direct":
            continue
        error = query(curve_queries)-actual_curve
        for region, mask in coverage_masks(curve_queries, samples["coverage"]).items():
            curve_rows.append({**keys, "region": region, "grid_points": int(mask.sum()),
                               "windows": int(mask.any(axis=1).sum()), "mu_rmse": rmse(error[mask]),
                               "fx_rmse_n": rmse(force_scale*error[mask])})
        initial_error = query(current_slip)[:, 0]-samples["mu"]
        forecast_rows.append({**keys, "step": 0, "time_ms": 0, "region": "all", "n": n,
                              "v_rmse_m_s": None, "omega_rmse_rad_s": None, "slip_rmse": None,
                              "mu_rmse": rmse(initial_error), "fx_rmse_n": rmse(force_scale*initial_error),
                              "mu_at_true_slip_rmse": rmse(initial_error)})
        predicted_slip = slip(predicted, c["radius"])
        mu_error = query(predicted_slip)-factual_mu
        oracle_slip_error = query(actual_slip)-factual_mu
        predictions = predicted-actual
        for region, masks in coverage_masks(actual_slip, samples["coverage"]).items():
            for step in d["forecast_steps"]:
                k = step-1
                mask = masks[:, k]
                forecast_rows.append({
                    **keys, "step": step, "time_ms": 1000*step*c["control_dt"], "region": region, "n": int(mask.sum()),
                    "v_rmse_m_s": rmse(predictions[mask, k, 0]), "omega_rmse_rad_s": rmse(predictions[mask, k, 1]),
                    "slip_rmse": rmse(predicted_slip[mask, k]-actual_slip[mask, k]),
                    "mu_rmse": rmse(mu_error[mask, k]), "fx_rmse_n": rmse(force_scale*mu_error[mask, k]),
                    "mu_at_true_slip_rmse": rmse(oracle_slip_error[mask, k]),
                })
    return forecast_rows, curve_rows, peak_rows


def cross_road_permutation(roads, rng):
    _, labels, counts = np.unique(roads, return_inverse=True, return_counts=True)
    shift = counts.max()
    if 2*shift > len(roads):
        raise ValueError("A different-road permutation needs at most half the windows from any one road")
    order = rng.permutation(len(roads))
    grouped = order[np.argsort(labels[order], kind="stable")]
    permutation = np.empty_like(grouped)
    permutation[grouped] = np.roll(grouped, shift)
    return permutation


def ridge_predictions(train, test, targets, penalty):
    center, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-12] = 1
    x = (train-center)/scale
    y = targets.mean(axis=0)
    regularizer = len(x)*penalty*np.eye(x.shape[1])
    beta = np.linalg.solve(x.T@x + regularizer, x.T@(targets-y))
    return (test-center)/scale@beta+y


def probe_targets(c, d, samples, families, theta):
    mu = samples["mu"]
    curves = friction(np.array(d["probe_slips"])[None], theta[:, None])
    names = ["peak", "mu_current", "fx_current_n", "c1", "c2", "c3"]
    names += [f"mu_at_{lam:g}" for lam in d["probe_slips"]]
    family_labels = np.array([families == name for name in TYPICAL_ROADS]).T
    values = np.column_stack((samples["peak"], mu, c["mass"]*c["gravity"]*mu, theta, curves))
    return names, values, family_labels.astype(float)


@torch.inference_mode()
def latent_study(c, d, models, train, test, labels_train, labels_test, grid):
    train_h, test_h = train["history"], test["history"]
    roads, families, theta = labels_test
    _, train_families, train_theta = labels_train
    names, train_y, train_classes = probe_targets(c, d, train, train_families, train_theta)
    _, test_y, test_classes = probe_targets(c, d, test, families, theta)
    rng = np.random.default_rng(d["seed"])
    shuffle = cross_road_permutation(roads, rng)
    true_curve = friction(grid[None], theta[:, None, :])
    grid_batch = np.broadcast_to(grid, true_curve.shape)
    probes, distances, ablations, sample_pca = [], [], [], None
    lengths = sorted({max(2, round(c["history_steps"]*f)) for f in d["history_fractions"]})
    for seed in c["seeds"]:
        model = models["world", seed]
        for length in lengths:
            z_train = encode(model, train_h[:, -length:], c)
            z_test = encode(model, test_h[:, -length:], c)
            a, b = z_train.cpu().numpy().astype(float), z_test.cpu().numpy().astype(float)
            keys = {"seed": seed, "history_steps": length}
            row_train, row_test = train_h[:, -1].astype(float), test_h[:, -1].astype(float)
            features = {"mean": (np.zeros_like(row_train[:, :1]), np.zeros_like(row_test[:, :1])),
                        "current_row": (row_train, row_test), "z": (a, b),
                        "current_row_plus_z": (np.column_stack((row_train, a)), np.column_stack((row_test, b)))}
            for feature, (x_train, x_test) in features.items():
                y_hat = ridge_predictions(x_train, x_test, train_y, d["probe_ridge"])
                cls = ridge_predictions(x_train, x_test, train_classes, d["probe_ridge"]).argmax(axis=1)
                family_indices = test_classes.argmax(axis=1)
                correct = cls == family_indices
                for j, names_j in enumerate(names):
                    error = y_hat[:, j]-test_y[:, j]
                    variance = np.square(test_y[:, j]-test_y[:, j].mean()).sum()
                    probes.append({**keys, "feature": feature, "target": names_j,
                                   "n": len(b), "rmse": rmse(error), "mae": float(np.abs(error).mean()),
                                   "r2": float(1-np.square(error).sum()/variance) if variance else None,
                                   "accuracy": None, "balanced_accuracy": None})
                probes.append({**keys, "feature": feature, "target": "family", "n": len(b),
                               "rmse": None, "mae": None, "r2": None, "accuracy": float(correct.mean()),
                               "balanced_accuracy": float(np.mean([correct[family_indices == j].mean()
                                                                   for j in np.unique(family_indices)]))})
            i, j = np.triu_indices(len(b), 1)
            separation = pdist(b)
            for group, mask in {"same_road": roads[i] == roads[j],
                                "same_family_different_road": (roads[i] != roads[j]) & (families[i] == families[j]),
                                "different_family": families[i] != families[j]}.items():
                distances.append({**keys, "group": group, "pairs": int(mask.sum()),
                                  "z_l2_mean": float(separation[mask].mean()) if mask.any() else None})
            variants = {"intact": z_test, "zero": torch.zeros_like(z_test),
                        "train_mean": z_train.mean(dim=0).expand_as(z_test), "cross_road_shuffle": z_test[shuffle]}
            for variant, z in variants.items():
                curve = neural_curve(model, grid_batch, z, c)
                peak_error = grid[curve.argmax(axis=1)]-test["peak"]
                prediction = neural_rollout(model, test_h[:, -length:], test["actions"], z, c)
                err = prediction-test["states"][:, 1:]
                ablations.append({**keys, "variant": variant, "n": len(z),
                                  "peak_mae": float(np.abs(peak_error).mean()), "curve_mu_rmse": rmse(curve-true_curve),
                                  "endpoint_v_rmse_m_s": rmse(err[:, -1, 0]),
                                  "endpoint_omega_rmse_rad_s": rmse(err[:, -1, 1])})
            if length == c["history_steps"] and seed == c["seeds"][0]:
                center = a.mean(axis=0)
                _, sv, directions = np.linalg.svd(a-center, full_matrices=False)
                sample_pca = {"scores": (b-center)@directions[:2].T,
                              "explained_ratio": np.square(sv[:2])/np.square(sv).sum(),
                              "family": families, "road_id": roads, "coverage_max": test["coverage"][:, 1]}
    return probes, distances, ablations, sample_pca


def model_sizes(models, output):
    rows = []
    for (method, seed), model in models.items():
        size = (output/"models"/f"{method}_{seed}.pt").stat().st_size
        rows.append({"method": method, "seed": seed,
                     "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_bytes": size})
    return rows
