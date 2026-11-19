import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from abswm.data import make_history, sample_roads
from abswm.decoder_study import CurveDecoder
from abswm.models import WorldModel
from abswm.physics import TYPICAL_ROADS, advance_scalar
from abswm.tire_prior import (StructuredTire, fixed_forecast, friction_query, history_evidence, history_residual,
                             refine)
from abswm.tire_prior_data import generate_family, manifests, stress_pairs, tire_function
from abswm.tire_prior_training import prior_from_training


@pytest.fixture
def options():
    path = Path(__file__).resolve().parents[1]/"configs"/"physics_prior.json"
    return json.loads(path.read_text(encoding="utf-8"))


def model(c):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    ref = CurveDecoder(c, {"curve_bound": 2, "slip_feature_scale": .04}, WorldModel(c))
    return StructuredTire(ref).eval().requires_grad_(False)


def measured(c, limit=100):
    x, u = [[20., 20./c["radius"]]], []
    theta = TYPICAL_ROADS["wet"]
    for i in range(limit):
        torque = min((i+1)*8, 700)
        v, w, _ = advance_scalar(*x[-1], torque, theta, c)
        u.append(torque)
        x.append([v, w])
    return np.asarray(x), np.asarray(u)


def test_structured_curve_matches_its_physical_outputs(config):
    tire = model(config)
    z = torch.randn(4, config["hidden_dim"])
    shape = tire.parameters_for(z)
    physical = tire.physical(shape)
    lam = torch.linspace(0, 1, 1001).expand(len(z), -1)
    values = friction_query(tire, z, lam)
    torch.testing.assert_close(values[:, 0], torch.zeros_like(shape.height))
    torch.testing.assert_close(tire.friction(shape.peak, z), shape.height)
    torch.testing.assert_close(values[:, -1]/shape.height, shape.tail)
    # A flat neighborhood can tie in float32. Check the unique mathematical
    # peak with a double-precision grid; the public peak uses shape.peak directly.
    double_tire = model(config).double()
    dense = lam.double()
    dense_values = friction_query(double_tire, z.double(), dense)
    torch.testing.assert_close(dense[0, dense_values.argmax(dim=1)], double_tire.parameters_for(z.double()).peak,
                               atol=.001, rtol=0)
    with torch.enable_grad():
        origin = torch.zeros_like(shape.peak).requires_grad_(True)
        gradient, = torch.autograd.grad(tire.friction(origin, z).sum(), origin)
    torch.testing.assert_close(gradient, physical[:, 2], rtol=1e-5, atol=1e-5)
    difference = values.diff(dim=-1)
    assert (difference[lam[:, 1:] <= shape.peak[:, None]] >= -1e-7).all()
    assert (difference[lam[:, :-1] >= shape.peak[:, None]] <= 1e-7).all()


def test_prior_ignores_val_and_test_road_inputs(config, options):
    c = {**config, "roads": {"train": 9, "val": 2, "test": 2}}
    roads = sample_roads(c)
    p, labels = prior_from_training(roads, options, c)
    q, same = prior_from_training({**roads, "test": [{"invalid": True}], "val": []}, options, c)
    np.testing.assert_array_equal(same, labels)
    torch.testing.assert_close(p.samples, q.samples)
    assert np.isfinite(p.nll(torch.tensor(labels, dtype=torch.float32)).numpy()).all()


def test_history_fit_uses_previous_torque_and_no_future(config):
    x, u = measured(config)
    first = make_history(x, u, 60, 50, config)
    x[61:] = np.nan
    u[60:] = 1999
    replay = make_history(x, u, 60, 50, config)
    np.testing.assert_array_equal(first, replay)
    evidence = history_evidence(torch.from_numpy(first[None]), config)
    mask = evidence.wheel_mask & evidence.velocity_mask
    assert mask.any()
    torch.testing.assert_close(evidence.wheel_mu[mask], evidence.velocity_mu[mask], atol=1e-4, rtol=0)
    locked = torch.from_numpy(first[None]).clone()
    locked[..., 1] = 0
    assert not history_evidence(locked, config).wheel_mask.any()


def test_refinement_leaves_model_weights_fixed_and_uses_improved_residual(config, options):
    tire = model(config)
    x, u = measured(config)
    history = torch.tensor(make_history(x, u, 65, 50, config)[None])
    evidence = history_evidence(history, config)
    assert evidence.usable.all()
    z = tire.encode(history)
    snapshot = {k: v.clone() for k, v in tire.state_dict().items()}
    result = refine(tire, z, evidence, options, [0, 1, 3, 10])
    adjusted = result[10][0]
    assert result[10][1].item() <= result[0][1].item()+1e-8
    assert (adjusted-z).norm().item() <= options["refinement_radius"]+1e-6
    assert (adjusted-z).norm().item() > 0
    for key, value in tire.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
    assert all(p.grad is None for p in tire.parameters())
    forecasts, _ = fixed_forecast(tire, adjusted, history[:, -1, :2], torch.tensor([[200., 300., 400.]]), config)
    first, _ = fixed_forecast(tire, adjusted, history[:, -1, :2], torch.tensor([[200., 300., 0.]]), config)
    torch.testing.assert_close(forecasts[:, :2], first[:, :2])


def test_refinement_rejects_nonfinite_proposals_and_handles_weak_evidence(config, options):
    class Reject(nn.Module):
        def friction(self, lam, z):
            candidate = .8+z[..., 0]
            return torch.where(z[..., 0] == 0, candidate, torch.full_like(candidate, float("nan")))

    raw, u = measured(config)
    history = torch.tensor(make_history(raw, u, 60, 50, config)[None])
    z = torch.zeros((1, 2))
    evidence = history_evidence(history, config)
    result = refine(Reject(), z, evidence, options, [3])[3]
    torch.testing.assert_close(result[0], z, rtol=0, atol=0)
    assert result[2].item() == 3
    weak = history_evidence(torch.zeros_like(history), config)
    assert not weak.usable.any()
    warm = refine(model(config), torch.zeros(1, 32), weak, options, [3], warm=torch.full((1, 32), .02))[3]
    torch.testing.assert_close(warm[0], torch.full((1, 32), .02))


def test_independent_families_are_disjoint_and_stress_pairs_decouple_peak(config, options):
    c = {**config, "roads": {"train": 9, "val": 2, "test": 60}}
    original = sample_roads(c)
    d = {**options, "ab_ids": [r["id"] for r in original["test"][:2]], "magic_train_roads": 3, "magic_test_roads": 3}
    partitions = manifests(c, d, original)
    isolated = set(r["id"] for r in partitions["independent_magic"])
    assert isolated.isdisjoint(r["id"] for r in partitions["magic_train"])
    magic = generate_family("magic", 5, 128)
    values = tire_function(magic)(np.broadcast_to(np.linspace(0, 1, 501), (len(magic), 501)))
    assert np.all(values >= 0)
    assert np.all((values.argmax(axis=1) > 35) & (values.argmax(axis=1) < 150))
    pair = stress_pairs()[:2]
    grid = np.linspace(0, 1, 1001)
    mu = tire_function(pair)(np.broadcast_to(grid, (2, len(grid))))
    np.testing.assert_allclose(mu.max(axis=1), [.83, .83], atol=1e-6)
    np.testing.assert_allclose(grid[mu.argmax(axis=1)], [.11, .21], atol=.001)
