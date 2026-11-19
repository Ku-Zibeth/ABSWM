import json
from pathlib import Path

import numpy as np
import pytest
import torch

from abswm.config import fingerprint
from abswm.data import sample_roads
from abswm.decoder_stress import advance_tail, tail_friction
from abswm.decoder_study import CurveDecoder, FrozenPeak, curve_losses, fit, make_training_data, predict_from_history
from abswm.decoder_study import validate_partitions
from abswm.models import WorldModel
from abswm.physics import TYPICAL_ROADS, advance, friction, peak_slip


@pytest.fixture
def study():
    return json.loads((Path(__file__).resolve().parents[1]/"configs"/"decoder_study.json").read_text(encoding="utf-8"))


def example(c, d):
    history = np.zeros((4, 3, 4), dtype=np.float32)
    history[:, :, :2] = [15, 49]
    history[:, :, 2] = [0.0, 0.01, 0.02]
    state = np.repeat(history[:, -1:, :2], 3, axis=1)
    sample = {"history": history, "states": state, "actions": np.full((4, 2), 300),
              "mu": np.full(4, 0.2), "peak": np.array([0.15, 0.12, 0.06, 0.15])}
    roads = [{"theta": params} for params in TYPICAL_ROADS.values()]
    grid = np.linspace(0, 1, d["train_grid_points"])
    return sample, roads, grid


@pytest.mark.parametrize("peak_only", [True, False])
def test_frozen_training_has_no_encoder_gradients_or_weight_changes(config, study, peak_only):
    torch.set_num_threads(1)
    sample, roads, grid = example(config, study)
    train = make_training_data(config, sample, [0, 1, 2, 0], roads, grid)
    pretrained = WorldModel(config)
    model = FrozenPeak(config, pretrained) if peak_only else CurveDecoder(config, study, pretrained)
    initialized_head = [p.detach().clone() for p in (model.head if peak_only else model.decoder).parameters()]
    fit(config, study, model, train, train, epochs=2, seed=27, frozen=True, peak_only=peak_only)
    for before, after in zip(pretrained.encoder.parameters(), model.encoder.parameters()):
        assert not after.requires_grad
        assert after.grad is None
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert any(not torch.equal(p, original) for p, original in
               zip((model.head if peak_only else model.decoder).parameters(), initialized_head))


def test_same_context_queries_multiple_slips_and_respects_bounds(config, study):
    model = CurveDecoder(config, study, WorldModel(config))
    sample, _, _ = example(config, study)
    h = torch.from_numpy(sample["history"])
    z = model.encode(h)
    grid = torch.tensor([0.0, 0.026, 0.13, 0.42, 1.0])
    result = model.curve(z, grid)
    assert result.shape == (len(h), len(grid))
    assert torch.equal(result[:, 0], torch.zeros(len(h)))
    assert bool((result[:, 1:] > 0).all()) and bool((result <= study["curve_bound"]).all())
    assert not torch.equal(result[:, 1], result[:, 3])
    for i, lam in enumerate(grid):
        torch.testing.assert_close(result[:, i], model.friction(lam.expand(len(z)), z))
    peak = model.peak(h, grid_size=101)
    uniform = torch.linspace(0, 1, 101)
    torch.testing.assert_close(peak, uniform[model.curve(z, uniform).argmax(dim=1)])


def test_labels_stay_out_of_inference_and_splits_are_verified(config, study):
    sample, roads, grid = example(config, study)
    first = make_training_data(config, sample, [0, 1, 2, 0], roads, grid)
    changed = {**sample, "peak": sample["peak"][::-1].copy(), "mu": sample["mu"]+0.2}
    second = make_training_data(config, changed, [1, 2, 0, 1], roads, grid)
    assert set(first) == {"history", "actions", "states", "peak", "mu", "target_curve"}
    assert not torch.equal(first["target_curve"], second["target_curve"])
    assert not torch.equal(first["peak"], second["peak"])
    model = CurveDecoder(config, study, WorldModel(config))
    query = torch.as_tensor(grid, dtype=torch.float32)
    torch.testing.assert_close(predict_from_history(model, first["history"], query),
                               predict_from_history(model, second["history"], query), rtol=0, atol=0)
    c = {**config, "roads": {"train": 3, "val": 2, "test": 2}}
    roads = sample_roads(c)
    manifest = {"config_hash": fingerprint(c), "config": c, "roads": roads}
    validate_partitions(c, manifest)
    roads["test"][0] = roads["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_partitions(c, manifest)


def test_joint_loss_backpropagates_to_encoder_and_curve(config, study):
    sample, roads, grid = example(config, study)
    train = make_training_data(config, sample, [0, 1, 2, 0], roads, grid)
    model = CurveDecoder(config, study, WorldModel(config))
    weights = {"state": 0.6, "point": 0.5, "peak": 0.002}
    parts = curve_losses(model, train, torch.as_tensor(grid, dtype=torch.float32), study, weights, True)
    assert all(torch.isfinite(parts[key]) for key in ("total", "state", "point", "curve", "curve_objective", "peak"))
    assert not torch.equal(parts["curve"], parts["curve_objective"])
    parts["total"].backward()
    for module in (model.encoder, model.decoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert any(torch.count_nonzero(p.grad) for p in module.parameters())


def test_tail_stress_preserves_early_curve_peak_and_base_integrator(config):
    theta = np.array(list(TYPICAL_ROADS.values()))
    grid = np.unique(np.r_[np.linspace(0, 1, 1001), peak_slip(theta)])
    base = friction(grid[None], theta[:, None])
    for shift in (-1, 1):
        modified = tail_friction(grid[None], theta[:, None], shift)
        np.testing.assert_array_equal(modified[:, grid <= 0.25], base[:, grid <= 0.25])
        np.testing.assert_array_equal(modified.argmax(axis=1), base.argmax(axis=1))
        assert np.all(shift*(modified[:, -1]-base[:, -1]) > 0)
    state = np.array([[20., 60.], [12., 0.], [15., 50.]])
    torque = np.array([1000, 1900, 100])
    original, _ = advance(state, torque, theta, config)
    np.testing.assert_allclose(advance_tail(state, torque, theta, 0, config), original, atol=1e-12)
