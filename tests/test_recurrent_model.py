import numpy as np
import torch

from abswm.data import make_history
from abswm.decoder_study import CurveDecoder
from abswm.models import WorldModel
from abswm.physics import TYPICAL_ROADS
from abswm.recurrent_evaluation import acceptance
from abswm.recurrent_model import RecurrentTireModel
from abswm.recurrent_study import CANDIDATES, IMAGINED, OBSERVED, infer, make_cases, slew_limited, stream_posteriors


def build(c, imagine=False):
    torch.set_num_threads(1)
    decoder = CurveDecoder(c, {"slip_feature_scale": 0.04, "curve_bound": 2}, WorldModel(c))
    model = RecurrentTireModel(decoder, {"transition_limit": 0.015, "transition_width": 8}, imagine=imagine)
    return decoder, model


def observations():
    return torch.tensor([[[23., 76., 0.0087, 0.], [22.95, 71., 0.0719, 100.],
                          [22.90, 66., 0.1354, 200.], [22.85, 62., 0.1860, 300.]]])


def test_causal_observer_matches_e4_and_chunked_updates(config):
    reference, model = build(config)
    h = observations()
    posterior, sequence = model.observe(h, return_sequence=True)
    chunked = model.observe(h[:, 2:3], model.observe(h[:, :2]))
    chunked = model.observe(h[:, 3:], chunked)
    torch.testing.assert_close(reference.encode(h), posterior.z, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(chunked.z, posterior.z, atol=1e-7, rtol=1e-6)
    mutated = h.clone()
    mutated[:, 2:] *= 1.7
    _, other = model.observe(mutated, return_sequence=True)
    torch.testing.assert_close(other[:, :2], sequence[:, :2], atol=0, rtol=0)
    assert not torch.equal(other[:, -1], sequence[:, -1])


def test_zero_initialized_transition_matches_e4_preserves_posterior(config):
    reference, model = build(config, imagine=True)
    h = observations()
    memory = model.observe(h)
    z = memory.z.clone()
    low = torch.zeros((1, 4))
    high = torch.full_like(low, 1500.)
    x, mu, context = model.predict(memory, high)
    torch.testing.assert_close(reference.rollout(h, high), x)
    torch.testing.assert_close(context, z[:, None].expand_as(context))
    assert (mu >= 0).all()
    assert model.predict(memory, low)[0][0, -1, 1] > x[0, -1, 1]
    torch.testing.assert_close(memory.z, z, atol=0, rtol=0)
    with torch.no_grad():
        model.transition[0].weight.zero_()
        model.transition[0].bias.zero_()
        model.transition[0].weight[0, -1] = 2.
        model.transition[-1].weight[0, 0] = 1.
    _, _, altered = model.predict(memory, high)
    assert not torch.equal(altered[:, -1], model.predict(memory, low)[2][:, -1])
    maximum_step = torch.cat((z[:, None], altered), dim=1).diff(dim=1).abs().max()
    assert maximum_step <= model.transition_limit+1e-6
    torch.testing.assert_close(memory.z, z, atol=0, rtol=0)


def test_counterfactual_torques_respect_magnitude_and_rate(config):
    u = slew_limited([0., 1950.], [[2250., 2250., -500.], [-50., -50., 2400.]], config)
    assert u.min() >= 0 and u.max() <= config["torque_max"]
    assert np.abs(np.diff(np.column_stack(([0, 1950], u)), axis=1)).max() <= (
        config["torque_rate"]*config["control_dt"]
    )


def test_streamed_anchors_use_full_causal_prefix_and_ignore_labels(config):
    c = {**config, "history_steps": 4, "horizon_steps": 4}
    d = {"train_history_steps": 8, "torque_offset_nm": 600, "batch_size": 3}
    _, model = build(c)
    h = observations()[0].numpy()
    x = np.tile(h[:, :2], (3, 1))[None]
    u = np.arange(11, dtype=np.float32)[None]*75.
    raw = {"states": x, "actions": u, "road_index": np.zeros(1, dtype=int)}
    road = {"id": "example", "theta": TYPICAL_ROADS["wet"]}
    locators = np.array([[0, 2], [0, 7], [0, 3]])
    cases = make_cases(c, d, raw, locators, [road])
    memory = stream_posteriors(model, cases, c)
    for i, (trace, t) in enumerate(locators):
        full = make_history(x[trace], u[trace], t, t+c["history_steps"], c)
        direct = model.observe(torch.from_numpy(full[None]))
        torch.testing.assert_close(memory.z[i], direct.z[0])
    results, force, _ = infer(c, d, model, cases, persistent=True)
    assert results.shape == (len(locators), len(CANDIDATES), c["horizon_steps"], 2)
    changed = {**cases, "states": np.zeros_like(cases["states"]), "mu": np.full_like(cases["mu"], 2)}
    comparison, changed_force, _ = infer(c, d, model, changed, persistent=True)
    np.testing.assert_array_equal(comparison, results)
    np.testing.assert_array_equal(changed_force, force)


def test_acceptance_uses_validation_selection_and_guards_each_factual_metric(config):
    options = {"acceptance": {"max_factual_relative_regression": 0.05, "min_wheel_action_delta_relative_gain": 0.05}}
    shared = {"suite": "matched", "subset": "all", "step": config["horizon_steps"]}
    measurements, action = [], []
    for name, wheel, force in (("e4_a", 10., 1.), ("e4_a_stream", 9.9, 1.),
                               (OBSERVED, 9., 1.06), (IMAGINED, 8., 1.)):
        measurements.append({**shared, "branch": "factual", "method": name, "v_rmse": 1.,
                             "w_rmse": 1., "slip_rmse": 1., "fx_rmse_n": force})
        action.append({**shared, "branch_vs_hold": "all", "method": name, "delta_w_rmse": wheel})
    result = acceptance(options, config, measurements, action,
                        [{"method": OBSERVED, "val_score": 0.8}, {"method": IMAGINED, "val_score": 0.9}])
    assert result["validation_selected_candidate"] == OBSERVED
    assert not result["selected_candidate_passes"]
    assert not result["candidates"][1]["passes_factual_guard"]
    assert result["candidates"][2]["passes"]
