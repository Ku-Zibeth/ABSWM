import numpy as np
import torch

from abswm.controllers import simulate_closed
from abswm.data import make_history
from abswm.diagnostic_experiments import (
    flat_pchip,
    local_condition,
    paired_ramp,
    prefix_before_cap,
    replay_world,
)
from abswm.diagnostics import coverage_masks, cross_road_permutation, neural_rollout, parametric_forecast
from abswm.models import WorldModel
from abswm.physics import TYPICAL_ROADS, friction, slip


def test_cap_excludes_crossing_observation_and_action(config):
    speed, rate, cap = 30.0, 1800.0, 0.02
    a, au = paired_ramp(TYPICAL_ROADS["wet"], speed, rate, config, 0.2)
    b, bu = paired_ramp(TYPICAL_ROADS["dry"], speed, rate, config, 0.2)
    common = min(len(au), len(bu))
    np.testing.assert_array_equal(au[:common], bu[:common])
    for states, actions in ((a, au), (b, bu)):
        lam = slip(states, config["radius"])
        t, crossed = prefix_before_cap(lam, cap)
        assert crossed and lam[t+1] > cap
        assert np.max(lam[:t+1]) <= cap
        history = make_history(states, actions, t, config["history_steps"], config)
        assert history[:, 2].max() <= cap
        assert history[-1, 3] == actions[t-1]
    assert prefix_before_cap([0, 0.01], cap) == (1, False)


def test_diagnostic_coverage_regions_and_local_condition():
    queries = np.array([[0.05, 0.1, 0.2, 0.25]])
    masks = coverage_masks(queries, np.array([[0.1, 0.2]]))
    np.testing.assert_array_equal(masks["inside"], [[False, True, True, False]])
    np.testing.assert_array_equal(masks["outside"], masks["above"] | masks["below"])
    theta = TYPICAL_ROADS["wet"]
    assert local_condition(np.linspace(0.002, 0.02, 50), theta) > local_condition(
        np.linspace(0.002, 0.2, 50), theta
    )
    lam = np.array([0.03, 0.03, 0.09, 0.2])
    values = flat_pchip(lam, friction(lam, theta), np.array([0.0, 0.03, 0.2, 1.0]))
    np.testing.assert_allclose(values, friction([0.0, 0.03, 0.2, 0.2], theta))


def test_latent_shuffle_preserves_multiset_and_excludes_same_road():
    roads = np.array(["a", "b", "b", "c", "c", "a", "a", "d"])
    shuffled = cross_road_permutation(roads, np.random.default_rng(12))
    np.testing.assert_array_equal(np.sort(shuffled), np.arange(len(roads)))
    assert np.all(roads[shuffled] != roads)


def test_oracle_euler_matches_actual_model_integrator(config):
    torch.set_num_threads(1)
    theta = TYPICAL_ROADS["wet"]

    class OracleWorld(WorldModel):
        def friction(self, lam, z):
            a, b, d = theta
            return (a*(-torch.expm1(-b*lam))-d*lam).clamp_min(0)

    model = OracleWorld(config)
    h = np.array([[[20.0, 61.0, 0.085, 0]], [[4.0, 0.0, 1.0, 0]]], dtype=np.float32)
    u = np.array([[300, 1000, 2000], [2000, 0, 0]], dtype=np.float32)
    with torch.inference_mode():
        z = model.encode(torch.from_numpy(h))
        actual = neural_rollout(model, h, u, z, config)
        torch.testing.assert_close(torch.from_numpy(actual), model.rollout(torch.from_numpy(h), torch.from_numpy(u)))
    euler = parametric_forecast(h[:, -1, :2], u, theta, config, euler=True)
    np.testing.assert_allclose(actual, euler, atol=1e-5, rtol=0)


def test_replay_uses_original_estimates_on_update_steps(config):
    c = {**config, "max_time": 1.0, "history_steps": 11, "reference_period": 10}
    theta, gains = TYPICAL_ROADS["wet"], (6000.0, 20000.0)
    samples = iter([0.07, 0.34, 0.14, 0.9] * 10)
    original, trace = simulate_closed(theta, 15.0, c, gains, selector=lambda _: next(samples))
    recovered, replay = replay_world(trace, theta, 15.0, c, gains)
    assert original["distance_m"] == recovered["distance_m"]
    np.testing.assert_array_equal(replay["actions"], trace["actions"])
    np.testing.assert_array_equal(replay["estimate"], trace["estimate"])
