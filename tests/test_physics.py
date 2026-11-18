import numpy as np
import pytest

from abswm.physics import TYPICAL_ROADS, advance, advance_scalar, friction, peak_slip


@pytest.mark.parametrize("theta", TYPICAL_ROADS.values())
def test_analytic_peak_matches_dense_search(theta):
    grid = np.linspace(0, 1, 100001)
    assert float(peak_slip(theta)) == pytest.approx(grid[np.argmax(friction(grid, theta))], abs=1e-5)


def test_free_rolling_has_no_artificial_loss(config):
    state = np.array([20.0, 20.0/config["radius"]])
    after, distance = advance(state, 0.0, TYPICAL_ROADS["dry"], config)
    np.testing.assert_allclose(after, state, atol=1e-12)
    assert distance == pytest.approx(20*config["control_dt"])


def test_braking_dissipates_energy_and_release_recovers_wheel(config):
    x = np.array([20.0, 20.0/config["radius"]])
    energies = []
    for _ in range(120):
        energies.append(config["mass"]*x[0]**2/2+config["inertia"]*x[1]**2/2)
        x, _ = advance(x, config["torque_max"], TYPICAL_ROADS["wet"], config)
        assert np.all(x >= 0)
    assert x[1] < 1
    assert np.max(np.diff(energies)) < 1e-6
    released, _ = advance(x, 0, TYPICAL_ROADS["wet"], config)
    assert released[1] > x[1]


@pytest.mark.parametrize("state,torque", [([20., 60.], 1000.), ([5., 0.], 2000.), ([2.1, 1.], 0.)])
def test_scalar_and_batched_integrators_agree(config, state, torque):
    batch, distance = advance(state, torque, TYPICAL_ROADS["dry"], config)
    v, w, ds = advance_scalar(*state, torque, TYPICAL_ROADS["dry"], config)
    np.testing.assert_allclose(batch, [v, w], rtol=1e-12, atol=1e-12)
    assert float(distance) == pytest.approx(ds, abs=1e-12)


def test_step_refinement_changes_states_only_slightly(config):
    refined = {**config, "physics_dt": 0.0005}
    x, y = np.array([20., 60.]), np.array([20., 60.])
    for _ in range(100):
        x, _ = advance(x, 700., TYPICAL_ROADS["dry"], config)
        y, _ = advance(y, 700., TYPICAL_ROADS["dry"], refined)
    np.testing.assert_allclose(x, y, atol=0.002)
