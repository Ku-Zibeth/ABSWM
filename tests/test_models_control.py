import numpy as np
import pytest
import torch

from abswm.controllers import LeastSquaresPeak, SlipPI, simulate_closed
from abswm.models import HistoryEncoder, WorldModel
from abswm.physics import TYPICAL_ROADS, friction, peak_slip


def test_same_context_different_actions_produce_different_futures(config):
    torch.set_num_threads(1)
    model = WorldModel(config)
    h = torch.zeros(2, 5, 4)
    h[:, :, 0], h[:, :, 1], h[:, :, 2] = 20., 60., 0.1
    actions = torch.stack((torch.zeros(5), torch.full((5,), 1500.)))
    result = model.rollout(h, actions)
    assert result.shape == (2, 5, 2)
    assert result[1, -1, 1] < result[0, -1, 1]
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    z = model.encode(h)
    assert torch.equal(model.friction(torch.zeros(2), z), torch.zeros(2))


def test_controller_respects_limits_and_recovers_from_saturation(config):
    controller = SlipPI(config, 6000., 20000.)
    previous = 0.
    for lam in [0.0]*200+[0.95]*100+[0.05]*100:
        torque = controller.step(lam, .12)
        assert 0 <= torque <= config["torque_max"]
        assert abs(torque-previous) <= config["torque_rate"]*config["control_dt"]+1e-8
        previous = torque
    assert controller.torque > 0


def test_nls_recovers_peak_from_causal_consistent_history(config):
    theta = TYPICAL_ROADS["wet"]
    lam = np.linspace(.01, .3, 50)
    # Construct speed increments whose causal midpoint friction is exactly known.
    v = [25.]
    for l in (lam[1:]+lam[:-1])/2:
        v.append(v[-1]-config["gravity"]*friction(l, theta)*config["control_dt"])
    v = np.array(v)
    h = np.column_stack((v, (1-lam)*v/config["radius"], lam, np.zeros(50)))
    estimate = LeastSquaresPeak(config)(h)
    assert estimate == pytest.approx(float(peak_slip(theta)), abs=1e-4)


def test_closed_loop_reports_endpoint_and_complete_accounting(config):
    result, trace = simulate_closed(TYPICAL_ROADS["dry"], 15., config, (3000., 6000.), oracle=True)
    assert result["stopped"]
    assert trace["states"][-1, 0] == config["stop_speed"]
    assert len(trace["states"]) == len(trace["actions"])+1
    assert len(trace["time"]) == len(trace["states"])
    assert result["time_s"] == pytest.approx(trace["time"][-1])
    assert np.max(np.abs(np.diff(np.r_[0, trace["actions"]]))) <= config["torque_rate"]*config["control_dt"]+1e-6


def test_current_only_encoder_cannot_use_previous_observations(config):
    encoder = HistoryEncoder(config, current_only=True)
    h = torch.rand(2, 8, 4)
    h[1, -1] = h[0, -1]
    # This also checks that difference features are computed AFTER truncation.
    encoded = encoder(h)
    torch.testing.assert_close(encoded[0], encoded[1], rtol=0, atol=0)


def test_reference_clipping_does_not_hide_bad_peak_estimation(config):
    c = {**config, "max_time": 1.0}
    truth = float(peak_slip(TYPICAL_ROADS["dry"]))
    result, trace = simulate_closed(TYPICAL_ROADS["dry"], 20., c, (3000., 6000.),
                                    selector=lambda history: 1.0)
    assert result["peak_mae"] == pytest.approx(1.0-truth)
    assert trace["reference"].max() <= c["reference_max"]+1e-12
    assert trace["estimate"].max() == 1.0
