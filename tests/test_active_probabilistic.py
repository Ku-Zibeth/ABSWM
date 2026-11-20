from pathlib import Path

import numpy as np
import pytest
import torch

from abswm.active_belief import BeliefObserver, correction, information_step, normalized
from abswm.active_control import PI, PredictiveController, Supervisor, simulate
from abswm.active_data import branches, frozen_partitions, history_windows
from abswm.active_experiment import controller_for
from abswm.active_fast import oracle_forecast, static_forecast
from abswm.active_information import entropy, information_values, likelihood_posterior
from abswm.active_metrics import mixture_scores
from abswm.active_rollout import GRID, OraclePredictor, candidate_actions, curve_forecast, lower_cvar
from abswm.active_training import ProbabilisticTire, curve_grid, losses, train_data
from abswm.config import load_config
from abswm.data import sample_roads
from abswm.decoder_study import read_json
from abswm.physics import TYPICAL_ROADS, friction
from abswm.tire_prior_data import burckhardt, generate_family


@pytest.fixture
def config():
    return load_config("configs/full.json")


@pytest.fixture
def study():
    return read_json(Path("configs/active_probabilistic.json"))


@pytest.fixture
def road():
    return burckhardt({"id": "test-road", "family": "wet", "theta": TYPICAL_ROADS["wet"]})


def history(c, lam=.0, torque=0.):
    row = [15., 15.*(1-lam)/c["radius"], lam, torque]
    return np.repeat(np.array(row)[None], c["history_steps"], axis=0)


def test_probability_normalization_and_stable_update():
    assert np.allclose(normalized([1e25, 3e25]), [.25, .75])
    with pytest.raises(ValueError, match="nonnegative"):
        normalized([-1., 2.])
    w = [.4, .6]
    expected = np.array(w)*np.exp(-.5*np.array([.5, 1.5])**2)
    assert np.allclose(likelihood_posterior(w, np.array([.5, 1.5]), 1.), expected/expected.sum())
    assert np.allclose(likelihood_posterior(w, np.array([800., 801.]), 1.), [1., 0.])


def test_evidence_is_consumed_once_and_weak_history_retains_hypotheses(config):
    observer = BeliefObserver(config, None, .03)
    h0 = history(config)
    observer.observe(h0, 15)
    p0 = observer.weights.copy()
    observer.observe(h0, 16)
    assert observer.updates == 0
    assert np.array_equal(observer.weights, p0)
    h1 = h0.copy()
    h1[-1] = [14.995, 47., .06, 400.]
    observer.observe(h1, 17)
    one = observer.weights.copy()
    observer.observe(h1, 17)
    assert observer.updates == 1
    assert np.array_equal(observer.weights, one)
    with pytest.raises(ValueError, match="Conflicting"):
        observer.observe(h0, 17)
    with pytest.raises(ValueError, match="successive"):
        observer.observe(h1, 19)


def test_mismatch_and_degeneracy_rejuvenation(config):
    observer = BeliefObserver(config, None, .025)
    observer.observe(history(config), 0)
    for tick in range(1, 4):
        event = observer.assimilate(np.array([.5, .5, .5, .5, .5, .5]), 1.8, tick)
    assert event.mismatch and event.refreshed and not observer.trusted
    observer.weights = np.array([1., 0., 0., 0., 0., 0.])
    observer.bad_run = 0
    for tick in range(4, 7):
        event = observer.assimilate(np.zeros(6), 0., tick)
    assert event.degenerate and event.refreshed and observer.trusted
    assert np.all(observer.weights > 0) and observer.weights.sum() == pytest.approx(1)


def test_information_projection_preserves_nullspace_and_network(config, study):
    delta, eigenvalues, keep = information_step(np.array([.02, .03]), np.array([[1., 0], [1., 0]]),
                                                 np.ones(2), .05)
    assert keep == 1 and eigenvalues[0] == 0 and delta[0] > 0 and delta[1] == 0
    model = ProbabilisticTire(config, study).requires_grad_(False)
    weights = {name: p.clone() for name, p in model.state_dict().items()}
    raw = model(torch.tensor(history(config)[None]).float())[0][0].detach()
    xs = np.linspace(.02, .18, 12)
    target = curve_grid(raw[0]+.2, torch.from_numpy(xs)).numpy()
    update, report = correction(raw, model.latent_scale.numpy(), np.column_stack((xs, xs, target)), .035, "subspace")
    assert report["after"] <= report["before"]
    assert torch.isfinite(update).all()
    assert all(torch.equal(weights[name], v) for name, v in model.state_dict().items())


def test_bayes_and_refinement_use_distinct_intervals(config, study):
    model = ProbabilisticTire(config, study).requires_grad_(False)
    observer = BeliefObserver(config, model, .05, refinement="subspace")
    h = history(config, .09, 550.)
    observer.observe(h, 0)
    for tick in range(1, 7):
        h = h.copy()
        h[-1, 0] -= .02
        observer.observe(h, tick)
    assert observer.updates == 3 and len(observer.pending) == 3


def test_virtual_rollout_does_not_change_observer_or_reassign_hypotheses(config):
    observer = BeliefObserver(config, None, .04)
    h = history(config, .05, 600.)
    observer.observe(h, 0)
    prepared = candidate_actions(600, 8, config)[0]
    weights, curves, tick = observer.weights.copy(), observer.curves.copy(), observer.tick
    all_actions = observer.predict(h[-1, :2], prepared)
    reordered = observer.predict(h[-1, :2], prepared[::-1])
    assert np.array_equal(all_actions.state, reordered.state[:, ::-1])
    assert observer.tick == tick and np.array_equal(observer.weights, weights)
    assert np.array_equal(observer.curves, curves)


def test_information_gain_matches_symmetry_and_task_disagreement():
    p = np.array([.5, .5])
    x = np.array([[0., 0.], [0., 8.]])
    score = np.eye(2)
    h, v = information_values(x, p, score, .1)
    assert h[0] < 1e-12 and v[0] < 1e-12
    assert h[1] == pytest.approx(entropy(p)) and v[1] == pytest.approx(.5)
    _, v = information_values(x, p, np.array([[2., 1.], [4., 2.]]), .1)
    assert np.max(v) < 1e-12
    assert lower_cvar(np.array([[.0, 1.], [1., .0]]), np.array([.25, .75]), .5) == pytest.approx([.5, 0.])


def test_candidate_slew_and_first_action_only(config, study, road):
    ctrl = PredictiveController(config, study["options"] | {"horizon": 10, "slew_nm": 100.},
                                OraclePredictor(config, road), information="decision", gamma=0.)
    base = PredictiveController(config, ctrl.options, OraclePredictor(config, road))
    h = history(config)
    baseline, active = base.decide(h, 0), ctrl.decide(h, 0)
    assert baseline == active
    actions = candidate_actions([0., 400., 1950.], 12, config, random=np.random.default_rng(3))
    moves = np.diff(actions, axis=-1, prepend=np.broadcast_to(np.array([0., 400., 1950.])[:, None, None],
                                                            (*actions.shape[:2], 1)))
    assert np.max(np.abs(moves)) <= 100.00001
    assert actions.min() == 0 and actions.max() == 2000
    metrics, trace, _ = simulate(road, config | {"max_time": .01}, ctrl, 15., keep_trace=True)
    assert metrics["steps"] == 2 and trace[0]["applied"] == active.requested
    assert ctrl.predictor.weights.tolist() == [1.]


def test_safety_limits_and_failure_paths(config):
    actual = [15., 60.]
    out, reason = Supervisor(config).apply(actual, .1, 500., float("nan"))
    assert out == 400. and reason == "unavailable_action"
    out, reason = Supervisor(config).apply(actual, .1, 500., 700., False)
    assert 400 <= out <= 600 and reason == "untrusted_prediction"
    supervisor = Supervisor(config)
    assert supervisor.apply([15., 0.], 1., 500., 700.) == (400., "locked")
    assert supervisor.apply([15., 25.], .5, 400., 700.) == (300., "high_slip")
    assert supervisor.apply([2., 5.], .0, 300., 700.) == (200., "low_speed")
    assert supervisor.apply([np.nan, 1.], .0, 200., 700.) == (100., "invalid_state")


def test_oracle_isolation_and_forecast_equivalence(config, study, road):
    # C5 factory receives no road information that its predictor retains.
    options = study["options"] | {"horizon": 10, "slew_nm": 100.}
    nls_a = controller_for("C5", 0, road, config, options, {}, (6000, 20000))
    nls_b = controller_for("C5", 0, {}, config, options, {}, (6000, 20000))
    h = history(config, .08, 800.)
    assert nls_a.decide(h, 0) == nls_b.decide(h, 0)
    actions = candidate_actions(800, 10, config)[0]
    curves = np.stack([friction(GRID, p) for p in TYPICAL_ROADS.values()])
    fast = static_forecast(h[-1, :2], actions, curves, config)
    slow = curve_forecast(h[-1, :2], actions, curves, config)
    assert np.allclose(fast.state, slow.state, atol=1e-10)
    for target in [road, *[generate_family(kind, 1, 515)[0] for kind in ("magic", "rational")]]:
        predictor = OraclePredictor(config, target)
        exact = predictor.predict(h[-1, :2], actions)
        compiled = oracle_forecast(predictor, h[-1, :2], actions, target)
        assert np.allclose(exact.state, compiled.state, atol=1e-10)
        assert np.array_equal(exact.high, compiled.high)


def test_frozen_split_and_action_labels_are_paired(config, study):
    original = sample_roads(config | {"roads": {"train": 6, "val": 3, "test": 3}})
    d = study | {"training_steps": 8}
    frozen = frozen_partitions(config, d, original)
    all_roads = [r for roads in frozen.values() for r in roads]
    assert len(all_roads) == len({r["id"] for r in all_roads})
    assert all(r["family"] != "rational" for name, roads in frozen.items() if name.startswith("train") for r in roads)
    with pytest.raises(ValueError, match="held out"):
        frozen_partitions(config, d, original, {"historical": frozen["test_magic"]})
    suite = history_windows(config, d, frozen["train_burckhardt"], 14, 21)
    data = branches(config, d, suite, random_seed=22)
    assert np.array_equal(data["history"], suite["history"])
    assert data["actions"].shape[:2] == data["state"].shape[:2]
    assert np.all(data["actions"][:, 0] == suite["history"][:, -1, 3, None])
    model = ProbabilisticTire(config, d)
    tensor = train_data(config, d, suite, 22, training=True)
    obj = losses(model, {k: v[:3] for k, v in tensor.items()}, config, d, ("action", "ranking", "risk", "probability"))
    obj["loss"].backward()
    assert torch.isfinite(model.centers.grad).all()


def test_control_reproducibility_and_gaussian_probability_score(config, road):
    result, trace, _ = simulate(road, config, PI(config, (6000, 20000)), 15., keep_trace=True)
    duplicate, copied_trace, _ = simulate(road, config, PI(config, (6000, 20000)), 15., keep_trace=True)
    assert result == duplicate and result["stopped"] and result["end_speed"] == 2.
    assert [r["applied"] for r in trace] == [r["applied"] for r in copied_trace]
    nll, crps, inside, width = mixture_scores(np.zeros((1, 1)), np.zeros(1), np.ones(1), np.ones(1))
    assert nll.item() == pytest.approx(.5*np.log(2*np.pi))
    assert crps.item() == pytest.approx((np.sqrt(2)-1)/np.sqrt(np.pi))
    assert inside.all() and np.all(np.diff(width[:, 0]) > 0)
