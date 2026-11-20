import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from abswm.active_belief import information_step
from abswm.active_control import Decision, Supervisor
from abswm.active_rollout import GRID, candidate_actions, grid_coefficient, integrate
from abswm.active_training import ProbabilisticTire
from abswm.adaptive_belief import AdaptiveObserver, ContinuousTire, Innovation, TireBank, update_weights
from abswm.adaptive_simulation import AUDIT_HORIZONS, SpatialOracle, drive
from abswm.adaptive_training import gather_windows
from abswm.decoder_study import read_json
from abswm.feedback_control import CommittedController, SegmentDecision, feedback_plan, observation_branches
from abswm.physics import TYPICAL_ROADS
from abswm.spatial_road import SpatialRoad, make_partitions, scene
from abswm.tire_prior_data import burckhardt


@pytest.fixture
def settings():
    return read_json(Path("configs/adaptive_belief_feedback.json"))


@pytest.fixture
def options():
    return read_json(Path("configs/active_probabilistic.json"))["options"] | {"slew_nm": 100.}


def road(name):
    return burckhardt({"id": name, "family": name, "theta": TYPICAL_ROADS[name]})


def history(config, velocity=15., lam=.1, torque=500.):
    row = [velocity, velocity*(1-lam)/config["radius"], lam, torque]
    return np.repeat(np.array([row]), config["history_steps"], axis=0)


class TestSource:
    __test__ = False
    model = None

    def __init__(self):
        self.calls = []

    def make(self, context, generation):
        self.calls.append((context.copy(), generation))
        values = np.array([.75, 1.0]) if generation else np.array([.01, .03])
        curves = values[:, None]*np.minimum(1., 30*GRID)
        return TireBank(curves, np.array([.5, .5]), None, ((generation, 0), (generation, 1)))


def test_innovation_update_and_hysteresis():
    w, surprise = update_weights([.5, .5], [.2, 1.3], .25, .06)
    assert np.all(np.isfinite(w)) and np.sum(w) == pytest.approx(1.)
    assert w[0] > .999 and surprise > 0
    detector = Innovation(5., 1.1)
    assert not detector.push(3.)
    assert detector.push(6.)
    assert not detector.push(6.) and detector.detected
    for _ in range(20):
        detector.push(0.)
    assert not detector.detected


def test_exactly_once_observation_and_ignored_evidence(config, settings):
    observer = AdaptiveObserver(config, settings, TestSource(), revision="R0")
    past = history(config)
    observer.observe(past, 0)
    h = past.copy()
    h[-1, 0] -= .003
    observer.predict_executed(h[-1, 3], 1)
    observer.observe(h, 1)
    weights = observer.weights.copy()
    observer.observe(h, 1)
    assert np.array_equal(observer.weights, weights)
    h[-1, 0] -= .007
    with pytest.raises(ValueError, match="Conflicting"):
        observer.observe(h, 1)
    observer.observe(h, 2, evidence=False)
    assert np.array_equal(observer.weights, weights) and not observer.history
    assert observer.previous[0] == h[-1, 0]


def test_refreshed_hypotheses_use_only_past_and_held_aside_measurements(config, settings):
    source = TestSource()
    d = settings | {"revision_screen": 3, "revision_window": 9, "revision_cooldown": 5}
    observer = AdaptiveObserver(config, d, source, revision="R2", scale=.035, threshold=2.)
    h = history(config)
    observer.observe(h, 0)
    assert "road" not in inspect.signature(observer.observe).parameters
    for tick in range(1, 12):
        h = np.roll(h, -1, axis=0)
        h[-1] = history(config, velocity=15-tick*.8*config["gravity"]*config["control_dt"])[-1]
        observer.observe(h, tick)
        if observer.last.get("revision"):
            break
    report = observer.last
    assert report["revision"] == "regenerate" and report["screen_intervals"] == 3
    assert observer.generation == 1 and len(source.calls) == 2
    assert source.calls[-1][0][-1, 0] > observer.previous[0]
    assert len(set(observer.identities)) == len(observer.weights)
    assert np.all(observer.weights >= 0.) and sum(observer.weights) == pytest.approx(1.)
    assert not observer.history


def test_continuous_proposal_frozen_initial_model_and_curve_constraints(config, settings):
    from abswm.active_training import curve_grid

    donor = ProbabilisticTire(config, {"hypotheses": 4})
    origin = donor.centers.detach().clone()
    continuous = ContinuousTire(donor, settings["continuous_scale"]).eval()
    with torch.no_grad():
        continuous.prototype.centers.add_(.15)
        curves, logits = continuous(torch.tensor(history(config)[None], dtype=torch.float32))
        mu = curve_grid(curves, torch.as_tensor(GRID, dtype=torch.float32))
    assert torch.equal(origin, donor.centers.detach())
    assert mu.shape == (1, 8, len(GRID)) and torch.all(mu[..., 0] == 0)
    assert torch.all((0 <= mu) & (mu <= 1.8)) and torch.any(mu[:, 0] != mu[:, 4])
    assert logits.exp().sum().item() == pytest.approx(1.)


def test_whitened_refinement_leaves_null_direction_unchanged():
    step, values, rank = information_step(np.array([.02, .04]), np.array([[1., 0.], [2., 0.]]),
                                          np.array([1., 2.]), .05)
    assert rank == 1 and values[0] == 0. and step[1] == 0.
    assert np.isfinite(step).all()


def test_virtual_outcomes_depend_on_available_observations_only(config, settings):
    x = np.array([[14.8, 46.], [14.8, 46.], [14.8, 46.]])
    w = np.array([.2, .35, .45])
    branches = observation_branches(x, w, config, settings, .05, 3)
    assert sum(b.mass for b in branches) == pytest.approx(1.)
    assert all(np.allclose(b.posterior, w) for b in branches)
    assert len({tuple(b.state) for b in branches}) == len(branches)


def test_frozen_and_learning_policies_share_state_horizon_and_compute_budget(config, settings, options):
    observer = AdaptiveObserver(config, settings, TestSource())
    h = history(config)
    observer.observe(h, 0)
    bank = observer.snapshot()
    actions = candidate_actions(500., 7, config)[0]
    x = observer.predict(h[-1, :2], actions)
    assert np.allclose(x.state[:, ::-1], observer.predict(h[-1, :2], actions[::-1]).state)
    active = feedback_plan(h[-1, :2], 500., bank, config, settings, options, 3, "feedback", .05, detail=True)
    frozen = feedback_plan(h[-1, :2], 500., bank, config, settings, options, 3, "frozen_feedback", .05, detail=True)
    assert np.array_equal(active.segments, frozen.segments)
    assert np.allclose(active.frozen, frozen.objective)
    assert len(active.branches) == len(frozen.branches)
    assert all(a["velocity"] == b["velocity"] and a["wheel"] == b["wheel"] and
               a["mass"] == b["mass"] for a, b in zip(active.branches, frozen.branches))
    assert observer.tick == 0 and np.array_equal(observer.weights, bank.weights)
    assert not observer.history and observer.prepared is None
    identical = TireBank(np.repeat(bank.curves[:1], 2, axis=0), bank.weights, None, bank.identities)
    same = feedback_plan(h[-1, :2], 500., identical, config, settings, options, 3, "feedback", .05)
    assert np.max(same.value) < 1e-10 and same.selected == same.matched


def test_all_commitments_respect_torque_and_slew_limits(config, settings, options):
    h = history(config, torque=1950.)
    for length in settings["commit_steps"]:
        source = TestSource()
        observer = AdaptiveObserver(config, settings, source)
        controller = CommittedController(config, settings, options, observer, length, "passive")
        first = controller.decide(h, 0)
        assert first.planned_steps == length
        sequence = controller.plan[0].segments[controller.plan[1]]
        assert (sequence >= 0.).all() and (sequence <= config["torque_max"]).all()
        assert (np.abs(np.diff(np.r_[1950., sequence])) <= config["torque_rate"]*config["control_dt"]).all()
        controller.applied(first, first.requested, "")
        if length >= 2:
            next_h = h.copy()
            next_h[-1, 3] = first.requested
            second = controller.decide(next_h, 1)
            assert second.segment == first.segment and second.segment_offset == 1
            applied, reason = Supervisor(config).apply([14., 0.], 1., first.requested, second.requested)
            controller.applied(second, applied, reason)
            assert reason == "locked" and controller.plan is None
        else:
            assert controller.plan is None


def test_supervision_records_real_applied_action_and_short_branch_scores(config, options):
    class Probe:
        tick = -1

        def __init__(self):
            self.applied_inputs = []

        def predict_executed(self, torque, tick):
            self.applied_inputs.append((torque, tick))

    class Controller:
        def __init__(self):
            self.predictor = Probe()

        def decide(self, history, tick):
            self.predictor.tick = tick
            return SegmentDecision(690., segment=0, planned_steps=3,
                                   planned_exploration=True, information_changed=True)

        def applied(self, decision, torque, reason):
            pass

    c = config | {"max_time": .005}
    ctl = Controller()
    result = drive(SpatialRoad.build(scene("constant", [road("wet")]), c), ctl, (15., .84, 600.), options, audit=True)
    assert result["trace"][0]["requested"] == pytest.approx(690.)
    assert result["trace"][0]["applied"] == pytest.approx(500.)
    assert ctl.predictor.applied_inputs == [(500., 1)]
    assert result["segments"][0]["reason"] == "high_slip"
    assert len(result["audit"]) == len(AUDIT_HORIZONS)
    assert result["metrics"]["interrupted"] == 1
    assert result["metrics"]["exploration_count"] == 0


def test_exactly_matching_plant_and_oracle_use_independent_spatial_paths(config):
    low, high = road("snow"), road("dry")
    field = SpatialRoad.build(scene("patch", [high, low, high], (3., 5.), (0., 0.)), config)
    positions = [2.99, 3., 4.99, 5.]
    magnitude = [field.profile_at(s).max() for s in positions]
    assert magnitude[0] == magnitude[3] and magnitude[1] == magnitude[2] and magnitude[0] > magnitude[1]
    actions = candidate_actions(950., 20, config)[0]
    x = [15., 42., 2.99]
    current = SpatialOracle(field)
    future = SpatialOracle(field, preview=True)
    current.position = future.position = x[2]
    np.testing.assert_allclose(current.predict(x[:2], actions).state,
                               field.forecast(x, actions, preview=False)[0].state)
    assert not np.allclose(future.predict(x[:2], actions).state, current.predict(x[:2], actions).state)
    predicted, intervals = field.forecast([15., 42., 1.8], actions)
    assert np.ptp(intervals[0, :, -1]) > 0.
    alternative = SpatialRoad.build(scene("high_low", [high, road("wet")], (3.,), (0.,)), config)
    np.testing.assert_allclose(field.forecast([15., 42., 0.], actions)[0].state,
                               alternative.forecast([15., 42., 0.], actions)[0].state)
    assert (predicted.state >= 0).all()


def test_rk4_and_substep_risks_match_legacy_integrator_on_constant_profile(config):
    field = SpatialRoad.build(scene("constant", [road("snow")]), config)
    x = [12., 40.]
    actions = candidate_actions(800., 18, config)[0]
    truth, intervals = field.forecast([*x, 0.], actions, stop=0.)
    exact = integrate(x, actions[None], grid_coefficient(field.profiles), config, rk4=True)
    np.testing.assert_allclose(truth.state, exact.state, atol=1e-9)
    assert np.array_equal(truth.high, exact.high) and np.array_equal(truth.lock, exact.lock)
    assert np.all((0 <= intervals[2]) & (intervals[2] <= config["control_dt"]+1e-12))


def test_the_plant_terminates_and_records_timeouts(config, options):
    class Hold:
        def decide(self, history, tick):
            return Decision(float(history[-1, 3]))

        def applied(self, decision, torque, reason):
            pass

    c = config | {"max_time": .01}
    spatial = SpatialRoad.build(scene("constant", [road("dry")]), c)
    stopped = drive(spatial, Hold(), (2.002, .055, 800.), options)
    assert stopped["metrics"]["stopped"] and stopped["metrics"]["end_speed"] == pytest.approx(2.)
    assert stopped["metrics"]["time_s"] < c["control_dt"]
    failure = drive(spatial, Hold(), (15., 0., 0.), options)
    assert not failure["metrics"]["stopped"] and failure["metrics"]["failure"] == "timeout"


def test_split_components_are_disjoint_and_partition_is_reproducible(config, settings):
    p = make_partitions(config, settings)
    assert p == make_partitions(config, settings)
    components = [{r["id"] for collection in split.values() for s in collection for r in s["segments"]}
                  for split in p.values()]
    assert sum(map(len, components)) == len(set.union(*components))
    assert all(r["family"] != "rational" for s in p["train"]["dynamic"] for r in s["segments"])
    with pytest.raises(ValueError, match="Reused"):
        make_partitions(config, settings, components[0])
    wrong = {"metrics": {"road_id": p["validation"]["constant"][0]["id"]}, "trace": []}
    with pytest.raises(ValueError, match="partition"):
        gather_windows([wrong], p["train"], settings)
