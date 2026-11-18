import numpy as np

from abswm.data import generate_split, make_history, sample_roads, windows


def test_splits_are_disjoint_and_deterministic(config):
    c = {**config, "roads": {"train": 6, "val": 3, "test": 3}}
    a, b = sample_roads(c), sample_roads(c)
    assert a == b
    ids = [r["id"] for split in a.values() for r in split]
    assert len(set(ids)) == len(ids)
    assert all(0.03 <= r["peak"] <= 0.30 for split in a.values() for r in split)


def test_future_observations_and_current_action_cannot_change_context(config):
    states = np.column_stack((np.linspace(20, 19, 20), np.linspace(65, 55, 20)))
    actions = np.arange(19)*10.
    original = make_history(states, actions, 8, 12, config)
    states[9:] = -999
    actions[8:] = 9999
    np.testing.assert_array_equal(original, make_history(states, actions, 8, 12, config))
    assert original[-1, 3] == 70
    assert original[0, 3] == 0


def test_windows_are_aligned_and_do_not_expose_hidden_parameters(config):
    c = {**config, "data_time": 0.6, "roads": {"train": 1, "val": 1, "test": 1},
         "initial_speeds": [15.], "excitation_kinds": ["ramp"]}
    traces = generate_split(sample_roads(c)["train"], c, 0)
    samples = windows(traces, c, 16, 9)
    assert samples["history"].shape == (16, c["history_steps"], 4)
    for n, (i, t) in enumerate(samples["locators"]):
        np.testing.assert_allclose(samples["history"][n, -1, :2], traces["states"][i, t])
        np.testing.assert_array_equal(samples["states"][n, 1], traces["states"][i, t+1])
        assert samples["actions"][n, 0] == traces["actions"][i, t]
    assert "theta" not in samples and "theta" not in traces


def test_excitation_does_not_query_hidden_peak(config):
    # At the first instant, before any differing observations, applied torques
    # must match even when the hidden road is replaced.
    c = {**config, "data_time": 0.6, "initial_speeds": [15.], "excitation_kinds": ["steps"]}
    roads = sample_roads({**c, "roads": {"train": 1, "val": 1, "test": 1}})
    first = generate_split(roads["train"], c, 0)
    other = [{**roads["train"][0], "theta": [0.2, 90., 0.06]}]
    second = generate_split(other, c, 0)
    assert first["actions"][0, 0] == second["actions"][0, 0]
