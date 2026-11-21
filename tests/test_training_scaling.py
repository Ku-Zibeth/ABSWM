from pathlib import Path

import numpy as np
import torch

from abswm.decoder_study import CurveDecoder, make_training_data, read_json
from abswm.models import WorldModel
from abswm.physics import TYPICAL_ROADS
from abswm.scaling_data import atomic_torch, nested_locators
from abswm.scaling_followup import assign_tiers, replace_roads
from abswm.scaling_training import train_budget


def test_window_scaling_keeps_original_prefix_and_adds_unique_locators():
    data = {"road_index": np.array([0, 0, 1, 1]), "lengths": np.full(4, 80)}
    c = {"history_steps": 8, "horizon_steps": 4}
    archived = np.array([[0, 1], [0, 1], [1, 40], [2, 27]])
    small = nested_locators(data, c, archived, ["dry", "wet"], 2, 77)
    large = nested_locators(data, c, archived, ["dry", "wet"], 8, 77)
    np.testing.assert_array_equal(large[:len(small)], small)
    np.testing.assert_array_equal(large[:len(archived)], archived)
    assert len({tuple(row) for row in large}) == len(large)-1
    assert len(large) == 8*len(archived)
    assert np.count_nonzero(data["road_index"][large[:, 0]] == 1) == 8


def test_matched_step_checkpoint_can_resume_inside_epoch(config, tmp_path):
    c = config | {"batch_size": 2, "horizon_steps": 2, "torch_threads": 1}
    study = read_json(Path(__file__).resolve().parents[1]/"configs"/"decoder_study.json")
    h = np.zeros((5, 3, 4), dtype=np.float32)
    h[..., :2] = [16., 51.]
    h[:, -1, 0] += np.linspace(0, 1, len(h))
    sample = {"history": h, "states": np.repeat(h[:, -1:, :2], 3, axis=1),
              "actions": np.full((5, 2), 130.), "peak": np.linspace(.10, .25, 5), "mu": np.full(5, .2)}
    road = {"theta": next(iter(TYPICAL_ROADS.values()))}
    data = make_training_data(c, sample, [0]*5, [road], np.linspace(0, 1, study["train_grid_points"]))
    cache = tmp_path/"data.pt"
    atomic_torch(cache, data)
    model = CurveDecoder(c, study, WorldModel(c), 11).requires_grad_(False)
    job = {"arm": "B", "seed": 11, "windows": 5, "model": model, "c": c,
           "study": study, "train": str(cache), "val": str(cache)}
    split = tmp_path/"split"
    first = train_budget(job | {"path": split, "updates": [1]})
    payload = torch.load(first[1], weights_only=True)
    assert payload["samples"] == 2
    continued = train_budget(job | {"path": split, "updates": [1, 6]})
    whole = train_budget(job | {"path": tmp_path/"whole", "updates": [1, 6]})
    a, b = (torch.load(out[6], weights_only=True) for out in (continued, whole))
    assert a["samples"] == b["samples"] == 10
    assert a["rng"] == b["rng"]
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], atol=0, rtol=0)


def test_road_scaling_pairs_acquisition_slots_and_covers_nested_roads():
    points = np.array([(drive, t) for drive in range(4) for t in range(1, 13)], dtype=np.int32)
    tiers = assign_tiers(points, 2, 4, 19)
    lengths = np.full(16, 40)
    slot = int(np.flatnonzero((tiers > 0) & (points[:, 1] >= 10))[0])
    lengths[points[slot, 0]+4*tiers[slot]] = points[slot, 1]
    c = {"history_steps": 6, "horizon_steps": 2}
    small, _ = replace_roads(points, tiers, 2, 2, 2, lengths, c, 19)
    large, meta = replace_roads(points, tiers, 4, 2, 2, lengths, c, 19)
    assert len(large) == len(small) == len(points)
    assert set(small[:, 0]//2) == set(range(4))
    assert set(large[:, 0]//2) == set(range(8))
    np.testing.assert_array_equal(large[:, 0] % 4, points[:, 0])
    assert np.all(large[:, 1] < lengths[large[:, 0]]-c["horizon_steps"])
    assert len({tuple(pair) for pair in large}) == len(large)
    assert meta["redrawn_past_end"] > 0
