"""ABSWM 2.0 reference and deterministic observation/forecast updates."""

from __future__ import annotations

import copy
import hashlib
from typing import NamedTuple

import torch
from torch import nn

from .config import fingerprint, load_config
from .decoder_study import CurveDecoder, read_json
from .evaluation import load_models


def registered_reference(registry_path, base, reference):
    registry = read_json(registry_path)
    c = load_config(base/"config.json")
    if c != load_config(registry_path.parent/registry["base_config"]) or registry["seeds"] != c["seeds"]:
        raise ValueError("ABSWM 2.0 requires the registered base profile and seeds")
    study = read_json(registry_path.parent/registry["decoder_config"])
    manifest = read_json(reference/"experiment.json")
    evaluation = read_json(reference/"evaluation.json")
    if manifest["base_config"] != c or manifest["study_config"] != study:
        raise ValueError("Reference training options do not match the registry")
    method = registry["reference_model"]
    if method != "e4_a" or evaluation["validation_chosen_e4"] != method:
        raise ValueError("Expected the registered, validation-selected E4 A reference")
    for name, expected in manifest["input_sha256"].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference was trained against different base input: {name}")
    test_files = {"test.npz": base/"data"/"test.npz",
                  "open_loop_windows.npz": base/"open_loop_windows.npz"}
    for name, path in test_files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != evaluation["base_test_sha256"][name]:
            raise ValueError(f"Reference was evaluated against different test input: {name}")
    run_key = manifest["experiment_key"]
    if run_key != fingerprint({key: manifest[key] for key in ("base_config", "study_config", "input_sha256",
                                                                "training_source_sha256")}):
        raise ValueError("Inconsistent reference manifest")
    models = {}
    originals = load_models(c, base)
    for seed in registry["seeds"]:
        path = reference/"models"/f"{method}_{seed}.pt"
        payload = torch.load(path, map_location=c["device"], weights_only=True)
        if payload["experiment_key"] != run_key:
            raise ValueError(f"Checkpoint does not match the reference manifest: {path.name}")
        model = CurveDecoder(c, study, originals["world", seed]).to(c["device"])
        model.load_state_dict(payload["state_dict"])
        models[seed] = model.eval().requires_grad_(False)
    return c, models


class Posterior(NamedTuple):
    z: torch.Tensor
    last_observation: torch.Tensor


def slip_tensor(state, radius):
    return (1-radius*state[..., 1]/state[..., 0].clamp_min(1e-6)).clamp(0, 1)


class RecurrentTireModel(nn.Module):
    """Persist GRU memory between measurements and keep imagined state separate."""

    def __init__(self, reference, options, imagine=False):
        super().__init__()
        self.tire = copy.deepcopy(reference).requires_grad_(False)
        self.tire.encoder.requires_grad_(True)
        self.c = reference.c
        self.transition = None
        self.transition_limit = options["transition_limit"]
        if imagine:
            width, hidden = options["transition_width"], self.c["hidden_dim"]
            self.transition = nn.Sequential(nn.Linear(hidden+3, width), nn.SiLU(), nn.Linear(width, hidden))
            nn.init.zeros_(self.transition[-1].weight)
            nn.init.zeros_(self.transition[-1].bias)

    def observe(self, observations, memory=None, *, return_sequence=False):
        """Consume measured [v, omega, slip, previous torque] rows in time order."""
        encoder = self.tire.encoder
        features = observations/encoder.scale
        if encoder.differences:
            previous = observations[:, :1] if memory is None else memory.last_observation[:, None]
            diff = observations-torch.cat((previous, observations[:, :-1]), dim=1)
            derivatives = torch.stack((-diff[..., 0]/(2*encoder.gravity*encoder.dt),
                                       diff[..., 2]/(20*encoder.dt)), dim=-1).clamp(-5, 5)
            features = torch.cat((features, derivatives), dim=-1)
        hidden = None if memory is None else memory.z.unsqueeze(0)
        sequence, final = encoder.gru(features, hidden)
        posterior = Posterior(final[-1], observations[:, -1])
        return (posterior, sequence) if return_sequence else posterior

    def predict(self, memory, actions):
        """Roll out each torque sequence from the same posterior without changing it."""
        c = self.c
        x, z = memory.last_observation[:, :2], memory.z
        dt = c["physics_dt"]
        states, forces, contexts = [], [], []
        for u in actions.unbind(dim=1):
            u = u.clamp(0, c["torque_max"])
            for _ in range(round(c["control_dt"]/dt)):
                mu = self.tire.friction(slip_tensor(x, c["radius"]), z)
                force = c["mass"]*c["gravity"]*mu
                dw = (c["radius"]*force-u)/c["inertia"]
                x = (x+dt*torch.stack((-c["gravity"]*mu, dw), dim=-1)).clamp_min(0)
            if self.transition is not None:
                inputs = torch.cat((z, x/self.tire.encoder.scale[:2], u[:, None]/c["torque_max"]), dim=1)
                z = z+self.transition_limit*self.transition(inputs).tanh()
            states.append(x)
            forces.append(self.tire.friction(slip_tensor(x, c["radius"]), z))
            contexts.append(z)
        return torch.stack(states, dim=1), torch.stack(forces, dim=1), torch.stack(contexts, dim=1)
