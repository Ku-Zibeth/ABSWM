from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class HistoryEncoder(nn.Module):
    def __init__(self, c, current_only=False):
        super().__init__()
        self.current_only = current_only
        self.differences = c.get("encoder_difference_features", False)
        self.dt, self.gravity = c["control_dt"], c["gravity"]
        self.register_buffer("scale", torch.tensor([40.0, 40.0/c["radius"], 1.0, c["torque_max"]]))
        self.gru = nn.GRU(6 if self.differences else 4, c["hidden_dim"], batch_first=True)

    def forward(self, history):
        history = history[:, -1:] if self.current_only else history
        features = history / self.scale
        if self.differences:
            # Deterministic causal preprocessing: small speed increments are
            # resolved on an acceleration scale, without adding sensor/truth inputs.
            increments = history[:, 1:] - history[:, :-1]
            acceleration = -increments[..., 0]/(2*self.gravity*self.dt)
            slip_rate = increments[..., 2]/(20*self.dt)
            diff = torch.stack((acceleration, slip_rate), dim=-1)
            diff = F.pad(diff, (0, 0, 1, 0)).clamp(-5, 5)
            features = torch.cat((features, diff), dim=-1)
        _, hidden = self.gru(features)
        return hidden[-1]


class WorldModel(nn.Module):
    def __init__(self, c, current_only=False):
        super().__init__()
        self.c = c
        self.encoder = HistoryEncoder(c, current_only)
        width = c["mlp_dim"]
        self.decoder = nn.Sequential(nn.Linear(c["hidden_dim"]+1, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.constant_(self.decoder[-1].bias, 2.0)

    def encode(self, history):
        return self.encoder(history)

    def friction(self, lam, z):
        # Nonnegative, bounded, and exactly zero at zero slip. No prescribed peak shape.
        lam = torch.clamp(lam, 0, 1)
        raw = self.decoder(torch.cat((lam.unsqueeze(-1), z), dim=-1)).squeeze(-1)
        return 2.0 * torch.tanh(lam * F.softplus(raw))

    def rollout(self, history, actions):
        z = self.encode(history)
        x = history[:, -1, :2]
        predictions = []
        c = self.c
        dt = c["physics_dt"]
        # Road context is fixed in an imagined trajectory; no imagined observations
        # are presented to the encoder as if they were real measurements.
        for u in actions.unbind(dim=1):
            for _ in range(round(c["control_dt"]/dt)):
                lam = (1-c["radius"]*x[:, 1]/x[:, 0].clamp_min(1e-6)).clamp(0, 1)
                mu = self.friction(lam, z)
                force = c["mass"]*c["gravity"]*mu
                dx = torch.stack((-c["gravity"]*mu,
                                  (c["radius"]*force-u.clamp(0, c["torque_max"]))/c["inertia"]), dim=-1)
                x = (x+dt*dx).clamp_min(0)
            predictions.append(x)
        return torch.stack(predictions, dim=1)

    def peak(self, history, grid_size=1001):
        z = self.encode(history)
        grid = torch.linspace(0, 1, grid_size, device=history.device)
        values = self.friction(grid.expand(len(z), -1).reshape(-1),
                               z[:, None, :].expand(-1, grid_size, -1).reshape(-1, z.shape[-1]))
        values = values.reshape(len(z), grid_size)
        return grid[values.argmax(dim=1)]


class DirectPeak(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.encoder = HistoryEncoder(c)
        self.lo, self.hi = c["reference_min"], c["reference_max"]
        w = c["mlp_dim"]
        self.head = nn.Sequential(nn.Linear(c["hidden_dim"], w), nn.SiLU(),
                                  nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 1))

    def peak(self, history):
        return self.lo+(self.hi-self.lo)*torch.sigmoid(self.head(self.encoder(history)).squeeze(-1))


def build_model(kind, c):
    if kind == "direct":
        return DirectPeak(c)
    if kind in ("world", "current"):
        return WorldModel(c, current_only=kind == "current")
    raise ValueError(f"Unknown model kind: {kind}")


def seed_everything(seed, c):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(c["torch_threads"])
    torch.use_deterministic_algorithms(True)


def load_model(path, device="cpu"):
    # Only load checkpoints produced by this project; tensor-only safe deserialization.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["kind"], checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
