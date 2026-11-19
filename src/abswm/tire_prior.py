"""Train-only probability prior, structured decoder and causal latent refinement."""

from __future__ import annotations

import copy
import math
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .recurrent_model import slip_tensor


class TireShape(NamedTuple):
    height: torch.Tensor
    peak: torch.Tensor
    rise: torch.Tensor
    tail: torch.Tensor
    fall: torch.Tensor
    blend: torch.Tensor


class StructuredTire(nn.Module):
    """Monotone normalized rise and fall with a shared zero-slope peak."""

    def __init__(self, reference):
        super().__init__()
        self.c = reference.c
        self.encoder = copy.deepcopy(reference.encoder)
        width = self.c["mlp_dim"]
        self.head = nn.Sequential(nn.Linear(self.c["hidden_dim"], width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 6))
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([-0.4, -1., -1., .5, -1.4, -.5]))

    def encode(self, history):
        return self.encoder(history)

    def parameters_for(self, z):
        a, p, k, r, q, b = self.head(z).sigmoid().unbind(dim=-1)
        return TireShape(.04+1.76*a, .01+.44*p, 1.25+12*k, r, 1.05+2.7*q, b)

    @staticmethod
    def physical(shape):
        return torch.stack((shape.height, shape.peak, shape.height*shape.rise/shape.peak, shape.tail), dim=-1)

    @staticmethod
    def evaluate(lam, shape):
        rise = (lam/shape.peak).clamp(0, 1)
        fall = ((lam-shape.peak)/(1-shape.peak)).clamp(0, 1)
        front = 1-(1-rise).pow(shape.rise)
        back = (1-shape.blend)*fall.pow(shape.fall)+shape.blend*(fall.square()*(3-2*fall))
        curve = torch.where(lam <= shape.peak, front, 1-(1-shape.tail)*back)
        return shape.height*curve

    def friction(self, lam, z):
        return self.evaluate(lam, self.parameters_for(z))


def friction_query(model, z, slip):
    if isinstance(model, StructuredTire):
        shape = TireShape(*(value[:, None] for value in model.parameters_for(z)))
        return model.evaluate(slip, shape)
    return model.friction(slip, z[:, None, :].expand(-1, slip.shape[-1], -1))


def differentiable_descriptors(values, grid):
    # Keep the smooth position estimate local so distant, flat tails do not pull it.
    discrete = grid[values.detach().argmax(dim=-1)]
    allowed = (grid-discrete[:, None]).abs() <= .10
    weights = (values/.015).masked_fill(~allowed, -torch.inf).softmax(dim=-1)
    height = (weights*values).sum(dim=-1).clamp_min(1e-6)
    peak = (weights*grid).sum(dim=-1)
    stiffness = (values[:, 1]-values[:, 0])/(grid[1]-grid[0])
    return torch.stack((height, peak, stiffness, values[:, -1]/height), dim=-1)


class PhysicalPrior(nn.Module):
    """A standardized Gaussian kernel mixture using only training road descriptors."""

    def __init__(self, training, bandwidth):
        super().__init__()
        self.register_buffer("center", training.mean(dim=0))
        self.register_buffer("scale", training.std(dim=0).clamp_min(1e-5))
        self.register_buffer("samples", (training-self.center)/self.scale)
        self.bandwidth = bandwidth

    def nll(self, physical, dims=4):
        query = ((physical-self.center)/self.scale)[..., :dims]
        distance = ((query[:, None]-self.samples[None, :, :dims])/self.bandwidth).square().sum(dim=-1)
        log_kernel = -.5*distance-dims*math.log(self.bandwidth*math.sqrt(2*math.pi))
        log_density = torch.logsumexp(log_kernel, dim=1)-math.log(len(self.samples))
        return -torch.logaddexp(log_density, log_density.new_full((), -22.))


def range_loss(physical, d):
    height, peak = physical[:, 0], physical[:, 1]
    lo_h, hi_h = d["height_range"]
    lo_p, hi_p = d["peak_range"]
    return ((height-hi_h).relu().square()+(lo_h-height).relu().square()
            +((peak-hi_p).relu()/.2).square()+((lo_p-peak).relu()/.2).square()).mean()


def soft_shape(values, grid):
    intervals = (grid[1:]+grid[:-1])/2
    peak = grid[values.detach().argmax(dim=-1)]
    dy = values.diff(dim=-1)
    sign = torch.where(intervals < peak[:, None]-.015, 1., torch.where(intervals > peak[:, None]+.015, -1., 0.))
    wrong = (-sign*dy*100).relu().square().mean()
    curvature = values.diff(n=2, dim=-1)*100
    smooth = (curvature.square()*(grid[1:-1] >= .02)).mean()
    return wrong, smooth


class HistoryEvidence(NamedTuple):
    left: torch.Tensor
    right: torch.Tensor
    velocity_mu: torch.Tensor
    wheel_mu: torch.Tensor
    velocity_mask: torch.Tensor
    wheel_mask: torch.Tensor
    usable: torch.Tensor


def history_evidence(history, c):
    start, end = history[:, :-1], history[:, 1:]
    dt, g = c["control_dt"], c["gravity"]
    left, right = start[..., 2], end[..., 2]
    velocity = -(end[..., 0]-start[..., 0])/(dt*g)
    wheel = (c["inertia"]*(end[..., 1]-start[..., 1])/dt+end[..., 3])/(c["radius"]*c["mass"]*g)
    mask = (end[..., 0] > c["stop_speed"]+.05) & ((left+right)/2 > .002) & ((left-right).abs() < .18)
    velocity_mask = mask & (velocity >= -1e-4) & (velocity <= 2.)
    wheel_mask = mask & (start[..., 1] > .5) & (end[..., 1] > .5) & (wheel >= -1e-4) & (wheel <= 2.)
    span = torch.where(velocity_mask, (left+right)/2, 0).amax(dim=1)-torch.where(
        velocity_mask, (left+right)/2, 1).amin(dim=1)
    usable = (velocity_mask.sum(dim=1) >= 6) & (span >= .003)
    return HistoryEvidence(left, right, velocity, wheel, velocity_mask, wheel_mask, usable)


def history_residual(model, z, evidence, mode):
    midpoint = friction_query(model, z, (evidence.left+evidence.right)/2)
    if mode in ("integral", "joint"):
        averaged = (friction_query(model, z, evidence.left)+4*midpoint+friction_query(model, z, evidence.right))/6
    else:
        averaged = midpoint

    def mse(prediction, target, mask):
        return ((prediction-target).square()*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)

    velocity = mse(averaged, evidence.velocity_mu, evidence.velocity_mask)
    wheel = mse(averaged, evidence.wheel_mu, evidence.wheel_mask)
    if mode == "wheel":
        return wheel
    if mode == "joint":
        return torch.where(evidence.wheel_mask.any(dim=1), (velocity+wheel)/2, velocity)
    return velocity


def refine(model, initial, evidence, d, checkpoints, *, prior=None, warm=None):
    """Batched inference-time optimization. Model weights and history remain fixed."""
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze the model before latent refinement")
    start = initial.detach()
    trusted = start if warm is None else torch.where(torch.isfinite(warm).all(dim=1, keepdim=True), warm.detach(), start)
    current = torch.where(evidence.usable[:, None], .65*trusted+.35*start, trusted)

    def project(z):
        shift = z-start
        return start+shift*(d["refinement_radius"]/shift.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1)

    def objective(z):
        error = history_residual(model, z, evidence, "joint")
        cost = error+d["refinement_anchor"]*(z-start).square().sum(dim=1)
        if prior is not None:
            cost = cost+d["refinement_prior"]*prior.nll(model.physical(model.parameters_for(z)))
        return cost, error

    current = project(current)
    with torch.no_grad():
        value, residual = objective(current)
        good = torch.isfinite(value)
        current = torch.where(good[:, None], current, start)
    values = {0: (current.detach(), residual.detach(), torch.zeros_like(value, dtype=torch.int64))}
    rejected = torch.zeros_like(value, dtype=torch.int64)
    for k in range(1, max(checkpoints)+1):
        z = current.detach().requires_grad_(True)
        cost, _ = objective(z)
        gradient, = torch.autograd.grad(torch.where(torch.isfinite(cost), cost, 0).sum(), z)
        with torch.no_grad():
            gradient = torch.nan_to_num(gradient)
            direction = gradient/gradient.norm(dim=1, keepdim=True).clamp_min(1e-8)
            proposal = project(current-d["refinement_rate"]*direction)
            candidate, _ = objective(proposal)
            accept = evidence.usable & good & torch.isfinite(candidate) & (candidate < value)
            current = torch.where(accept[:, None], proposal, current)
            value = torch.where(accept, candidate, value)
            smaller = project(current-.25*d["refinement_rate"]*direction)
            candidate, _ = objective(smaller)
            recover = evidence.usable & good & ~accept & torch.isfinite(candidate) & (candidate < value)
            current = torch.where(recover[:, None], smaller, current)
            value = torch.where(recover, candidate, value)
            rejected = rejected+(evidence.usable & ~(accept | recover)).int()
            if k in checkpoints:
                _, residual = objective(current)
                values[k] = (current.detach(), residual.detach(), rejected.clone())
    return values


@torch.inference_mode()
def fixed_forecast(model, z, initial, actions, c):
    """Keep road context fixed while integrating candidate vehicle dynamics."""
    state = initial
    dt = c["physics_dt"]
    shape = model.parameters_for(z) if isinstance(model, StructuredTire) else None

    def coefficient(x):
        lam = slip_tensor(x, c["radius"])
        return model.evaluate(lam, shape) if shape is not None else model.friction(lam, z)

    states, forces = [], []
    for torque in actions.unbind(dim=1):
        for _ in range(round(c["control_dt"]/dt)):
            mu = coefficient(state)
            dw = (c["radius"]*c["mass"]*c["gravity"]*mu-torque)/c["inertia"]
            state = (state+dt*torch.stack((-c["gravity"]*mu, dw), dim=1)).clamp_min(0)
        states.append(state)
        forces.append(coefficient(state))
    return torch.stack(states, dim=1), torch.stack(forces, dim=1)
