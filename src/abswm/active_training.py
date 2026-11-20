"""Structured multi-hypothesis learning with one tire context per shared past."""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.optimize import least_squares
from torch import nn
from torch.nn import functional as F

from .active_data import branches, history_windows
from .active_experiment import digest, environment
from .active_rollout import GRID
from .config import fingerprint, write_json
from .decoder_study import read_json
from .evaluation import write_csv
from .models import HistoryEncoder, seed_everything
from .tire_prior import StructuredTire, TireShape
from .tire_prior_data import tire_function


VARIANTS = {
    "rollout": ("mixed", ()),
    "action": ("mixed", ("action",)),
    "ranking": ("mixed", ("ranking",)),
    "risk": ("mixed", ("risk",)),
    "full": ("mixed", ("action", "ranking", "risk", "probability")),
    "single": ("single", ("action", "ranking", "risk", "probability")),
    "half_single": ("half_single", ("action", "ranking", "risk", "probability")),
    "mixed_more": ("mixed_more", ("action", "ranking", "risk", "probability")),
}


def tire_shape(raw):
    a, p, k, tail, q, blend = raw.sigmoid().unbind(dim=-1)
    return TireShape(.04+1.76*a, .01+.44*p, 1.25+12*k, tail, 1.05+2.7*q, blend)


def curve_grid(raw, grid):
    shape = TireShape(*(part[..., None] for part in tire_shape(raw)))
    return StructuredTire.evaluate(grid, shape)


class ProbabilisticTire(nn.Module):
    def __init__(self, c, d, initial=None):
        super().__init__()
        self.encoder = HistoryEncoder(c)
        self.k = d["hypotheses"]
        hidden = c["hidden_dim"]
        self.branch = nn.Linear(hidden, self.k*7)
        nn.init.zeros_(self.branch.bias)
        nn.init.normal_(self.branch.weight, std=.01)
        self.centers = nn.Parameter(torch.zeros(self.k, 6) if initial is None else torch.as_tensor(
            initial, dtype=torch.float32))
        self.register_buffer("latent_scale", torch.ones(6))

    def forward(self, history):
        estimate = self.branch(self.encoder(history)).reshape(-1, self.k, 7)
        raw = self.centers+.5*estimate[..., :6].tanh()
        return raw, estimate[..., 6].log_softmax(dim=-1)


def torch_forecast(initial, actions, raw, c):
    """Differentiable projection Euler, used with all candidates in each batch."""
    grid = torch.linspace(0., 1., 129, dtype=raw.dtype, device=raw.device)
    curves = curve_grid(raw, grid)
    n = curves.shape[-1]-1
    batch, k, actions_n = len(raw), raw.shape[1], actions.shape[1]
    curves = curves[:, :, None].expand(-1, -1, actions_n, -1)
    v = initial[:, None, None, 0].expand(batch, k, actions_n)
    w = initial[:, None, None, 1].expand_as(v)
    r, g, dt = c["radius"], c["gravity"], c["physics_dt"]
    factor = r*c["mass"]*g/c["inertia"]

    def coefficient(lam):
        index = lam*n
        lower = index.long().clamp(max=n-1)
        left = curves.gather(-1, lower[..., None]).squeeze(-1)
        right = curves.gather(-1, lower[..., None]+1).squeeze(-1)
        return left+(index-lower)*(right-left)

    lams, speeds, wheels, mus = [], [], [], []
    highest = lam = (1-r*w/v.clamp_min(1e-6)).clamp(0, 1)
    lowest = w
    for torque in actions.unbind(dim=-1):
        brake = torque[:, None]/c["inertia"]
        for _ in range(round(c["control_dt"]/dt)):
            mu = coefficient(lam)
            v = (v-dt*g*mu).clamp_min(0)
            w = (w+dt*(factor*mu-brake)).clamp_min(0)
            lam = (1-r*w/v.clamp_min(1e-6)).clamp(0, 1)
            highest, lowest = torch.maximum(highest, lam), torch.minimum(lowest, w)
        lams.append(lam)
        speeds.append(v)
        wheels.append(w)
        mus.append(coefficient(lam))
    state = torch.stack((torch.stack(speeds, dim=-1), torch.stack(wheels, dim=-1)), dim=-1)
    reward = (initial[:, None, None, 0]-v)/(g*c["control_dt"]*actions.shape[-1])
    return state, torch.stack(lams, dim=-1), torch.stack(mus, dim=-1), reward, highest, lowest


def losses(model, batch, c, d, terms):
    history, actions = batch["history"], batch["actions"]
    raw, logp = model(history)
    state, slip, mu, reward, high, wheel = torch_forecast(history[:, -1, :2], actions, raw, c)
    ref_state = batch["state"][:, None]
    ref_slip, ref_mu = batch["slip"][:, None], batch["mu"][:, None]
    error = ((state-ref_state)/state.new_tensor([.35, 15])).square().mean(dim=(-1, -2, -3))
    error += ((slip-ref_slip)/.18).square().mean(dim=(-1, -2))*.35
    error += ((mu-ref_mu)/.2).square().mean(dim=(-1, -2))*.45
    # One mixture component scores all actions and all future steps together.
    rollout = -torch.logsumexp(logp-2*error, dim=1).mean()/2
    probability = logp.exp()
    expected = torch.einsum("bk,bka->ba", probability, reward)
    state_delta = state[..., -1, :]-state[:, :, :1, -1, :]
    ref_delta = ref_state[..., -1, :]-ref_state[:, :, :1, -1, :]
    expected_delta = torch.einsum("bk,bkad->bad", probability, state_delta)
    action = ((expected_delta[:, 1:]-ref_delta[:, 0, 1:])/state.new_tensor([.25, 12])).square().mean()
    slip_end = torch.einsum("bk,bka->ba", probability, slip[..., -1])
    mu_end = torch.einsum("bk,bka->ba", probability, mu[..., -1])
    for predicted, target, scale in ((slip_end, batch["slip"][..., -1], .15),
                                     (mu_end, batch["mu"][..., -1], .2)):
        action = action+.15*(((predicted-predicted[:, :1])-(target-target[:, :1]))/scale).square().mean()
    truth = batch["reward"]
    differences = truth[:, :, None]-truth[:, None, :]
    meaningful = differences.abs() > d["tie_margin_mu"]
    predicted = expected[:, :, None]-expected[:, None, :]
    ranking = (F.softplus(-differences.sign()*predicted/.025)*meaningful).sum()/meaningful.sum().clamp_min(1)
    high_probability = torch.einsum("bk,bka->ba", probability, ((high-.8)/.045).sigmoid())
    lock_probability = torch.einsum("bk,bka->ba", probability, ((1.-wheel)/.8).sigmoid())
    risk = (high_probability-batch["high"]).square().mean()+(lock_probability-batch["lock"]).square().mean()
    curves = curve_grid(raw, torch.linspace(0., 1., batch["curve"].shape[-1], device=raw.device))
    residual = ((curves-batch["curve"][:, None])/.15).square().mean(dim=-1)
    proper = -torch.logsumexp(logp-3*residual, dim=1).mean()/3
    values = {"rollout": rollout, "action": action, "ranking": ranking, "risk": risk, "probability": proper}
    total = rollout+sum(d["loss_weights"][term]*values[term] for term in terms)
    return {"loss": total, **values}


def fit_parameters(grid, target, initial=None):
    """Offline fitting for prototype initialization and representation diagnostics."""
    x0 = np.array([0., -1., -1., .5, -1.4, -.5]) if initial is None else initial

    def objective(raw):
        value = curve_grid(torch.as_tensor(raw), torch.as_tensor(grid)).numpy()
        return np.concatenate((value-target, .002*(raw-x0)))

    return least_squares(objective, x0, bounds=(-9, 9), max_nfev=45).x


def train_data(c, d, suite, seed, *, training):
    data = branches(c, d, suite, random_seed=seed if training else None)
    grid = GRID[::4]
    data["curve"] = tire_function(suite["roads"])(np.broadcast_to(grid, (len(suite["roads"]), len(grid))))
    return {name: torch.as_tensor(x, dtype=torch.float32) for name, x in data.items()}


def init_worker():
    environment(2)


def train_job(job):
    c, d, variant, seed, reference, root, data_path, validation_path, key = job
    target = root/f"{variant}_{seed}.pt"
    if target.exists() and target.with_suffix(".csv").exists():
        saved = torch.load(target, weights_only=True)
        if saved["key"] != key:
            raise ValueError(f"Stale checkpoint: {target}")
        return saved["summary"]
    seed_everything(seed, c)
    dataset, terms = VARIANTS[variant]
    train = torch.load(data_path, weights_only=True)
    validation = torch.load(validation_path, weights_only=True)
    profiles = train["curve"].numpy()
    initial_curves, _ = kmeans2(profiles, d["hypotheses"], minit="++", seed=seed)
    initial = np.stack([fit_parameters(GRID[::4], curve) for curve in sorted(initial_curves, key=np.max)])
    model = ProbabilisticTire(c, d, initial)
    model.latent_scale.copy_(torch.as_tensor(initial.std(axis=0).clip(.5, 3), dtype=torch.float32))
    source = torch.load(reference/"models"/f"e4_a_{seed}.pt", weights_only=True)["state_dict"]
    model.encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in source.items()
                                   if k.startswith("encoder.")})
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": .25*d["learning_rate"]},
                                   {"params": [model.centers, *model.branch.parameters()]}], lr=d["learning_rate"])
    best, log, start = float("inf"), [], perf_counter()
    print(f"train {variant} seed={seed} windows={len(train['history'])}", flush=True)
    rng = torch.Generator().manual_seed(seed)
    for epoch in range(1, d["epochs"]+1):
        model.train()
        for ids in torch.randperm(len(train["history"]), generator=rng).split(d["batch_size"]):
            batch = {name: x[ids] for name, x in train.items()}
            values = losses(model, batch, c, d, terms)
            optimizer.zero_grad()
            values["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        summary = {}
        with torch.no_grad():
            for ids in torch.arange(len(validation["history"])).split(d["batch_size"]):
                values = losses(model, {name: x[ids] for name, x in validation.items()}, c, d, terms)
                for name, value in values.items():
                    summary[name] = summary.get(name, 0.)+value.item()*len(ids)/len(validation["history"])
        score = summary["rollout"]+.4*summary["action"]+.14*summary["ranking"]+.2*summary["risk"]
        row = {"epoch": epoch, "val_score": score, **{f"val_{name}": value for name, value in summary.items()}}
        log.append(row)
        if score < best:
            best, record = score, row
            weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    summary = {"method": variant, "seed": seed, "dataset": dataset, "windows": len(train["history"]),
               "seconds": perf_counter()-start, **record}
    write_csv(target.with_suffix(".csv"), log)
    temp = target.with_suffix(".tmp")
    torch.save({"key": key, "summary": summary, "state": weights}, temp)
    temp.replace(target)
    print(f"done {variant} seed={seed} score={best:.5f} seconds={summary['seconds']:.1f}", flush=True)
    return summary


def merge(*datasets):
    return {name: torch.cat([d[name] for d in datasets]) for name in datasets[0]}


def run_training(c, d, paths, partitions, output, workers, *, smoke=False):
    d = d | ({"fit_windows": 32, "validation_windows": 16, "epochs": 1, "training_steps": 10} if smoke else {})
    variants = ("rollout", "full") if smoke else tuple(VARIANTS)
    folder = output/"models"
    folder.mkdir(exist_ok=True)
    data_root = output/"data"
    data_root.mkdir(exist_ok=True)
    key = fingerprint({"experiment": digest(output/"experiment.json"), "configuration": d, "variants": variants,
                       "source": {name: digest(folder_path) for name in
                                  ("active_training.py", "active_data.py", "active_rollout.py")
                                  for folder_path in [paths[0].parent.parent/"src"/"abswm"/name]}})
    protocol = {"training_key": key, "study": d, "variants": variants, "smoke": smoke}
    write_json(output/"training_protocol.json", protocol)
    datasets = {}
    seed = d["partition_seed"]+11
    for name, count, offset in (("burckhardt", d["fit_windows"], 1), ("magic", d["fit_windows"]//2, 2)):
        suite = history_windows(c, d, partitions[f"train_{name}"], count, seed+offset)
        data = train_data(c, d, suite, seed+offset, training=True)
        datasets[name] = data
    # Keep the same original windows in the matched subset and mixed dataset.
    n = min(len(datasets["burckhardt"]["history"])//2, len(datasets["magic"]["history"]))
    half = {name: x[:n] for name, x in datasets["burckhardt"].items()}
    single = {name: x[:2*n] for name, x in datasets["burckhardt"].items()}
    magic = {name: x[:n] for name, x in datasets["magic"].items()}
    for name, data in (("half_single", half), ("single", single), ("mixed", merge(half, magic)),
                       ("mixed_more", merge(single, magic))):
        torch.save(data, data_root/f"{name}.pt")
    val = [train_data(c, d, history_windows(c, d, partitions[f"validation_{family}"],
                                           d["validation_windows"]//2, seed+i), seed+i, training=False)
           for i, family in enumerate(("burckhardt", "magic"), 4)]
    torch.save(merge(*val), data_root/"validation.pt")
    jobs = [(c, d, variant, s, paths[2], folder, data_root/f"{VARIANTS[variant][0]}.pt",
             data_root/"validation.pt", key) for variant, s in itertools.product(variants, c["seeds"])]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        result = list(pool.map(train_job, jobs))
    write_csv(output/"training_summary.csv", result)
    return protocol


def load_trained(output, c, d, *, variants=None, expected_key=None):
    protocol = read_json(output/"training_protocol.json")
    if expected_key is not None and expected_key != protocol["training_key"]:
        raise ValueError("Model protocol changed after selection")
    result = {}
    for variant, seed in itertools.product(variants or protocol["variants"], c["seeds"]):
        path = output/"models"/f"{variant}_{seed}.pt"
        payload = torch.load(path, weights_only=True)
        if payload["key"] != protocol["training_key"]:
            raise ValueError(f"Invalid checkpoint provenance: {path}")
        model = ProbabilisticTire(c, d)
        model.load_state_dict(payload["state"])
        result[variant, seed] = model.eval().requires_grad_(False)
    return result
