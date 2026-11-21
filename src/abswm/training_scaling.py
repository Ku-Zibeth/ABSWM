"""Validation-only S1/S2 orchestration for the registered E4 A."""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .active_experiment import digest, environment
from .adaptive_experiment import seal, table
from .config import fingerprint
from .data import load_split
from .decoder_study import read_json
from .recurrent_model import registered_reference
from .scaling_data import atomic_torch, dataset, describe_windows, freeze_partitions, nested_locators
from .scaling_evaluation import evaluate, frozen_labels, paired_summary, summary
from .scaling_training import batches, train_budget
from .tire_prior_data import burckhardt


CONTROL_METRICS = ("action_regret_ns", "action_wheel_rmse", "force_rmse")
REPO = Path(__file__).resolve().parents[2]


def guarded(candidate, reference, d):
    def within(key, limit):
        value, baseline = candidate[key], reference[key]
        return value is None if baseline is None else value is not None and value <= baseline+limit

    drop = d["selection"]["ranking_max_drop"]
    ranking = reference["ranking_accuracy"]
    return (within("high_fnr", d["upgrade_gates"]["max_high_slip_fnr_increase"])
            and within("violation_rate", d["selection"]["violation_max_increase"])
            and (ranking is None or candidate["ranking_accuracy"] >= ranking-drop))


def validation_selection(result, reference, d):
    weights = [d["selection"][name+"_weight"] for name in ("action_regret", "action_wheel", "force")]
    score = sum(w*result[k]/max(reference[k], 1e-12) for k, w in zip(CONTROL_METRICS, weights))
    return score, guarded(result, reference, d)


def justify_b4(all_rows, d):
    out = []
    for budget in ("same_epochs", "same_updates"):
        older, recent = (all_rows[f"B{i}_{budget}"] for i in (2, 3))
        b2, b3, baseline = summary(older), summary(recent), summary(all_rows["A0"])
        gains = {key: 1-b3[key]/max(b2[key], 1e-12) for key in CONTROL_METRICS[:2]}
        consistency = {key: 0 for key in gains}
        for seed in sorted({r["seed"] for r in recent}):
            a, b = (summary([r for r in rows if r["seed"] == seed]) for rows in (recent, older))
            for key in gains:
                consistency[key] += a[key] < b[key]
        safety = guarded(b3, baseline, d)
        recommended = any(gains[key] >= d["selection"]["b4_min_gain"] and consistency[key] >= 2
                          and all(value >= -d["selection"]["b4_other_metric_max_regression"]
                                  for other, value in gains.items() if other != key) for key in gains) and safety
        out.append({"budget": budget, "relative_b2_to_b3_gain": gains, "seeds_with_improvement": consistency,
                    "risk_and_ranking_pass": safety, "recommend_b4": recommended})
    return {"run_b4": any(item["recommend_b4"] for item in out), "evidence": out}


class Study:
    def __init__(self, config_path, output, workers, smoke, *, read_only=False):
        self.output, self.d, self.smoke = output, read_json(config_path), smoke
        self.workers = workers
        registry_path = REPO/"configs"/"abswm_2.json"
        registry = read_json(registry_path)
        self.base, self.reference = (REPO/registry[key] for key in ("base_run", "reference_run"))
        self.c, self.models = registered_reference(registry_path, self.base, self.reference)
        environment(self.c["torch_threads"])
        self.study = read_json(registry_path.parent/registry["decoder_config"])
        self.archived = {seed: read_json(self.reference/"models"/f"e4_a_{seed}.json") for seed in self.models}
        if any(item["weights"] or item["peak_weighted_curve"] or item["frozen_encoder"] or
               item["epochs"] != self.study["epochs_joint"] for item in self.archived.values()):
            raise ValueError("Registered E4 A objective or provenance changed")
        self.sampling = read_json(self.base/"models"/"sampling.json")
        self.roads = read_json(self.base/"data"/"manifest.json")["roads"]
        self.n0 = 128 if smoke else self.c["train_windows"]
        self.validation_n = 32 if smoke else self.c["val_windows"]
        active = REPO/"outputs"/"active_probabilistic"
        self.options = read_json(active/"experiment.json")["configuration"]["options"] | {
            "horizon": round(read_json(active/"p0_policy.json")["selected_horizon_ms"]/(1000*self.c["control_dt"])),
            "slew_nm": self.c["control_dt"]*self.c["torque_rate"]}
        self.checkpoints, self.rows, self.timing, self.training = {}, {}, {}, {}
        self.window_details = {}
        self.val_path = output/"cache"/"validation.pt"
        # Report/plot changes do not invalidate expensive training checkpoints.
        dependencies = [registry_path, config_path, active/"p0_policy.json", self.base/"data"/"manifest.json",
                        self.base/"models"/"sampling.json", self.base/"data"/"train.npz", self.base/"data"/"val.npz",
                        *(self.reference/"models"/f"e4_a_{seed}.pt" for seed in self.models),
                        *(Path(__file__).with_name(name+".py") for name in ("scaling_training", "scaling_data",
                          "scaling_evaluation", "decoder_study", "models", "data", "active_rollout", "active_fast"))]
        provenance = {
            "config": self.d, "base_config": self.c, "decoder_config": self.study,
            "mpc_options": self.options, "registered_donors": self.archived, "smoke": smoke,
            "python_torch": str(torch.__version__),
            "cpu": {"logical": os.cpu_count(), "threads_per_worker": self.c["torch_threads"]},
            "inputs": {p.name: digest(p) for p in dependencies},
        }
        key = fingerprint(provenance)
        manifest = output/"experiment.json"
        if manifest.exists() and read_json(manifest)["experiment_key"] != key:
            raise ValueError(f"Incompatible experiment in {output}")
        frozen = output/"frozen_partitions.json"
        if read_only:
            self.policy = read_json(output/"validation_policy.json")
            if not manifest.exists() or digest(frozen) != self.policy["partition_sha256"]:
                raise ValueError("The completed S1–S2 run and its frozen partition are required")
            return
        output.mkdir(parents=True, exist_ok=True)
        seal(manifest, provenance | {"experiment_key": key, "workers_this_invocation": workers})
        if not frozen.exists():
            historical = [REPO/"outputs"/name/"frozen_partitions.json" for name in
                          ("physics_prior", "active_probabilistic", "adaptive_belief_feedback")]
            seal(frozen, freeze_partitions(self.c, self.d, self.roads, historical))
        self.policy = {"partition_sha256": digest(frozen), "selection": self.d["selection"],
                       "registered_joint_epochs": self.study["epochs_joint"],
                       "selection_reference": "registered A0 weights with matched open-loop validation only",
                       "s1": "Shortlist by 50 ms action-regret, relative-hold wheel and force composite. "
                             "Advance 4x to 8x if admissible and materially better than 2x, or recovering "
                             "from a rejected 2x. If A0 wins, choose the shortest evaluated continuation for S2.",
                       "s2": "Fix archived donor, joint optimizer and S1 continuation budget. "
                             "For B4 require a 3% B3-vs-B2 improvement in either key action metric, "
                             "improvement for at least 2 seeds, at most 2% loss in the other metric and "
                             "risk/ranking guardrails against A0. Either matched arm may qualify.",
                       "test_sealed": True}
        policy_path = output/"validation_policy.json"
        if policy_path.exists() and read_json(policy_path)["partition_sha256"] != self.policy["partition_sha256"]:
            raise ValueError("Frozen partition changed")
        seal(policy_path, self.policy)
        seal(output/"acceptance.json", {"status": "final_test_not_started", "gates": self.d["upgrade_gates"],
                                       "partition_sha256": self.policy["partition_sha256"],
                                       "registered_reference": "E4 A"})

    def prepare_validation(self):
        c = self.c
        data = load_split(self.base/"data", "val")
        points = np.asarray(self.sampling["val_locators"], dtype=np.int32)[:self.validation_n]
        if not self.val_path.exists():
            atomic_torch(self.val_path, dataset(c, self.study, data, self.roads["val"], points))
        self.val_hist = torch.load(self.val_path, weights_only=True)["history"].numpy()
        self.val_roads = [burckhardt(self.roads["val"][i]) for i in data["road_index"][points[:, 0]]]
        self.labels = frozen_labels(self.val_hist, self.val_roads, c,
                                    sorted({self.options["horizon"], c["horizon_steps"]}))

    def prepare_windows(self, multiplier):
        path = self.output/"cache"/f"windows_{multiplier}.pt"
        info = path.with_suffix(".json")
        if not path.exists() or not info.exists():
            c = self.c
            data = load_split(self.base/"data", "train")
            roads = self.roads["train"]
            archived = np.asarray(self.sampling["train_locators"], dtype=np.int32)[:self.n0]
            points = nested_locators(data, c, archived, [r["family"] for r in roads],
                                     multiplier, self.d["data_sampling_seed"])
            tensor_data = dataset(c, self.study, data, roads, points)
            for prefix in [n for n in self.d["window_multipliers"] if n <= multiplier]:
                lower = path.with_name(f"windows_{prefix}.npy")
                if lower.exists():
                    np.testing.assert_array_equal(points[:self.n0*prefix], np.load(lower, allow_pickle=False))
            atomic_torch(path, tensor_data)
            np.save(path.with_suffix(".npy"), points, allow_pickle=False)
            seal(info, describe_windows(points, tensor_data["history"].numpy(), data, roads, len(points)))
        self.window_details[multiplier] = read_json(info)
        return path

    def train(self, pool, arm, multiplier, targets):
        train = self.prepare_windows(multiplier)
        futures = {}
        for seed, target in targets.items():
            payload = {"path": str(self.output/"models"/f"{arm}_{seed}"), "arm": arm, "seed": seed,
                       "windows": multiplier*self.n0, "updates": target, "model": self.models[seed],
                       "c": self.c, "study": self.study, "train": str(train), "val": str(self.val_path)}
            futures[pool.submit(train_budget, payload)] = seed
        results = {}
        for future in as_completed(futures):
            paths = future.result()
            results[futures[future]] = paths
        return results

    def checkpoint_summary(self, method, seed, path):
        identity = path.resolve().as_posix()
        self.checkpoints[method, seed] = identity
        cache = self.output/"cache"/f"evaluation_{fingerprint({'checkpoint': identity})[:20]}.json"
        if cache.exists():
            info = read_json(cache)
        else:
            model = deepcopy(self.models[seed])
            model.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
            rows, timing = evaluate(model, self.val_hist, self.val_roads, self.labels, self.c, self.options,
                                    self.d["tie_margin"], method=method, seed=seed, timing=True)
            info = {"rows": rows, "timing": timing}
            seal(cache, info)
        self.rows.setdefault(method, []).extend(row | {"method": method} for row in info["rows"])
        self.timing.setdefault(method, []).extend(row | {"method": method} for row in info["timing"])
        if method == "A0":
            with (self.reference/"models"/f"e4_a_{seed}_history.csv").open(encoding="utf-8", newline="") as stream:
                history = [{k: float(v) if v else None for k, v in row.items()} for row in csv.DictReader(stream)]
            best = self.archived[seed]["best_epoch"]
            self.training[method, seed] = history[best-1] | {
                "updates": 0, "samples_seen": 0, "additional_epochs": 0,
                "archived_joint_epoch": best, "training_s": 0., "validation_s": 0., "history": history,
                "phase": "archived_joint_curve_training", "optimizer": "archived_weights_only"}
        else:
            payload = torch.load(path, weights_only=True)
            history = payload["history"]
            best = self.archived[seed]["best_epoch"]
            self.training[method, seed] = history[-1] | {"additional_epochs": history[-1]["epoch"],
                "archived_joint_epoch": best, "history": history, "phase": "continuation",
                "optimizer": "fresh_adam_then_exact_resume"}

    def assess(self, name, paths, updates):
        for seed in self.models:
            self.checkpoint_summary(name, seed, Path(paths[seed][updates[seed]]))
        return validation_selection(summary(self.rows[name]), summary(self.rows["A0"]), self.d)

    def epoch_scaling(self, pool):
        self.prepare_windows(1)
        for seed in self.models:
            self.checkpoint_summary("A0", seed, self.reference/"models"/f"e4_a_{seed}.pt")
        epoch = self.study["epochs_joint"]
        options = [(f"A{i}", i if self.smoke else epoch*(factor-1))
                   for i, factor in enumerate(self.d["epoch_multipliers"]) if i]
        evaluations = {"A0": (1., True)}
        paths = {}
        for name, extra in options:
            if name == "A3" and not self.smoke:
                old, recent = (evaluations[n] for n in ("A1", "A2"))
                if not recent[1] or (old[1] and recent[0] >= old[0]*(1-self.d["selection"]["epoch_material_gain"])):
                    self.policy["a3"] = "stopped after A2 for validation saturation or safety/ranking regression"
                    break
            updates = dict.fromkeys(self.models, extra*batches(self.n0, self.c))
            paths[name] = self.train(pool, "S1", 1, {s: [n] for s, n in updates.items()})
            evaluations[name] = self.assess(name, paths[name], updates)
            print(f"{name}: validation score and guardrails {evaluations[name]}", flush=True)
        winner = min((k for k, v in evaluations.items() if v[1]), key=lambda k: evaluations[k][0])
        chosen = "A1" if winner == "A0" else winner
        self.policy.update({"s1_scores": evaluations, "s1_winner_including_a0": winner, "s2_budget_source": chosen})
        extra = dict(options)[chosen]
        return extra, chosen

    def window_scaling(self, pool, extra, source, limit):
        matched = extra*batches(self.n0, self.c)
        if limit == 3:
            for arm in ("same_epochs", "same_updates"):
                for seed in self.models:
                    self.checkpoint_summary("B0_"+arm, seed, Path(self.checkpoints[source, seed]))
        start = 4 if limit == 4 else 1
        for i in range(start, limit+1):
            mul = self.d["window_multipliers"][i]
            equal_epochs = extra*batches(mul*self.n0, self.c)
            paths = self.train(pool, f"B{i}", mul, dict.fromkeys(self.models, [matched, equal_epochs]))
            for arm, count in (("same_updates", matched), ("same_epochs", equal_epochs)):
                name = f"B{i}_{arm}"
                score = self.assess(name, paths, dict.fromkeys(self.models, count))
                print(f"{name}: validation score and guardrails {score}", flush=True)
            self.export_results()

    def export_results(self):
        rows, grouped, train, curves, paired, generalization = [], [], [], [], [], []
        for name, group in self.rows.items():
            rows.extend(group)
            score, safe = validation_selection(summary(group), summary(self.rows["A0"]), self.d)
            metadata = [self.training[name, seed] for seed in self.models]
            grouped.append({"method": name, "score": score, "guardrails": safe, "seed": "pooled",
                            "train_windows": metadata[0]["train_windows"], **summary(group)})
            if name != "A0":
                against = ["A0"]
                if name.startswith("B") and name[1] != "0":
                    prefix, arm = name.split("_", 1)
                    against.append(f"B{int(prefix[1])-1}_{arm}")
                for reference in against:
                    if reference in self.rows:
                        paired.extend({"method": name, "reference": reference, **diff} for diff in paired_summary(
                            group, self.rows[reference], self.d["bootstrap_samples"], self.d["partition_seed"]))
            for seed in self.models:
                meta = self.training[name, seed]
                record = {"method": name, "seed": seed, "checkpoint": self.checkpoints[name, seed]}
                train.append(record | {k: v for k, v in meta.items() if k != "history"})
                curves.extend(record | row | {"phase": meta["phase"]} for row in meta["history"])
                seed_rows = [r for r in group if r["seed"] == seed]
                grouped.append({"method": name, "seed": seed, "train_windows": meta["train_windows"],
                                "updates": meta["updates"], "training_s": meta["training_s"], **summary(seed_rows)})
            splits = [("regime", x) for x in sorted({r["regime"] for r in group})]
            splits += [("surface", x) for x in sorted({r["surface"] for r in group})]
            for split, label in splits:
                generalization.append({"method": name, "split": split, "level": label, "family": "burckhardt",
                                       **summary([r for r in group if r[split] == label])})
        table(self.output, "training_summary", train)
        table(self.output, "learning_curves", curves)
        table(self.output, "epoch_scaling", [r for r in grouped if r["method"].startswith("A")])
        table(self.output, "window_scaling", [r for r in grouped if r["method"].startswith("B")])
        table(self.output, "timing", [r for timing in self.timing.values() for r in timing])
        table(self.output, "action_prediction", rows)
        table(self.output, "action_regret", grouped)
        table(self.output, "risk_evaluation", grouped)
        table(self.output, "paired_statistics", paired)
        table(self.output, "generalization", generalization)
        seal(self.output/"window_manifest.json", self.window_details)

    def run(self, *, b4=False):
        start = perf_counter()
        prior = self.output/"stage_status.json"
        if b4 and (not prior.exists() or not read_json(prior).get("b4_decision", {}).get("run_b4")):
            raise ValueError("B4 requires a completed and qualifying B3 comparison")
        self.prepare_validation()
        with ProcessPoolExecutor(max_workers=self.workers, initializer=environment,
                                 initargs=(self.c["torch_threads"],)) as pool:
            extra, source = self.epoch_scaling(pool)
            self.export_results()
            self.window_scaling(pool, extra, source, 3)
            decision = justify_b4(self.rows, self.d)
            self.policy["b4_diagnostics"] = decision
            seal(self.output/"validation_policy.json", self.policy)
            status = {"completed": "S2 B3", "elapsed_s_this_run": perf_counter()-start,
                      "s1_best": self.policy["s1_winner_including_a0"], "s2_epochs": extra,
                      "s2_budget_source": source, "b4_decision": decision, "final_test_sealed": True}
            if b4:
                if not decision["run_b4"]:
                    raise ValueError("Reproduced B3 did not qualify for B4")
                self.window_scaling(pool, extra, source, 4)
                status["completed"] = "S2 B4"
                status["elapsed_s_this_run"] = perf_counter()-start
            seal(prior, status)
            print(status, flush=True)
