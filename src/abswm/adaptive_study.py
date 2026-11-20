"""Staged ABSWM 3.1 protocol for the operator-run simulation study."""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .active_experiment import digest
from .active_metrics import mean_rows
from .adaptive_analysis import (calibrate, choose_innovation, feedback_factorial, numerical_trials, open_predictions,
                                open_samples, previous_safety, summarize_open)
from .adaptive_experiment import (CORE, load_new_models, models_for, run_episodes, seal, specs, table)
from .adaptive_outputs import acceptance, adaptation_statistics, closed_results, duration_results, report_episodes
from .adaptive_training import materialize, train_variants
from .config import fingerprint
from .decoder_study import read_json
from .spatial_road import FAMILIES, SCENARIOS, SpatialRoad


STAGES = ("p0", "p1", "p2", "p3", "train", "test")


def _subset(scenes, group_key, n=1):
    result, counts = [], {}
    for road in scenes:
        group = road[group_key]
        counts[group] = counts.get(group, 0)+1
        if counts[group] <= n:
            result.append(road)
    return result


def _constant_subset(scenes, counts):
    return [road for family, n in zip(FAMILIES, counts)
            for road in [s for s in scenes if s["family"] == family][:n]]


class Study:
    def __init__(self, paths, c, d, old, parts, key, workers, smoke):
        self.paths, self.output = paths, paths.output
        self.c, self.d, self.old, self.parts = c, d, old, parts
        self.key, self.workers, self.smoke = key, workers, smoke
        self.models, self.gains = models_for(paths, c, d, old)
        self.seeds = c["seeds"][:1] if smoke else c["seeds"]

    def batch(self, jobs, phase, policy, *, trained=False):
        return run_episodes(self.paths, self.c, self.d, self.old, policy, jobs, phase, self.key,
                            self.workers, weights=policy.get("weights") if trained else None,
                            training_key=policy.get("training_key"))

    def cases(self, scenes, methods, **kwargs):
        return list(specs(scenes, methods, self.c, self.d, smoke=self.smoke, **kwargs))

    def valid_roads(self):
        return self.parts["validation"]["constant"]

    def p0(self, _):
        c, d = self.c, self.d
        previous_safety(self.paths)
        routes = _constant_subset(self.valid_roads(), (3, 1, 0))
        peak_rows = []
        choices = d["peak_candidates"][:1] if self.smoke else d["peak_candidates"]
        for i, peak in enumerate(choices):
            group = self.batch(self.cases(routes, ("B4",), single_seed=True), f"p0_peak_{i}", {"peak": peak})
            cost = np.mean([x["metrics"]["distance_m"]+40*x["metrics"]["high_slip_s"]+
                            80*x["metrics"]["lock_s"]+2500*(not x["metrics"]["stopped"]) for x in group])
            peak_rows.append({"candidate": i, "cost": cost, "settings": peak})
        table(self.output, "peak_validation", peak_rows)
        peak = choices[min(peak_rows, key=lambda row: row["cost"])["candidate"]]
        policy = {"peak": peak, "source": "prototype", "revision": "R2", "refinement": "none"}
        methods = ("B0", "B3", "old_peak", "protection", "B4", "strong_deterministic",
                   "B7", "old_rk4", "old_cvar", "B8", "B11")
        jobs = self.cases(routes, methods, trace=True, audit=True, single_seed=True)
        episodes = self.batch(jobs, "p0_closed", policy)
        closed_results(self.output, episodes, d, "p0_")
        report_episodes(self.output, episodes, "p0_", traces=True)
        table(self.output, "integrator_protection_comparison",
              [x["metrics"] for x in episodes if x["metrics"]["method"] in {"B7", "old_rk4"}])
        numerical_trials(c, d, self.old, self.models, routes[:1] if self.smoke else routes,
                         self.output, self.seeds[:1])
        return policy

    def p1(self, policy):
        d, c = self.d, self.c
        constant = _constant_subset(self.valid_roads(), (3, 1, 0))
        dynamic = _subset(self.parts["validation"]["dynamic"], "kind")
        if self.smoke:
            constant, dynamic = constant[:1], dynamic[:1]
        episodes = self.batch(self.cases(constant+dynamic, ("B0",), trace=True),
                              "p1_innovation_history", policy)
        selected = choose_innovation(c, d, episodes, policy, self.models, constant+dynamic, self.output)
        policy |= {"innovation_scale": float(selected["scale"]), "innovation_threshold": float(selected["threshold"])}
        methods = ("e4", "ensemble", "prototype", "continuous", "library")
        raw = list(open_predictions(c, d, self.old, policy, self.models,
                                    open_samples(c, d, constant, smoke=self.smoke, full=False), methods, self.seeds))
        scores = []
        for name in methods:
            rows = [r for r in raw if r[0]["method"] == name]
            high_brier = np.mean([np.mean((p.weights@f.high-y.high[0])**2) for _, p, y, f, _ in rows])
            action = np.mean([r[-1]["action_wheel_rmse"] for r in rows])
            regret = np.mean([r[-1]["action_regret"] for r in rows])
            scores.append({"source": name, "action_wheel_rmse": action, "action_regret": regret,
                           "brier": high_brier, "selection_score": action/8+regret+high_brier})
        table(self.output, "representation_validation", scores)
        eligible = [x for x in scores if x["source"] in {"prototype", "continuous"}]
        source = min(eligible, key=lambda row: row["selection_score"])["source"]
        policy |= {"source": source, "representation_candidates": [x["source"] for x in eligible]}
        calibration = calibrate(self.old["predictive_scale"], [r for r in raw if r[0]["method"] == source])
        summarize_open(raw, calibration, self.output, "p1_")
        return policy | {"calibration": calibration}

    def p2(self, policy):
        d = self.d
        routes = _constant_subset(self.valid_roads(), (3, 1, 0))
        lengths = d["commit_steps"][:2] if self.smoke else d["commit_steps"]
        cases = [j for n in lengths for j in self.cases(routes, ("B10", "B13", "B14"), length=n, single_seed=True)]
        episodes = self.batch(cases, "p2_duration_validation", policy)
        duration_results(self.output, episodes, d, "p2_")
        scores = []
        for length in lengths:
            rows = [p["metrics"] for p in episodes if p["metrics"]["method"] == "B13"
                    and p["metrics"]["commit_steps"] == length]
            scores.append({"length": length, "cost": float(np.mean([m["distance_m"]+60*m["high_slip_s"]+
                                                                    100*m["lock_s"]+2500*(not m["stopped"])
                                                                    for m in rows]))})
        table(self.output, "commitment_validation", scores)
        return policy | {"commit_steps": min(scores, key=lambda item: (item["cost"], item["length"]))["length"]}

    def p3(self, policy):
        d = self.d
        scenes = _subset(self.parts["validation"]["dynamic"], "kind")
        methods = ("B0", "B7", "B8", "B10", "B13", "B14", "recent_nls", "fixed_bank", "forget",
                   "regenerate", "refine_full", "refine_subspace", "refine_selected", "window_encoder")
        if self.smoke:
            scenes = scenes[:2]
        jobs = self.cases(scenes, methods, trace=True, single_seed=True)
        episodes = self.batch(jobs, "p3_validation", policy)
        changes = report_episodes(self.output, episodes, "p3_", traces=True)
        closed_results(self.output, episodes, d, "p3_")
        adaptation_statistics(self.output, changes, d)
        choices = {"regenerate": "none", "refine_full": "full", "refine_subspace": "subspace", "refine_selected": "selected"}
        results = []
        for name, refine in choices.items():
            rows = [x["metrics"] for x in episodes if x["metrics"]["method"] == name]
            cost = np.mean([r["distance_m"]+100*r["lock_s"]+50*r["high_slip_s"]+10*r["safety_rate"]+
                            2500*(not r["stopped"]) for r in rows])
            results.append({"mode": refine, "cost": cost})
        table(self.output, "refinement_validation", results)
        return policy | {"refinement": min(results, key=lambda x: x["cost"])["mode"]}

    def train(self, policy):
        d, c = self.d, self.c
        weight_key = fingerprint([self.key, "P4 matched-size fine-tuning", policy])
        data_record = self.output/"closed_loop_data_manifest.json"
        if not data_record.exists():
            sites = _constant_subset(self.parts["train"]["constant"], (6, 3, 0))
            changes = _subset(self.parts["train"]["dynamic"], "kind")
            methods = ("B0", "B1", "B2", "B4", "B6", "B7", "B10", "B13")
            if self.smoke:
                sites, changes, methods = sites[:2], changes[:2], ("B0", "B4", "B10", "B13")
            jobs = self.cases(sites, methods, trace=True, single_seed=True)
            jobs += self.cases(changes, ("B0", "B7", "B10", "B13"), trace=True, single_seed=True)
            trained_from = self.batch(jobs, "p4_training_histories", policy)
            validation_jobs = self.cases(_constant_subset(self.valid_roads(), (3, 2, 0)),
                                          ("B0", "B6", "B10"), trace=True, single_seed=True)
            validation_jobs += self.cases(_subset(self.parts["validation"]["dynamic"], "kind"),
                                           ("B0", "B10"), trace=True, single_seed=True)
            natural = self.batch(validation_jobs, "p4_validation_histories", policy)
            materialize(c, d, self.output, weight_key, trained_from, natural, self.parts,
                        self.paths.legacy/"data"/"mixed_more.pt")
        elif read_json(data_record)["key"] != weight_key:
            raise ValueError("Training features belong to another frozen protocol")
        scores = train_variants(c, d, self.output, weight_key, policy["source"], self.models, self.seeds,
                                self.workers, self.smoke)
        weights = {r["method"]: digest(self.output/"models"/(r["method"]+".pt")) for r in scores}
        load_new_models(self.paths, self.models, weights, weight_key)
        candidates = mean_rows([r for r in scores if r["kind"] == policy["source"]], ("distribution",),
                               ("val_loss", "val_risk", "val_action", "val_observation"))
        candidate = min(candidates, key=lambda r: r["val_loss"])["distribution"]
        rows = [j | {"override": {"trained": variant}} for variant in (None, candidate)
                for j in self.cases(_constant_subset(self.valid_roads(), (3, 1, 0)), ("B10",), single_seed=True)]
        groups = self.batch(rows, "p4_training_controller_validation", policy | {"weights": weights,
                            "training_key": weight_key}, trained=True)
        k = len(rows)//2
        costs = []
        for tag, batch in zip((None, candidate), (groups[:k], groups[k:])):
            m = [r["metrics"] for r in batch]
            cost = np.mean([r["distance_m"]+70*r["high_slip_s"]+100*r["lock_s"]+15*r["safety_rate"]+
                            2500*(not r["stopped"]) for r in m])
            costs.append({"trained": tag, "validation_cost": cost})
        table(self.output, "training_policy_selection", costs)
        selected = min(costs, key=lambda r: r["validation_cost"])["trained"]
        policy |= {"training_key": weight_key, "weights": weights, "trained": selected}
        raw = list(open_predictions(c, d, self.old, policy, self.models,
                                    open_samples(c, d, _constant_subset(self.valid_roads(), (4, 3, 0)),
                                                 smoke=self.smoke, full=False),
                                    ("adaptive",), self.seeds))
        policy["calibration"] = calibrate(self.old["predictive_scale"], raw)
        seal(self.output/"model_registry.json", {"official": "e4_a", "status": "research", "weights": weights,
                                                 "candidate": policy["source"], "selected_fine_tuning": selected})
        return policy

    def _formal_jobs(self, policy):
        d = self.d
        constant = _constant_subset(self.parts["test"]["constant"], d["closed_constant_test"])
        dynamic = self.parts["test"]["dynamic"]
        narrow = _subset(constant, "family")+_subset(dynamic, "kind")
        methods = ("protection", "old_peak", "strong_deterministic", "strong_feedback", "task_proxy", "preview",
                   "recent_nls", "fixed_bank", "forget", "regenerate", "refine_full", "refine_selected",
                   "refine_subspace", "window_encoder", "prototype", "continuous", "library", "library_feedback",
                   "e4_closed")
        jobs = self.cases(constant+dynamic, CORE)
        jobs += [j for j in self.cases(narrow, methods) if j["method"] != "preview" or j["scene"]["kind"] != "constant"]
        if policy["trained"] is not None:
            jobs += [j | {"override": {"trained": None}} for j in self.cases(narrow, ("pretrain",))]
        cases_with_trace = {r["id"] for r in narrow}
        primary_seed = self.seeds[0]
        for job in jobs:
            method, seed = job["method"], job["seed"]
            job["trace"] = method == "B0" or (seed in (0, primary_seed) and job["scene"]["id"] in cases_with_trace)
            job["audit"] = job["trace"] and method in {"B7", "B8", "B10", "B13", "B14", "protection", "B4", "B11"}
        return jobs, constant, dynamic, narrow

    def test(self, policy):
        d, c = self.d, self.c
        # This file is the only entry point for final selection. Its hash is checked on resume.
        path = self.output/"validation_policy.json"
        final = policy | {"thresholds": d["thresholds"], "paths": {"partitions": digest(self.output/"frozen_partitions.json")},
                          "comparison": {"primary": ["B13", "B14", "B10"], "adaptation": ["B10", "fixed_bank"]},
                          "search_budget": {"horizon": d["total_horizon"], "first": d["first_actions"],
                                            "second": d["second_actions"], "observation_branches": d["observation_branches"]},
                          "objective": self.old["study"]["options"], "experiment_key": self.key}
        final = final | {"key": fingerprint(final)}
        if path.exists():
            stored = read_json(path)
            if stored.pop("key") != final["key"] or fingerprint(stored) != final["key"]:
                raise ValueError("A different policy is already registered against the final test")
        if not path.exists():
            seal(path, final)
        load_new_models(self.paths, self.models, final["weights"], final["training_key"])
        jobs, constant, dynamic, narrow = self._formal_jobs(final)
        if self.smoke:
            jobs = [j for j in jobs if j["scene"]["id"] in {constant[0]["id"], dynamic[0]["id"]}]
        episodes = self.batch(jobs, "final", final, trained=True)
        a, b = closed_results(self.output, episodes, d)
        changes = report_episodes(self.output, episodes, traces=True)
        adaptation = adaptation_statistics(self.output, changes, d)
        lengths = d["commit_steps"][:2] if self.smoke else d["commit_steps"]
        duration_roads = [constant[0], dynamic[0]] if self.smoke else narrow
        duration_specs = [j for length in lengths for j in
                          self.cases(duration_roads, ("B10", "B13", "B14"), length=length, single_seed=True)]
        durations = self.batch(duration_specs, "final_duration", final, trained=True)
        duration_results(self.output, durations, d)
        normal = [x for x in episodes if x["metrics"]["method"] == "B0"]
        print("P5: scoring frozen open-loop predictions and probability calibration", flush=True)
        open_methods = ("e4", "e4_closed", "prototype", "continuous", "ensemble", "library",
                        "adaptive", "untrained", "frozen", "subspace", "selected")
        if not self.smoke:
            open_methods += tuple("trained_"+k for k in ("open", "closed", "mixed", "hard", "dynamic"))
        samples = open_samples(c, d, self.parts["test"]["constant"]+dynamic, smoke=self.smoke,
                               dynamic_episodes=normal)
        predictions = open_predictions(c, d, self.old, final, self.models, samples, open_methods, self.seeds)
        _, probability, risk = summarize_open(predictions, final["calibration"], self.output, "")
        if not self.smoke:
            from .adaptive_analysis import historical_samples

            cases = historical_samples(c, self.paths)
            stress = open_predictions(c, d, self.old, final, self.models, cases,
                                      ("e4", "prototype", "continuous", "ensemble", "adaptive"), self.seeds)
            summarize_open(stress, final["calibration"], self.output, "historical_")
        print("P5: running matched real-feedback interventions", flush=True)
        self.factorial(final, normal, constant+dynamic)
        result = acceptance(self.output, c, d, final, a, b, adaptation, probability, risk, self.smoke)
        from .adaptive_figures import make_figures

        make_figures(self.output)
        return {"acceptance_passed": result["passed"], "n_closed": len(episodes),
                "n_duration": len(durations), "policy_key": final["key"]}

    def factorial(self, policy, normal, scenes):
        by_id = {r["id"]: r for r in scenes}
        entries = []
        for scene_kind in ("constant", "high_low", "low_high", "patch", "shape"):
            episode = next((r for r in normal if r["metrics"]["kind"] == scene_kind and len(r["trace"]) > 20), None)
            if episode is None:
                continue
            road = SpatialRoad.build(by_id[episode["metrics"]["road_id"]], self.c)
            if scene_kind == "constant":
                indices = [(4, "weak"), (min(50, len(episode["trace"])//3), "platform")]
            else:
                anchor = next((i for i, r in enumerate(episode["trace"]) if r["position_m"] >= road.starts[0]), None)
                indices = [] if anchor is None else [(anchor+2, "post_change")]
            for index, kind in indices:
                if index < len(episode["trace"])-3:
                    entries.append((road, episode, index, kind))
        if self.smoke:
            entries = entries[:1]
        profile = self.d if not self.smoke else self.d | {"commit_steps": self.d["commit_steps"][:1]}
        feedback_factorial(self.c, profile, self.old, policy, self.paths, self.models, self.gains,
                           entries, self.seeds[:1])

    def run(self, stop):
        policy, completed = {}, {}
        names = STAGES if stop == "all" else STAGES[:STAGES.index(stop)+1]
        records = self.output/"stages"
        records.mkdir(exist_ok=True)
        for name in names:
            mark = perf_counter()
            path = records/f"{name}.json"
            key = fingerprint([self.key, name, policy])
            if path.exists():
                cached = read_json(path)
                if cached["key"] != key:
                    raise ValueError(f"Cached phase {name} has a different frozen input")
                print(f"{name}: using completed phase", flush=True)
            else:
                print(f"{name}: starting", flush=True)
                seal(self.output/"run_status.json", {"running": name, "completed": completed, "smoke": self.smoke})
                result = getattr(self, name)(policy.copy())
                cached = {"result": result, "seconds": perf_counter()-mark, "key": key}
                seal(path, cached)
                print(f"{name}: completed in {cached['seconds']:.1f} s", flush=True)
            policy = cached["result"]
            completed[name] = cached["seconds"]
        status = {"completed": completed, "smoke": self.smoke, "finished": names[-1] == "test"}
        seal(self.output/"run_status.json", status)
        if names[-1] == "test":
            seal(self.output/"completed.json", status | policy)
        return policy
