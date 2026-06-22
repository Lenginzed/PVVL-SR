#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv

from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.logger import to_jsonable
from lag_extensions.reward_shaping.situation_renderer import SituationRenderer
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler


FORMAL_CATEGORIES = LABEL_NAMES + ["mixed_ambiguous", "normal_rollout"]


def main():
    parser = argparse.ArgumentParser(description="Build a 100-sample formal offline semantic dataset.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--output-dir", default="scripts/results/aircombat_semantic_dataset_stage7_formal100")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml")
    parser.add_argument("--target-total", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=12000)
    parser.add_argument("--sample-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--agent-side", choices=["ego", "all"], default="ego")
    args = parser.parse_args()

    if args.env_name != "SingleCombat":
        raise NotImplementedError("Only SingleCombat is supported for the formal semantic dataset.")
    summary = build_formal_dataset(
        scenario_name=args.scenario_name,
        output_dir=args.output_dir,
        config_path=args.config,
        target_total=args.target_total,
        max_candidates=args.max_candidates,
        sample_interval=args.sample_interval,
        seed=args.seed,
        agent_side=args.agent_side,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_formal_dataset(
    scenario_name,
    output_dir,
    config_path,
    target_total,
    max_candidates,
    sample_interval,
    seed,
    agent_side,
):
    config = load_reward_shaping_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    env = SingleCombatEnv(scenario_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.reset()

    extractor = StateExtractor()
    labeler = TacticalLabeler(config.get("thresholds", {}))
    renderer_config = dict(config.get("renderer", {}))
    renderer_config.setdefault("prompt_version", config.get("vlm", {}).get("prompt_version", "compact_v1"))
    renderer = SituationRenderer(renderer_config, config.get("thresholds", {}))

    rng = np.random.RandomState(seed)
    candidates = []
    previous_metrics = {}
    previous_scores = {}
    neutral_steps = {}
    agent_ids = _agent_ids(env, agent_side)
    rollout_step = 0

    while len(candidates) < max_candidates:
        if rollout_step % sample_interval == 0:
            for agent_id in agent_ids:
                state = extractor.extract(env, agent_id)
                metrics = compute_aircombat_metrics(state)
                labels, neutral_count = labeler.evaluate(
                    metrics,
                    previous_metrics=previous_metrics.get(agent_id),
                    previous_label_scores=previous_scores.get(agent_id),
                    neutral_steps=neutral_steps.get(agent_id, 0),
                )
                neutral_steps[agent_id] = neutral_count
                previous_metrics[agent_id] = metrics
                previous_scores[agent_id] = {name: value["score"] for name, value in labels.items()}
                scores = _category_scores(labels)
                candidates.append({
                    "state": state,
                    "metrics": metrics,
                    "labels": labels,
                    "agent_id": agent_id,
                    "rollout_step": rollout_step,
                    "category_scores": scores,
                    "mixed_score": _mixed_score(scores),
                    "normal_score": _normal_score(scores),
                    "random_tiebreak": float(rng.rand()),
                })
                if len(candidates) >= max_candidates:
                    break
        actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
        _, _, dones, _ = env.step(actions)
        rollout_step += 1
        if np.all(dones):
            env.reset()
            previous_metrics.clear()
            previous_scores.clear()
            neutral_steps.clear()
    env.close()

    selected, selection_summary = _select_formal(candidates, target_total)
    index_path = os.path.join(output_dir, "index.jsonl")
    category_counts = {name: 0 for name in FORMAL_CATEGORIES + ["fill"]}
    with open(index_path, "w", encoding="utf-8") as index_file:
        for idx, item in enumerate(selected):
            sample_id = "sample_{:06d}".format(idx)
            sample_dir = os.path.abspath(os.path.join(output_dir, sample_id))
            os.makedirs(sample_dir, exist_ok=True)
            situation_type = item.get("situation_type", item.get("selected_for", "fill"))
            metadata = renderer.render_sample(
                item["state"],
                item["metrics"],
                item["labels"],
                sample_dir,
                sample_id=sample_id,
                image_filename="image.png",
                metadata_filename="metadata.json",
            )
            metadata.update({
                "situation_type": situation_type,
                "selected_by_label": item.get("selected_for"),
                "selection_score": float(item.get("selection_score", 0.0)),
                "rollout_step": int(item["rollout_step"]),
                "category_scores": item["category_scores"],
            })
            _write_json(metadata["metadata_path"], metadata)
            _write_json(os.path.join(sample_dir, "physical_labels.json"), item["labels"])
            _write_json(os.path.join(sample_dir, "metrics.json"), metrics_for_logging(item["metrics"]))
            _write_json(os.path.join(sample_dir, "env_state_snapshot.json"), _state_snapshot(item["state"]))
            _write_json(os.path.join(sample_dir, "selection.json"), {
                "sample_id": sample_id,
                "situation_type": situation_type,
                "selected_by_label": item.get("selected_for"),
                "selection_score": float(item.get("selection_score", 0.0)),
                "agent_id": item["agent_id"],
                "rollout_step": item["rollout_step"],
                "category_scores": item["category_scores"],
            })
            category_counts[situation_type] = category_counts.get(situation_type, 0) + 1
            index_file.write(json.dumps(to_jsonable({
                "sample_id": sample_id,
                "situation_type": situation_type,
                "selected_by_label": item.get("selected_for"),
                "selection_score": item.get("selection_score", 0.0),
                "agent_id": item["agent_id"],
                "rollout_step": item["rollout_step"],
                "image_path": metadata["image_path"],
                "metadata_path": metadata["metadata_path"],
                "state_hash": metadata["state_hash"],
                "image_hash": metadata["image_hash"],
                "category_scores": item["category_scores"],
            }), sort_keys=True) + "\n")

    summary = {
        "scenario_name": scenario_name,
        "output_dir": os.path.abspath(output_dir),
        "target_total": target_total,
        "num_samples": len(selected),
        "num_candidates": len(candidates),
        "sample_interval": sample_interval,
        "index_path": os.path.abspath(index_path),
        "category_counts": category_counts,
        "selection_summary": selection_summary,
    }
    _write_json(os.path.join(output_dir, "dataset_summary.json"), summary)
    return summary


def _select_formal(candidates, target_total):
    quota = int(math.ceil(float(target_total) / len(FORMAL_CATEGORIES)))
    selected = []
    used = set()
    summary = {"quota_per_category": quota, "shortfalls": {}, "picked": {}}

    for label in LABEL_NAMES:
        picked = _pick_ranked(
            selected,
            used,
            candidates,
            target_total,
            quota,
            label,
            lambda item, label=label: item["category_scores"][label],
            minimum_score=0.05,
        )
        summary["picked"][label] = picked
        summary["shortfalls"][label] = max(0, quota - picked)
        if len(selected) >= target_total:
            break

    if len(selected) < target_total:
        picked = _pick_ranked(
            selected,
            used,
            candidates,
            target_total,
            quota,
            "mixed_ambiguous",
            lambda item: item["mixed_score"],
            minimum_score=0.15,
        )
        summary["picked"]["mixed_ambiguous"] = picked
        summary["shortfalls"]["mixed_ambiguous"] = max(0, quota - picked)

    if len(selected) < target_total:
        picked = _pick_ranked(
            selected,
            used,
            candidates,
            target_total,
            quota,
            "normal_rollout",
            lambda item: item["normal_score"],
            minimum_score=0.0,
        )
        summary["picked"]["normal_rollout"] = picked
        summary["shortfalls"]["normal_rollout"] = max(0, quota - picked)

    if len(selected) < target_total:
        ranked = sorted(
            candidates,
            key=lambda item: (
                max(item["category_scores"].values()),
                item["mixed_score"],
                item["random_tiebreak"],
            ),
            reverse=True,
        )
        for item in ranked:
            key = _candidate_key(item)
            if key in used:
                continue
            clone = dict(item)
            clone["selected_for"] = "fill"
            clone["situation_type"] = "fill"
            clone["selection_score"] = float(max(item["category_scores"].values()))
            selected.append(clone)
            used.add(key)
            if len(selected) >= target_total:
                break
    summary["picked"]["fill"] = sum(1 for item in selected if item.get("situation_type") == "fill")
    return selected[:target_total], summary


def _pick_ranked(selected, used, candidates, target_total, quota, category, score_fn, minimum_score):
    ranked = sorted(
        candidates,
        key=lambda item: (score_fn(item), item["random_tiebreak"]),
        reverse=True,
    )
    picked = 0
    for item in ranked:
        if len(selected) >= target_total or picked >= quota:
            break
        score = float(score_fn(item))
        if score < minimum_score:
            break
        key = _candidate_key(item)
        if key in used:
            continue
        clone = dict(item)
        clone["selected_for"] = category if category in LABEL_NAMES else None
        clone["situation_type"] = category
        clone["selection_score"] = score
        selected.append(clone)
        used.add(key)
        picked += 1
    return picked


def _category_scores(labels):
    return {name: float(labels[name]["score"]) for name in LABEL_NAMES}


def _mixed_score(scores):
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if values.size == 0:
        return 0.0
    midness = np.mean(np.maximum(0.0, 1.0 - np.abs(values - 0.5) * 2.0))
    active = float(np.sum(values > 0.15))
    return float(0.7 * midness + 0.3 * min(active / 3.0, 1.0))


def _normal_score(scores):
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if values.size == 0:
        return 0.0
    max_score = float(np.max(values))
    neutral = float(scores.get("neutral_stalemate", 0.0))
    return float(0.7 * (1.0 - max_score) + 0.3 * neutral)


def _candidate_key(item):
    state = item["state"]
    ego = tuple(round(float(x), 1) for x in np.asarray(state["ego"]["position"]).reshape(-1))
    enemy = tuple(round(float(x), 1) for x in np.asarray(state["enemy"]["position"]).reshape(-1))
    return (item["agent_id"], ego, enemy)


def _agent_ids(env, agent_side):
    if agent_side == "ego":
        return [env.ego_ids[0]]
    return (env.ego_ids + env.enm_ids)[: env.num_agents]


def _state_snapshot(state):
    return {
        "agent_id": state["agent_id"],
        "enemy_id": state["enemy_id"],
        "ego": state["ego"],
        "enemy": state["enemy"],
        "relative": state["relative"],
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
