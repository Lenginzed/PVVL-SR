#!/usr/bin/env python
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv

from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.logger import to_jsonable
from lag_extensions.reward_shaping.situation_renderer import SituationRenderer
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler


def main():
    parser = argparse.ArgumentParser(description="Build a label-balanced 1v1 semantic diagnostic dataset.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--output-dir", default="scripts/results/aircombat_semantic_dataset_stage6_balanced30")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml")
    parser.add_argument("--target-total", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--sample-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--agent-side", choices=["ego", "all"], default="ego")
    args = parser.parse_args()

    if args.env_name != "SingleCombat":
        raise NotImplementedError("Only SingleCombat is supported for this diagnostic builder.")
    summary = build_balanced_dataset(
        scenario_name=args.scenario_name,
        output_dir=args.output_dir,
        config_path=args.config,
        target_total=args.target_total,
        max_candidates=args.max_candidates,
        sample_interval=args.sample_interval,
        seed=args.seed,
        agent_side=args.agent_side,
    )
    print(summary)


def build_balanced_dataset(
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
                candidates.append({
                    "state": state,
                    "metrics": metrics,
                    "labels": labels,
                    "agent_id": agent_id,
                    "rollout_step": rollout_step,
                    "category_scores": _category_scores(labels),
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

    selected = _select_balanced(candidates, target_total)
    index_path = os.path.join(output_dir, "index.jsonl")
    category_counts = {name: 0 for name in LABEL_NAMES}
    with open(index_path, "w", encoding="utf-8") as index_file:
        for idx, item in enumerate(selected):
            sample_id = "sample_{:06d}".format(idx)
            sample_dir = os.path.abspath(os.path.join(output_dir, sample_id))
            os.makedirs(sample_dir, exist_ok=True)
            metadata = renderer.render_sample(
                item["state"],
                item["metrics"],
                item["labels"],
                sample_dir,
                sample_id=sample_id,
                image_filename="image.png",
                metadata_filename="metadata.json",
            )
            _write_json(os.path.join(sample_dir, "physical_labels.json"), item["labels"])
            _write_json(os.path.join(sample_dir, "metrics.json"), metrics_for_logging(item["metrics"]))
            _write_json(os.path.join(sample_dir, "env_state_snapshot.json"), _state_snapshot(item["state"]))
            category = item.get("selected_for", "unassigned")
            if category in category_counts:
                category_counts[category] += 1
            index_file.write(json.dumps(to_jsonable({
                "sample_id": sample_id,
                "selected_for": category,
                "agent_id": item["agent_id"],
                "rollout_step": item["rollout_step"],
                "image_path": metadata["image_path"],
                "metadata_path": metadata["metadata_path"],
                "state_hash": metadata["state_hash"],
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
    }
    _write_json(os.path.join(output_dir, "dataset_summary.json"), summary)
    return summary


def _select_balanced(candidates, target_total):
    quota = int(math.ceil(target_total / len(LABEL_NAMES)))
    selected = []
    used = set()
    for label in LABEL_NAMES:
        ranked = sorted(candidates, key=lambda item: item["category_scores"][label], reverse=True)
        picked = 0
        for item in ranked:
            key = _candidate_key(item)
            if key in used:
                continue
            if item["category_scores"][label] <= 0.0 and picked > 0:
                break
            clone = dict(item)
            clone["selected_for"] = label
            selected.append(clone)
            used.add(key)
            picked += 1
            if picked >= quota or len(selected) >= target_total:
                break
        if len(selected) >= target_total:
            break
    if len(selected) < target_total:
        ranked = sorted(candidates, key=lambda item: max(item["category_scores"].values()), reverse=True)
        for item in ranked:
            key = _candidate_key(item)
            if key in used:
                continue
            clone = dict(item)
            clone["selected_for"] = "fill"
            selected.append(clone)
            used.add(key)
            if len(selected) >= target_total:
                break
    return selected[:target_total]


def _category_scores(labels):
    return {name: float(labels[name]["score"]) for name in LABEL_NAMES}


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
