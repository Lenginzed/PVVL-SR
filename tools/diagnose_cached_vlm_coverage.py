#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv

from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.logger import to_jsonable
from lag_extensions.reward_shaping.semantic_cache_retrieval import (
    NearestSemanticCache,
    distribution_difference,
    feature_dict_from_metrics,
    feature_vector_from_metrics,
    load_dataset_feature_rows,
    summarize_label_distribution,
    write_json,
)
from lag_extensions.reward_shaping.semantic_reward import state_cache_key
from lag_extensions.reward_shaping.situation_renderer import SituationRenderer
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler
from lag_extensions.reward_shaping.vlm_cache import VLMScoreCache


def main():
    parser = argparse.ArgumentParser(description="Diagnose exact and nearest cached_vlm coverage on PPO-style rollouts.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-dir", default=None, help="Optional cached dataset dir for distribution and nearest diagnostics.")
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--policy", choices=["random"], default="random")
    parser.add_argument("--output-dir", default="scripts/results/stage7_cache_coverage_random1000")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--agent-side", choices=["ego", "all"], default="ego")
    parser.add_argument("--nearest-threshold", type=float, default=2.5)
    parser.add_argument("--no-render-images", action="store_true", default=False)
    parser.add_argument("--render-limit", type=int, default=None)
    args = parser.parse_args()

    if args.env_name != "SingleCombat":
        raise NotImplementedError("Only SingleCombat coverage diagnosis is implemented.")
    summary = diagnose_coverage(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def diagnose_coverage(args):
    config = load_reward_shaping_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    cache = VLMScoreCache(args.cache, enabled=True)
    dataset_rows = load_dataset_feature_rows(args.dataset_dir, args.cache) if args.dataset_dir else []
    nearest_cache = NearestSemanticCache(dataset_rows, threshold=args.nearest_threshold) if dataset_rows else None

    env = SingleCombatEnv(args.scenario_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.reset()

    extractor = StateExtractor()
    labeler = TacticalLabeler(config.get("thresholds", {}))
    renderer_config = dict(config.get("renderer", {}))
    renderer_config.setdefault("prompt_version", config.get("vlm", {}).get("prompt_version", "compact_v1"))
    renderer = SituationRenderer(renderer_config, config.get("thresholds", {}))

    previous_metrics = {}
    previous_scores = {}
    neutral_steps = {}
    agent_ids = _agent_ids(env, args.agent_side)
    rollout_rows = []
    seen_state_hashes = set()
    rendered_count = 0
    exact_hits = 0
    nearest_hits = 0
    nearest_distances = []
    nearest_accepted_distances = []
    coverage_path = os.path.join(args.output_dir, "coverage_rollout.jsonl")
    render_images = not args.no_render_images
    render_limit = args.render_limit if args.render_limit is not None else args.rollout_steps * len(agent_ids)

    with open(coverage_path, "w", encoding="utf-8") as rollout_file:
        state_index = 0
        for rollout_step in range(args.rollout_steps):
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
                physical_scores = {name: float(labels[name]["score"]) for name in LABEL_NAMES}
                state_hash = state_cache_key(state)
                keys = [state_hash]
                metadata = {}
                image_path = None
                if render_images and rendered_count < render_limit:
                    sample_id = "rollout_{:06d}".format(state_index)
                    sample_dir = os.path.join(args.output_dir, "rollout_rendered", sample_id)
                    metadata = renderer.render_sample(
                        state,
                        metrics,
                        labels,
                        sample_dir,
                        sample_id=sample_id,
                        image_filename="image.png",
                        metadata_filename="metadata.json",
                    )
                    image_path = metadata.get("image_path")
                    rendered_count += 1
                    keys.extend([metadata.get("state_hash"), metadata.get("image_hash"), metadata.get("image_id")])

                cache_entry = None
                for key in keys:
                    if not key:
                        continue
                    cache_entry = cache.get(key)
                    if cache_entry is not None:
                        break
                exact_hit = cache_entry is not None
                if exact_hit:
                    exact_hits += 1
                seen_state_hashes.add(state_hash)

                feature_vector = feature_vector_from_metrics(metrics_for_logging(metrics), state)
                nearest_info = {}
                if nearest_cache is not None:
                    nearest_info = nearest_cache.lookup(feature_vector)
                    if nearest_info["nearest_distance"] is not None:
                        nearest_distances.append(float(nearest_info["nearest_distance"]))
                    if nearest_info["accepted"]:
                        nearest_hits += 1
                        nearest_accepted_distances.append(float(nearest_info["nearest_distance"]))

                row = {
                    "state_index": state_index,
                    "rollout_step": rollout_step,
                    "agent_id": agent_id,
                    "state_hash": state_hash,
                    "image_hash": metadata.get("image_hash"),
                    "image_path": image_path,
                    "exact_cache_hit": exact_hit,
                    "fallback_physical": not exact_hit,
                    "physical_scores": physical_scores,
                    "feature_dict": feature_dict_from_metrics(metrics_for_logging(metrics), state),
                    "nearest": {
                        "accepted": bool(nearest_info.get("accepted", False)),
                        "nearest_distance": nearest_info.get("nearest_distance"),
                        "nearest_sample_id": nearest_info.get("nearest_sample_id"),
                    } if nearest_cache is not None else None,
                }
                rollout_rows.append(row)
                rollout_file.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
                state_index += 1
            actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
            _, _, dones, _ = env.step(actions)
            if np.all(dones):
                env.reset()
                previous_metrics.clear()
                previous_scores.clear()
                neutral_steps.clear()
    env.close()

    total = len(rollout_rows)
    rollout_distribution = summarize_label_distribution(rollout_rows)
    cached_distribution = summarize_label_distribution(dataset_rows)
    summary = {
        "env_name": args.env_name,
        "scenario_name": args.scenario_name,
        "policy": args.policy,
        "rollout_steps": args.rollout_steps,
        "total_rollout_states": total,
        "rendered_image_count": rendered_count,
        "exact_cache_hit_count": exact_hits,
        "exact_cache_hit_rate": exact_hits / total if total else 0.0,
        "fallback_physical_count": total - exact_hits,
        "fallback_physical_rate": (total - exact_hits) / total if total else 0.0,
        "unique_situation_count": len(seen_state_hashes),
        "repeated_situation_count": total - len(seen_state_hashes),
        "cache_entry_count": len(cache.entries),
        "coverage_path": os.path.abspath(coverage_path),
        "output_dir": os.path.abspath(args.output_dir),
        "rollout_label_distribution": rollout_distribution,
        "cached_dataset_label_distribution": cached_distribution if dataset_rows else None,
        "distribution_difference_rollout_minus_cached": distribution_difference(rollout_distribution, cached_distribution) if dataset_rows else None,
        "nearest_retrieval": {
            "enabled": nearest_cache is not None,
            "threshold": args.nearest_threshold,
            "nearest_hit_count": nearest_hits,
            "nearest_hit_rate": nearest_hits / total if total else 0.0,
            "nearest_rejected_count": total - nearest_hits,
            "nearest_rejected_rate": (total - nearest_hits) / total if total else 0.0,
            "average_nearest_distance": _mean(nearest_distances),
            "average_accepted_nearest_distance": _mean(nearest_accepted_distances),
        },
    }
    if summary["exact_cache_hit_rate"] <= 0.01:
        summary["diagnosis"] = "Exact cached_vlm hits are near zero; this is expected in continuous state space and motivates nearest retrieval or a surrogate semantic reward network."
    else:
        summary["diagnosis"] = "Exact cached_vlm coverage is non-trivial for this rollout."
    write_json(os.path.join(args.output_dir, "coverage_summary.json"), summary)
    return summary


def _agent_ids(env, agent_side):
    if agent_side == "ego":
        return [env.ego_ids[0]]
    return (env.ego_ids + env.enm_ids)[: env.num_agents]


def _mean(values):
    return float(np.mean(values)) if values else None


if __name__ == "__main__":
    main()
