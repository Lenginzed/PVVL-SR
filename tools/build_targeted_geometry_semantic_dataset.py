#!/usr/bin/env python
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.logger import to_jsonable
from lag_extensions.reward_shaping.situation_renderer import SituationRenderer
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler


SITUATION_TYPES = [
    "ego_tail_advantage",
    "enemy_tail_threat",
    "effective_attack_window",
    "near_attack_but_bad_angle",
    "close_but_overshoot",
    "head_on",
    "beam_aspect",
    "neutral_far",
]


def main():
    parser = argparse.ArgumentParser(description="Build synthetic targeted geometry samples for missing air-combat semantics.")
    parser.add_argument("--output-dir", default="scripts/results/aircombat_semantic_dataset_stage8_targeted_geometry80")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml")
    parser.add_argument("--samples-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    summary = build_dataset(args.output_dir, args.config, args.samples_per_type, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_dataset(output_dir, config_path, samples_per_type, seed):
    config = load_reward_shaping_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    labeler = TacticalLabeler(config.get("thresholds", {}))
    renderer_config = dict(config.get("renderer", {}))
    renderer_config.setdefault("prompt_version", config.get("vlm", {}).get("prompt_version", "compact_v1"))
    renderer = SituationRenderer(renderer_config, config.get("thresholds", {}))

    sample_index = 0
    counts = {name: 0 for name in SITUATION_TYPES}
    hard_counts = {name: {label: 0 for label in LABEL_NAMES} for name in SITUATION_TYPES}
    index_path = os.path.join(output_dir, "index.jsonl")
    with open(index_path, "w", encoding="utf-8") as index_file:
        for situation_type in SITUATION_TYPES:
            for local_idx in range(samples_per_type):
                sample_id = "sample_{:06d}".format(sample_index)
                sample_dir = os.path.abspath(os.path.join(output_dir, sample_id))
                os.makedirs(sample_dir, exist_ok=True)
                state = _synthetic_state(situation_type, local_idx, rng)
                metrics = compute_aircombat_metrics(state)
                labels, _ = labeler.evaluate(metrics)
                metadata = renderer.render_sample(
                    state,
                    metrics,
                    labels,
                    sample_dir,
                    sample_id=sample_id,
                    image_filename="image.png",
                    metadata_filename="metadata.json",
                )
                metadata.update({
                    "synthetic": True,
                    "situation_type": situation_type,
                    "selected_by_label": _selected_label(situation_type),
                    "synthetic_generation": {
                        "version": "targeted_geometry_v1",
                        "local_index": local_idx,
                        "description": _description(situation_type),
                    },
                })
                _write_json(metadata["metadata_path"], metadata)
                _write_json(os.path.join(sample_dir, "physical_labels.json"), labels)
                _write_json(os.path.join(sample_dir, "metrics.json"), metrics_for_logging(metrics))
                _write_json(os.path.join(sample_dir, "env_state_snapshot.json"), _state_snapshot(state))
                _write_json(os.path.join(sample_dir, "selection.json"), {
                    "sample_id": sample_id,
                    "situation_type": situation_type,
                    "selected_by_label": _selected_label(situation_type),
                    "synthetic": True,
                })
                counts[situation_type] += 1
                for label in LABEL_NAMES:
                    hard_counts[situation_type][label] += int(bool(labels[label]["hard"]))
                index_file.write(json.dumps(to_jsonable({
                    "sample_id": sample_id,
                    "situation_type": situation_type,
                    "selected_by_label": _selected_label(situation_type),
                    "synthetic": True,
                    "image_path": metadata["image_path"],
                    "metadata_path": metadata["metadata_path"],
                    "state_hash": metadata["state_hash"],
                    "image_hash": metadata["image_hash"],
                }), sort_keys=True) + "\n")
                sample_index += 1

    summary = {
        "output_dir": os.path.abspath(output_dir),
        "num_samples": sample_index,
        "samples_per_type": samples_per_type,
        "synthetic_count": sample_index,
        "synthetic_ratio": 1.0,
        "situation_counts": counts,
        "hard_counts_by_type": hard_counts,
        "index_path": os.path.abspath(index_path),
    }
    _write_json(os.path.join(output_dir, "dataset_summary.json"), summary)
    return summary


def _synthetic_state(situation_type, index, rng):
    altitude = 6000.0 + rng.uniform(-500.0, 500.0)
    enemy_altitude = altitude + rng.uniform(-250.0, 250.0)
    speed = 250.0 + rng.uniform(-15.0, 15.0)
    enemy_speed = 230.0 + rng.uniform(-15.0, 15.0)
    distance = _distance_for_type(situation_type, rng)
    north = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    south = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    east = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    west = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    ego_position = np.array([0.0, 0.0, altitude], dtype=np.float64)
    enemy_position = np.array([distance, 0.0, enemy_altitude], dtype=np.float64)
    ego_heading = north.copy()
    enemy_heading = north.copy()

    if situation_type == "ego_tail_advantage":
        enemy_position = np.array([distance, rng.uniform(-120.0, 120.0), enemy_altitude])
        ego_heading = _towards(ego_position, enemy_position)
        enemy_heading = ego_heading.copy()
        speed = enemy_speed + 20.0 + rng.uniform(0.0, 20.0)
    elif situation_type == "enemy_tail_threat":
        enemy_position = np.array([-distance, rng.uniform(-120.0, 120.0), enemy_altitude])
        ego_heading = north.copy()
        enemy_heading = _towards(enemy_position, ego_position)
        enemy_speed = speed + 10.0 + rng.uniform(0.0, 20.0)
    elif situation_type == "effective_attack_window":
        enemy_position = np.array([distance, rng.uniform(-80.0, 80.0), enemy_altitude])
        ego_heading = _towards(ego_position, enemy_position)
        enemy_heading = ego_heading.copy()
        speed = enemy_speed + 30.0 + rng.uniform(0.0, 25.0)
    elif situation_type == "near_attack_but_bad_angle":
        enemy_position = np.array([distance, rng.uniform(-100.0, 100.0), enemy_altitude])
        ego_heading = east.copy()
        enemy_heading = north.copy()
    elif situation_type == "close_but_overshoot":
        enemy_position = np.array([distance, rng.uniform(-30.0, 30.0), enemy_altitude])
        ego_heading = north.copy()
        enemy_heading = north.copy()
        speed = enemy_speed + 50.0
    elif situation_type == "head_on":
        enemy_position = np.array([distance, rng.uniform(-150.0, 150.0), enemy_altitude])
        ego_heading = _towards(ego_position, enemy_position)
        enemy_heading = _towards(enemy_position, ego_position)
        enemy_speed = speed + rng.uniform(-10.0, 10.0)
    elif situation_type == "beam_aspect":
        enemy_position = np.array([rng.uniform(-200.0, 200.0), distance, enemy_altitude])
        ego_heading = north.copy()
        enemy_heading = west.copy()
    elif situation_type == "neutral_far":
        enemy_position = np.array([distance, rng.uniform(1500.0, 2500.0), enemy_altitude])
        ego_heading = north.copy()
        enemy_heading = east.copy()
        enemy_speed = speed + rng.uniform(-20.0, 20.0)

    ego_velocity = ego_heading * speed
    enemy_velocity = enemy_heading * enemy_speed
    state = {
        "agent_id": "synthetic_ego",
        "enemy_id": "synthetic_enemy",
        "ego": _aircraft("synthetic_ego", ego_position, ego_velocity, ego_heading),
        "enemy": _aircraft("synthetic_enemy", enemy_position, enemy_velocity, enemy_heading),
    }
    rel = state["enemy"]["position"] - state["ego"]["position"]
    rel_dist = float(np.linalg.norm(rel))
    los = rel / max(rel_dist, 1e-8)
    rel_vel = state["enemy"]["velocity"] - state["ego"]["velocity"]
    state["relative"] = {
        "relative_position": rel,
        "distance": rel_dist,
        "line_of_sight_unit_vector": los,
        "relative_velocity": rel_vel,
        "distance_rate": float(np.dot(rel_vel, los)),
    }
    return state


def _distance_for_type(situation_type, rng):
    if situation_type == "close_but_overshoot":
        return float(rng.uniform(220.0, 450.0))
    if situation_type == "neutral_far":
        return float(rng.uniform(10000.0, 13000.0))
    if situation_type in ("effective_attack_window", "near_attack_but_bad_angle", "head_on", "beam_aspect"):
        return float(rng.uniform(2500.0, 5000.0))
    return float(rng.uniform(2500.0, 4500.0))


def _aircraft(uid, position, velocity, heading):
    heading = _unit(heading)
    velocity = np.asarray(velocity, dtype=np.float64)
    return {
        "uid": uid,
        "position": np.asarray(position, dtype=np.float64),
        "velocity": velocity,
        "speed": float(np.linalg.norm(velocity)),
        "altitude": float(np.asarray(position, dtype=np.float64)[2]),
        "roll": 0.0,
        "pitch": math.asin(float(np.clip(heading[2], -1.0, 1.0))),
        "yaw": math.atan2(float(heading[1]), float(heading[0])),
        "heading_vector": heading,
        "missile_warning": False,
        "num_under_missiles": 0,
        "num_launch_missiles": 0,
        "lock": False,
        "is_alive": True,
        "is_crash": False,
        "is_shotdown": False,
    }


def _towards(src, dst):
    return _unit(np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64))


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _selected_label(situation_type):
    mapping = {
        "ego_tail_advantage": "ego_tail_advantage",
        "enemy_tail_threat": "enemy_tail_threat",
        "effective_attack_window": "effective_attack_window",
        "head_on": "enemy_missile_threat_zone",
        "neutral_far": "neutral_stalemate",
    }
    return mapping.get(situation_type)


def _description(situation_type):
    descriptions = {
        "ego_tail_advantage": "Ego is behind enemy, heading toward enemy.",
        "enemy_tail_threat": "Enemy is behind ego, heading toward ego.",
        "effective_attack_window": "Ego is behind enemy, in range, aligned, and closing.",
        "near_attack_but_bad_angle": "Range is suitable but ego aspect/aim angle is poor.",
        "close_but_overshoot": "Ego is aligned but too close, representing overshoot risk.",
        "head_on": "Both aircraft point toward each other.",
        "beam_aspect": "Side crossing / beam aspect.",
        "neutral_far": "Far neutral geometry without a clear immediate advantage.",
    }
    return descriptions.get(situation_type, situation_type)


def _state_snapshot(state):
    return {
        "agent_id": state["agent_id"],
        "enemy_id": state["enemy_id"],
        "ego": state["ego"],
        "enemy": state["enemy"],
        "relative": state["relative"],
        "synthetic": True,
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
