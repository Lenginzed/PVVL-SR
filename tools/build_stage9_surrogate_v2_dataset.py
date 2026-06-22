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


CATEGORIES = [
    "energy_advantage_boundary",
    "energy_disadvantage_boundary",
    "neutral_far_energy_boundary",
    "ego_tail_advantage_boundary",
    "effective_attack_window_boundary",
    "enemy_missile_threat_boundary",
]


def main():
    parser = argparse.ArgumentParser(description="Build stage9 boundary augmented semantic dataset for surrogate_v2.")
    parser.add_argument("--output-dir", default="scripts/results/aircombat_semantic_dataset_stage9_boundary_augmented")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml")
    parser.add_argument("--samples-per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    summary = build_dataset(args.output_dir, args.config, args.samples_per_category, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_dataset(output_dir, config_path, samples_per_category, seed):
    config = load_reward_shaping_config(config_path)
    thresholds = config.get("thresholds", {})
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    labeler = TacticalLabeler(thresholds)
    renderer_config = dict(config.get("renderer", {}))
    renderer_config.setdefault("prompt_version", config.get("vlm", {}).get("prompt_version", "compact_v1"))
    renderer = SituationRenderer(renderer_config, thresholds)
    index_path = os.path.join(output_dir, "index.jsonl")
    counts = {name: 0 for name in CATEGORIES}
    hard_counts = {name: {label: 0 for label in LABEL_NAMES} for name in CATEGORIES}

    sample_index = 0
    with open(index_path, "w", encoding="utf-8") as index_file:
        for category in CATEGORIES:
            for local_idx in range(samples_per_category):
                subtype = _subtype(category, local_idx)
                state = _make_state(category, subtype, local_idx, rng, thresholds)
                metrics = compute_aircombat_metrics(state)
                labels, _ = labeler.evaluate(metrics)
                sample_id = "sample_{:06d}".format(sample_index)
                sample_dir = os.path.abspath(os.path.join(output_dir, sample_id))
                os.makedirs(sample_dir, exist_ok=True)
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
                    "situation_type": category,
                    "boundary_subtype": subtype,
                    "selected_by_label": _selected_label(category),
                    "synthetic_generation": {
                        "version": "stage9_boundary_augmented_v1",
                        "category": category,
                        "subtype": subtype,
                        "local_index": local_idx,
                    },
                })
                _write_json(metadata["metadata_path"], metadata)
                _write_json(os.path.join(sample_dir, "physical_labels.json"), labels)
                _write_json(os.path.join(sample_dir, "metrics.json"), metrics_for_logging(metrics))
                _write_json(os.path.join(sample_dir, "env_state_snapshot.json"), _state_snapshot(state))
                _write_json(os.path.join(sample_dir, "selection.json"), {
                    "sample_id": sample_id,
                    "situation_type": category,
                    "boundary_subtype": subtype,
                    "selected_by_label": _selected_label(category),
                    "synthetic": True,
                })
                for label in LABEL_NAMES:
                    hard_counts[category][label] += int(labels[label]["hard"])
                counts[category] += 1
                index_file.write(json.dumps(to_jsonable({
                    "sample_id": sample_id,
                    "situation_type": category,
                    "boundary_subtype": subtype,
                    "synthetic": True,
                    "selected_by_label": _selected_label(category),
                    "image_path": metadata["image_path"],
                    "metadata_path": metadata["metadata_path"],
                    "state_hash": metadata["state_hash"],
                    "image_hash": metadata["image_hash"],
                }), sort_keys=True) + "\n")
                sample_index += 1

    summary = {
        "output_dir": os.path.abspath(output_dir),
        "num_samples": sample_index,
        "synthetic_count": sample_index,
        "synthetic_ratio": 1.0,
        "samples_per_category": samples_per_category,
        "category_counts": counts,
        "hard_counts_by_category": hard_counts,
        "index_path": os.path.abspath(index_path),
    }
    _write_json(os.path.join(output_dir, "dataset_summary.json"), summary)
    return summary


def _subtype(category, index):
    subtype_map = {
        "energy_advantage_boundary": ["deltaE_near_zero", "deltaE_small_positive", "deltaE_mid_positive", "deltaE_large_positive"],
        "energy_disadvantage_boundary": ["deltaE_near_zero", "deltaE_small_negative", "deltaE_mid_negative", "deltaE_large_negative"],
        "neutral_far_energy_boundary": ["neutral_high_energy_adv", "neutral_high_energy_dis", "neutral_small_energy_adv", "neutral_small_energy_dis"],
        "ego_tail_advantage_boundary": ["tail_inside", "tail_outside", "good_aim_bad_tail", "good_tail_bad_aim"],
        "effective_attack_window_boundary": ["below_min_range", "inside_window", "above_max_range", "angle_good", "angle_bad"],
        "enemy_missile_threat_boundary": ["enemy_aim_inside", "enemy_aim_outside", "near_min_range", "near_max_range", "outside_max_range"],
    }
    values = subtype_map[category]
    return values[index % len(values)]


def _make_state(category, subtype, index, rng, thresholds):
    base_alt = 6000.0 + rng.uniform(-200.0, 200.0)
    base_speed = 250.0 + rng.uniform(-5.0, 5.0)
    if category.startswith("energy"):
        delta = _delta_energy(subtype, rng)
        return _state_with_energy_delta(delta, base_alt, base_speed, distance=6500.0, neutral=True, rng=rng)
    if category == "neutral_far_energy_boundary":
        delta = _delta_energy(subtype, rng)
        return _state_with_energy_delta(delta, base_alt, base_speed, distance=11000.0, neutral=True, rng=rng)
    if category == "ego_tail_advantage_boundary":
        return _tail_boundary_state(subtype, base_alt, base_speed, rng, thresholds)
    if category == "effective_attack_window_boundary":
        return _attack_boundary_state(subtype, base_alt, base_speed, rng, thresholds)
    if category == "enemy_missile_threat_boundary":
        return _missile_boundary_state(subtype, base_alt, base_speed, rng, thresholds)
    raise ValueError(category)


def _delta_energy(subtype, rng):
    if subtype.endswith("near_zero"):
        return float(rng.uniform(-1500.0, 1500.0))
    if "small_positive" in subtype or "small_energy_adv" in subtype:
        return float(rng.uniform(2500.0, 10000.0))
    if "mid_positive" in subtype:
        return float(rng.uniform(15000.0, 30000.0))
    if "large_positive" in subtype or "high_energy_adv" in subtype:
        return float(rng.uniform(40000.0, 65000.0))
    if "small_negative" in subtype or "small_energy_dis" in subtype:
        return float(rng.uniform(-10000.0, -2500.0))
    if "mid_negative" in subtype:
        return float(rng.uniform(-30000.0, -15000.0))
    if "large_negative" in subtype or "high_energy_dis" in subtype:
        return float(rng.uniform(-65000.0, -40000.0))
    return 0.0


def _state_with_energy_delta(delta_e, base_alt, base_speed, distance, neutral, rng):
    enemy_alt = base_alt
    ego_alt = base_alt + delta_e / 9.81
    ego_alt = float(np.clip(ego_alt, 1500.0, 12000.0))
    adjusted_delta = 9.81 * (ego_alt - enemy_alt)
    speed_adjust = max(abs(delta_e - adjusted_delta), 0.0)
    ego_speed = base_speed
    enemy_speed = base_speed
    if delta_e > adjusted_delta:
        ego_speed = math.sqrt(max(base_speed ** 2 + 2.0 * speed_adjust, 10.0))
    elif delta_e < adjusted_delta:
        enemy_speed = math.sqrt(max(base_speed ** 2 + 2.0 * speed_adjust, 10.0))
    lateral = rng.uniform(1800.0, 2600.0) if neutral else rng.uniform(-100.0, 100.0)
    ego_pos = np.array([0.0, 0.0, ego_alt])
    enemy_pos = np.array([distance, lateral, enemy_alt])
    ego_heading = np.array([0.0, 1.0, 0.0])
    enemy_heading = np.array([1.0, 0.0, 0.0])
    return _state(ego_pos, enemy_pos, ego_heading, enemy_heading, ego_speed, enemy_speed)


def _tail_boundary_state(subtype, altitude, speed, rng, thresholds):
    theta_tail = math.radians(float(thresholds.get("theta_tail_deg", 45.0)))
    theta_aim = math.radians(float(thresholds.get("theta_aim_deg", 60.0)))
    if subtype == "tail_inside":
        tail_angle = theta_tail - math.radians(rng.uniform(2.0, 8.0))
        aim_error = math.radians(rng.uniform(0.0, 8.0))
    elif subtype == "tail_outside":
        tail_angle = theta_tail + math.radians(rng.uniform(2.0, 8.0))
        aim_error = math.radians(rng.uniform(0.0, 8.0))
    elif subtype == "good_aim_bad_tail":
        tail_angle = theta_tail + math.radians(rng.uniform(8.0, 20.0))
        aim_error = math.radians(rng.uniform(0.0, 5.0))
    else:
        tail_angle = theta_tail - math.radians(rng.uniform(5.0, 15.0))
        aim_error = theta_aim + math.radians(rng.uniform(5.0, 15.0))
    distance = float(rng.uniform(2500.0, 4800.0))
    u = _rotate_2d(np.array([1.0, 0.0, 0.0]), tail_angle)
    ego_pos = np.array([0.0, 0.0, altitude])
    enemy_pos = ego_pos + distance * u + np.array([0.0, 0.0, rng.uniform(-80.0, 80.0)])
    enemy_heading = np.array([1.0, 0.0, 0.0])
    ego_heading = _rotate_2d(_unit(enemy_pos - ego_pos), aim_error)
    return _state(ego_pos, enemy_pos, ego_heading, enemy_heading, speed + 20.0, speed)


def _attack_boundary_state(subtype, altitude, speed, rng, thresholds):
    d_min = float(thresholds.get("d_attack_min", 500.0))
    d_max = float(thresholds.get("d_attack_max", 8000.0))
    theta_launch = math.radians(float(thresholds.get("theta_launch_deg", 30.0)))
    if subtype == "below_min_range":
        distance = max(120.0, d_min - rng.uniform(20.0, 120.0))
        aim_error = math.radians(3.0)
    elif subtype == "above_max_range":
        distance = d_max + rng.uniform(50.0, 400.0)
        aim_error = math.radians(3.0)
    elif subtype == "angle_bad":
        distance = rng.uniform(2500.0, 5500.0)
        aim_error = theta_launch + math.radians(rng.uniform(3.0, 12.0))
    else:
        distance = rng.uniform(1800.0, 5200.0)
        aim_error = math.radians(rng.uniform(0.0, 6.0))
    ego_pos = np.array([0.0, 0.0, altitude])
    enemy_pos = np.array([distance, rng.uniform(-80.0, 80.0), altitude + rng.uniform(-80.0, 80.0)])
    los = _unit(enemy_pos - ego_pos)
    ego_heading = _rotate_2d(los, aim_error)
    enemy_heading = los.copy()
    return _state(ego_pos, enemy_pos, ego_heading, enemy_heading, speed + 35.0, speed)


def _missile_boundary_state(subtype, altitude, speed, rng, thresholds):
    d_min = float(thresholds.get("d_missile_min", 500.0))
    d_max = float(thresholds.get("d_missile_max", 14000.0))
    theta = math.radians(float(thresholds.get("theta_enemy_launch_deg", 30.0)))
    if subtype == "near_min_range":
        distance = d_min + rng.uniform(-80.0, 120.0)
        aim_error = math.radians(5.0)
    elif subtype == "near_max_range":
        distance = d_max - rng.uniform(80.0, 400.0)
        aim_error = math.radians(5.0)
    elif subtype == "outside_max_range":
        distance = d_max + rng.uniform(80.0, 600.0)
        aim_error = math.radians(5.0)
    elif subtype == "enemy_aim_outside":
        distance = rng.uniform(3000.0, 8000.0)
        aim_error = theta + math.radians(rng.uniform(2.0, 12.0))
    else:
        distance = rng.uniform(3000.0, 8000.0)
        aim_error = theta - math.radians(rng.uniform(2.0, 10.0))
    ego_pos = np.array([0.0, 0.0, altitude])
    enemy_pos = np.array([distance, rng.uniform(-100.0, 100.0), altitude + rng.uniform(-80.0, 80.0)])
    los_enemy_to_ego = _unit(ego_pos - enemy_pos)
    enemy_heading = _rotate_2d(los_enemy_to_ego, aim_error)
    ego_heading = _unit(enemy_pos - ego_pos)
    return _state(ego_pos, enemy_pos, ego_heading, enemy_heading, speed, speed + rng.uniform(-10.0, 10.0))


def _state(ego_pos, enemy_pos, ego_heading, enemy_heading, ego_speed, enemy_speed):
    ego_heading = _unit(ego_heading)
    enemy_heading = _unit(enemy_heading)
    state = {
        "agent_id": "synthetic_ego",
        "enemy_id": "synthetic_enemy",
        "ego": _aircraft("synthetic_ego", ego_pos, ego_heading * ego_speed, ego_heading),
        "enemy": _aircraft("synthetic_enemy", enemy_pos, enemy_heading * enemy_speed, enemy_heading),
    }
    rel = state["enemy"]["position"] - state["ego"]["position"]
    dist = float(np.linalg.norm(rel))
    los = rel / max(dist, 1e-8)
    rel_vel = state["enemy"]["velocity"] - state["ego"]["velocity"]
    state["relative"] = {
        "relative_position": rel,
        "distance": dist,
        "line_of_sight_unit_vector": los,
        "relative_velocity": rel_vel,
        "distance_rate": float(np.dot(rel_vel, los)),
    }
    return state


def _aircraft(uid, position, velocity, heading):
    heading = _unit(heading)
    velocity = np.asarray(velocity, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    return {
        "uid": uid,
        "position": position,
        "velocity": velocity,
        "speed": float(np.linalg.norm(velocity)),
        "altitude": float(position[2]),
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


def _rotate_2d(vec, angle):
    vec = _unit(vec)
    c, s = math.cos(angle), math.sin(angle)
    return _unit(np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]]))


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _selected_label(category):
    mapping = {
        "energy_advantage_boundary": "energy_advantage",
        "energy_disadvantage_boundary": "energy_disadvantage",
        "neutral_far_energy_boundary": "neutral_stalemate",
        "ego_tail_advantage_boundary": "ego_tail_advantage",
        "effective_attack_window_boundary": "effective_attack_window",
        "enemy_missile_threat_boundary": "enemy_missile_threat_zone",
    }
    return mapping.get(category)


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
