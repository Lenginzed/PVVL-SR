import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, payload) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: str, rows: Sequence[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {key: 0.0 for key in ("count", "mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "max")}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def heading_vector_from_deg(heading_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(float(heading_deg))
    pitch = np.deg2rad(float(pitch_deg))
    vec = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-8)


def make_aircraft_state(
    position_neu: Sequence[float],
    heading_deg: float,
    speed_mps: float = 250.0,
    altitude_m: Optional[float] = None,
    pitch_deg: float = 0.0,
) -> Dict:
    position = np.asarray(position_neu, dtype=np.float64)
    if altitude_m is None:
        altitude_m = float(position[2])
    heading = heading_vector_from_deg(heading_deg, pitch_deg=pitch_deg)
    velocity = heading * float(speed_mps)
    return {
        "id": "synthetic",
        "position": position,
        "velocity": velocity,
        "speed": float(np.linalg.norm(velocity)),
        "altitude": float(altitude_m),
        "attitude": np.array([0.0, np.deg2rad(float(pitch_deg)), np.deg2rad(float(heading_deg))], dtype=np.float64),
        "heading_vector": heading,
        "missile_warning": False,
        "lock": False,
    }


def make_pair_state(
    ego_position: Sequence[float],
    enemy_position: Sequence[float],
    ego_heading_deg: float,
    enemy_heading_deg: float,
    ego_speed: float = 260.0,
    enemy_speed: float = 240.0,
    altitude_m: float = 6000.0,
) -> Dict:
    ego = make_aircraft_state(ego_position, ego_heading_deg, speed_mps=ego_speed, altitude_m=altitude_m)
    enemy = make_aircraft_state(enemy_position, enemy_heading_deg, speed_mps=enemy_speed, altitude_m=altitude_m)
    ego["id"] = "ego"
    enemy["id"] = "enemy"
    return {"ego_id": "ego", "enemy_id": "enemy", "ego": ego, "enemy": enemy}


def evaluate_synthetic_state(state: Dict, thresholds: Dict) -> Dict:
    labeler = TacticalLabeler(thresholds)
    metrics = compute_aircombat_metrics(state)
    labels, _ = labeler.evaluate(metrics, neutral_steps=0)
    return {
        "metrics": metrics_for_logging(metrics),
        "labels": labels,
        "scores": {name: float(labels[name]["score"]) for name in LABEL_NAMES},
        "hard": {name: bool(labels[name]["hard"]) for name in LABEL_NAMES},
    }


def labels_from_metrics_stream(rows: Iterable[Dict], thresholds: Dict) -> List[Dict]:
    labeler = TacticalLabeler(thresholds)
    prev_metrics: Dict[str, Dict] = {}
    prev_scores: Dict[str, Dict[str, float]] = {}
    neutral_steps: Dict[str, int] = {}
    out = []
    for idx, row in enumerate(rows):
        agent_id = str(row.get("agent_id", "agent0"))
        metrics = coerce_metrics(row)
        labels, count = labeler.evaluate(
            metrics,
            previous_metrics=prev_metrics.get(agent_id),
            previous_label_scores=prev_scores.get(agent_id),
            neutral_steps=neutral_steps.get(agent_id, 0),
        )
        prev_metrics[agent_id] = metrics
        prev_scores[agent_id] = {name: float(labels[name]["score"]) for name in LABEL_NAMES}
        neutral_steps[agent_id] = count
        out_row = dict(row)
        out_row["sample_index"] = idx
        for name in LABEL_NAMES:
            out_row[name + "_hard"] = float(labels[name]["hard"])
            out_row[name + "_score"] = float(labels[name]["score"])
        out.append(out_row)
    return out


def coerce_metrics(row: Dict) -> Dict:
    return {
        "distance": safe_float(row.get("distance")),
        "distance_rate": safe_float(row.get("distance_rate")),
        "ego_aim_angle": _angle_rad(row, "ego_aim_angle"),
        "enemy_aim_angle": _angle_rad(row, "enemy_aim_angle"),
        "ego_tail_angle": _angle_rad(row, "ego_tail_angle"),
        "enemy_tail_angle": _angle_rad(row, "enemy_tail_angle"),
        "ego_specific_energy": safe_float(row.get("ego_specific_energy")),
        "enemy_specific_energy": safe_float(row.get("enemy_specific_energy")),
        "energy_difference": safe_float(row.get("energy_difference")),
        "enemy_lock": bool(row.get("enemy_lock", False)),
        "ego_missile_warning": bool(row.get("ego_missile_warning", False)),
    }


def _angle_rad(row: Dict, key: str) -> float:
    if key in row:
        return safe_float(row.get(key))
    deg_key = key + "_deg"
    if deg_key in row:
        return math.radians(safe_float(row.get(deg_key)))
    return 0.0


def parse_stage10_best_runs(stage10_dir: str, groups: Sequence[str]) -> Dict[str, Optional[str]]:
    seed_table = os.path.join(stage10_dir, "stage10_seed_table.csv")
    records = []
    if os.path.exists(seed_table):
        with open(seed_table, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("group") in groups and str(row.get("success", "")).lower() == "true":
                    records.append(row)
    best = {}
    for group in groups:
        candidates = [row for row in records if row.get("group") == group]
        if not candidates:
            best[group] = None
            continue
        candidates.sort(
            key=lambda row: (
                -safe_float(row.get("win_rate")),
                safe_float(row.get("draw_rate")),
                -safe_float(row.get("effective_attack_window_time_ratio_mean")),
                -safe_float(row.get("ego_tail_advantage_time_ratio_mean")),
            )
        )
        best[group] = candidates[0].get("run_dir")
    return best


def collect_reward_step_logs(stage10_dir: str, groups: Optional[Sequence[str]] = None) -> List[str]:
    paths = []
    for root, _, files in os.walk(stage10_dir):
        for name in files:
            if name.endswith("_steps.jsonl") and "reward_shaping" in root:
                path = os.path.join(root, name)
                if groups and _group_from_stage10_path(path) not in set(groups):
                    continue
                paths.append(path)
    return sorted(paths)


def iter_reward_metrics(paths: Sequence[str], limit: int = 0) -> Iterable[Dict]:
    count = 0
    for path in paths:
        group = _group_from_stage10_path(path)
        seed = _seed_from_stage10_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                metrics = row.get("metrics", {})
                metrics = dict(metrics)
                metrics["agent_id"] = row.get("agent_id")
                metrics["group"] = group
                metrics["seed"] = seed
                metrics["source_path"] = path
                yield metrics
                count += 1
                if limit and count >= limit:
                    return


def _group_from_stage10_path(path: str) -> str:
    parts = Path(path).parts
    if "stage10_multiseed_ppo" in parts:
        idx = parts.index("stage10_multiseed_ppo")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def _seed_from_stage10_path(path: str) -> int:
    for part in Path(path).parts:
        if part.startswith("seed"):
            return int(part.replace("seed", "") or 0)
    return -1


def env_metrics_for_agents(env, extractor: Optional[StateExtractor] = None) -> List[Dict]:
    extractor = extractor or StateExtractor()
    rows = []
    for agent_id in (env.ego_ids + env.enm_ids)[: env.num_agents]:
        state = extractor.extract(env, agent_id)
        metrics = metrics_for_logging(compute_aircombat_metrics(state))
        metrics["agent_id"] = agent_id
        rows.append(metrics)
    return rows
