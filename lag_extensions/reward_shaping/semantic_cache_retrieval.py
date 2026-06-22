import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import LABEL_NAMES, project_root
from .logger import to_jsonable
from .vlm_cache import VLMScoreCache


FEATURE_NAMES = [
    "distance",
    "ego_aim_angle",
    "enemy_aim_angle",
    "ego_tail_angle",
    "enemy_tail_angle",
    "distance_rate",
    "ego_speed",
    "enemy_speed",
    "ego_altitude",
    "enemy_altitude",
    "energy_difference",
    "altitude_difference",
    "speed_difference",
    "energy_ratio",
]


def feature_dict_from_metrics(metrics: Dict, state: Optional[Dict] = None) -> Dict[str, float]:
    """Build a compact continuous feature representation for cache diagnostics.

    These features are deliberately physics-level and renderer-independent. They
    are intended for nearest-neighbor coverage analysis and future surrogate
    training, not as a replacement for physical verification.
    """
    ego_altitude = _metric_or_state(metrics, state, "ego_altitude", ("ego", "altitude"))
    enemy_altitude = _metric_or_state(metrics, state, "enemy_altitude", ("enemy", "altitude"))
    ego_speed = _metric_or_state(metrics, state, "ego_speed", ("ego", "speed"))
    enemy_speed = _metric_or_state(metrics, state, "enemy_speed", ("enemy", "speed"))
    energy_difference = _float(metrics.get("energy_difference", 0.0))
    ego_energy = _float(metrics.get("ego_specific_energy", 0.0))
    enemy_energy = _float(metrics.get("enemy_specific_energy", 0.0))
    energy_ratio = energy_difference / max(abs(ego_energy) + abs(enemy_energy), 1.0)
    return {
        "distance": _float(metrics.get("distance", 0.0)),
        "ego_aim_angle": _float(metrics.get("ego_aim_angle", 0.0)),
        "enemy_aim_angle": _float(metrics.get("enemy_aim_angle", 0.0)),
        "ego_tail_angle": _float(metrics.get("ego_tail_angle", 0.0)),
        "enemy_tail_angle": _float(metrics.get("enemy_tail_angle", 0.0)),
        "distance_rate": _float(metrics.get("distance_rate", 0.0)),
        "ego_speed": ego_speed,
        "enemy_speed": enemy_speed,
        "ego_altitude": ego_altitude,
        "enemy_altitude": enemy_altitude,
        "energy_difference": energy_difference,
        "altitude_difference": ego_altitude - enemy_altitude,
        "speed_difference": ego_speed - enemy_speed,
        "energy_ratio": energy_ratio,
    }


def feature_vector_from_metrics(metrics: Dict, state: Optional[Dict] = None) -> List[float]:
    features = feature_dict_from_metrics(metrics, state)
    return [float(features[name]) for name in FEATURE_NAMES]


def load_dataset_feature_rows(dataset_dir: str, cache_path: Optional[str] = None) -> List[Dict]:
    dataset_dir = _resolve_path(dataset_dir)
    cache = VLMScoreCache(cache_path, enabled=True) if cache_path else None
    rows = []
    for sample_dir in sample_dirs(dataset_dir):
        metadata = _read_json(os.path.join(sample_dir, "metadata.json"))
        metrics = _read_json(os.path.join(sample_dir, "metrics.json"))
        state = _read_optional_json(os.path.join(sample_dir, "env_state_snapshot.json")) or {}
        physical_labels = _read_optional_json(os.path.join(sample_dir, "physical_labels.json")) or {}
        cache_entry = None
        if cache is not None:
            for key in (metadata.get("state_hash"), metadata.get("image_hash"), metadata.get("image_id")):
                if key:
                    cache_entry = cache.get(key)
                    if cache_entry is not None:
                        break
        rows.append({
            "sample_id": os.path.basename(sample_dir),
            "sample_dir": os.path.abspath(sample_dir),
            "metadata": metadata,
            "metrics": metrics,
            "state": state,
            "feature_dict": feature_dict_from_metrics(metrics, state),
            "feature_vector": feature_vector_from_metrics(metrics, state),
            "physical_scores": labels_to_scores(physical_labels),
            "cache_entry": cache_entry,
        })
    return rows


def labels_to_scores(labels: Dict) -> Dict[str, float]:
    scores = {}
    for name in LABEL_NAMES:
        value = labels.get(name, 0.0)
        if isinstance(value, dict):
            value = value.get("score", 0.0)
        scores[name] = float(np.clip(value, 0.0, 1.0))
    return scores


def summarize_label_distribution(rows: Iterable[Dict]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {name: 0.0 for name in LABEL_NAMES}
    return {
        name: float(np.mean([row.get("physical_scores", {}).get(name, 0.0) for row in rows]))
        for name in LABEL_NAMES
    }


def distribution_difference(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {name: float(a.get(name, 0.0) - b.get(name, 0.0)) for name in LABEL_NAMES}


class NearestSemanticCache:
    def __init__(self, rows: List[Dict], threshold: float = 2.5):
        self.rows = [row for row in rows if row.get("feature_vector")]
        self.threshold = float(threshold)
        if self.rows:
            matrix = np.asarray([row["feature_vector"] for row in self.rows], dtype=np.float64)
            self.mean = matrix.mean(axis=0)
            self.std = matrix.std(axis=0)
            self.std[self.std < 1e-6] = 1.0
            self.matrix = (matrix - self.mean) / self.std
        else:
            self.mean = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
            self.std = np.ones(len(FEATURE_NAMES), dtype=np.float64)
            self.matrix = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)

    def lookup(self, feature_vector: List[float]) -> Dict:
        if not self.rows:
            return {
                "accepted": False,
                "nearest_distance": None,
                "nearest_sample_id": None,
                "nearest_row": None,
            }
        vector = np.asarray(feature_vector, dtype=np.float64)
        normalized = (vector - self.mean) / self.std
        distances = np.linalg.norm(self.matrix - normalized[None, :], axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        row = self.rows[index]
        return {
            "accepted": bool(distance <= self.threshold),
            "nearest_distance": distance,
            "nearest_sample_id": row.get("sample_id"),
            "nearest_row": row,
        }


def sample_dirs(dataset_dir: str) -> List[str]:
    dataset_dir = _resolve_path(dataset_dir)
    if not os.path.isdir(dataset_dir):
        return []
    return [
        os.path.join(dataset_dir, name)
        for name in sorted(os.listdir(dataset_dir))
        if name.startswith("sample_") and os.path.isdir(os.path.join(dataset_dir, name))
    ]


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def _metric_or_state(metrics: Dict, state: Optional[Dict], metric_key: str, state_path: Tuple[str, str]) -> float:
    if metric_key in metrics:
        return _float(metrics.get(metric_key))
    if state:
        item = state
        for key in state_path:
            item = item.get(key, {}) if isinstance(item, dict) else {}
        if item != {}:
            return _float(item)
    return 0.0


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_optional_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    return _read_json(path)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root(), path))


def _float(value) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
