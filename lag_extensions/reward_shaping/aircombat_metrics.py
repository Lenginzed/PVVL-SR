import math
from typing import Dict

import numpy as np


EPS = 1e-8
G = 9.81


def safe_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def unit_vector(vec: np.ndarray, fallback: np.ndarray = None) -> np.ndarray:
    norm = safe_norm(vec)
    if norm > EPS:
        return vec / norm
    if fallback is not None:
        fallback_norm = safe_norm(fallback)
        if fallback_norm > EPS:
            return fallback / fallback_norm
    return np.zeros_like(vec, dtype=np.float64)


def clipped_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua = unit_vector(a)
    ub = unit_vector(b)
    if safe_norm(ua) <= EPS or safe_norm(ub) <= EPS:
        return math.pi / 2.0
    return float(math.acos(clipped_dot(ua, ub)))


def specific_energy(speed: float, altitude: float) -> float:
    return 0.5 * float(speed) ** 2 + G * float(altitude)


def compute_aircombat_metrics(state: Dict) -> Dict[str, float]:
    ego = state["ego"]
    enemy = state["enemy"]
    p_e = np.asarray(ego["position"], dtype=np.float64)
    p_b = np.asarray(enemy["position"], dtype=np.float64)
    v_e = np.asarray(ego["velocity"], dtype=np.float64)
    v_b = np.asarray(enemy["velocity"], dtype=np.float64)
    h_e = unit_vector(np.asarray(ego["heading_vector"], dtype=np.float64), fallback=v_e)
    h_b = unit_vector(np.asarray(enemy["heading_vector"], dtype=np.float64), fallback=v_b)

    r = p_b - p_e
    distance = max(safe_norm(r), EPS)
    u = r / distance
    relative_velocity = v_b - v_e

    ego_speed = safe_norm(v_e)
    enemy_speed = safe_norm(v_b)
    ego_energy = specific_energy(ego_speed, float(ego["altitude"]))
    enemy_energy = specific_energy(enemy_speed, float(enemy["altitude"]))

    metrics = {
        "relative_position": r,
        "distance": distance,
        "line_of_sight_unit_vector": u,
        "relative_velocity": relative_velocity,
        "distance_rate": float(np.dot(relative_velocity, u)),
        "ego_aim_angle": angle_between(h_e, u),
        "enemy_aim_angle": angle_between(h_b, -u),
        "ego_tail_angle": angle_between(h_b, u),
        "enemy_tail_angle": angle_between(-h_e, u),
        "ego_speed": ego_speed,
        "enemy_speed": enemy_speed,
        "ego_specific_energy": ego_energy,
        "enemy_specific_energy": enemy_energy,
        "energy_difference": ego_energy - enemy_energy,
        "ego_missile_warning": bool(ego.get("missile_warning", False)),
        "enemy_missile_warning": bool(enemy.get("missile_warning", False)),
        "ego_lock": bool(ego.get("lock", False)),
        "enemy_lock": bool(enemy.get("lock", False)),
    }
    return metrics


def metrics_for_logging(metrics: Dict[str, float]) -> Dict[str, float]:
    output = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            continue
        if key.endswith("_angle"):
            output[key] = float(value)
            output[key + "_deg"] = float(np.rad2deg(value))
        elif isinstance(value, (bool, np.bool_)):
            output[key] = bool(value)
        else:
            output[key] = float(value)
    return output
