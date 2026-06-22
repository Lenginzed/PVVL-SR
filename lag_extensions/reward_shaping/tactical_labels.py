import math
from typing import Dict, Optional, Tuple

import numpy as np

from .config import LABEL_NAMES


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def deg_to_rad(config: Dict, key: str) -> float:
    return float(np.deg2rad(config[key]))


def upper_range_score(distance: float, upper: float) -> float:
    return clip01(1.0 - distance / max(float(upper), 1e-8))


def window_range_score(distance: float, lower: float, upper: float) -> float:
    lower = float(lower)
    upper = float(upper)
    if upper <= lower:
        return 0.0
    if distance < lower or distance > upper:
        return 0.0
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return clip01(1.0 - abs(distance - center) / max(half_width, 1e-8))


class TacticalLabeler:
    """Physical scoring rules for the eight tactical semantic labels."""

    def __init__(self, thresholds: Dict):
        self.thresholds = thresholds

    def evaluate(
        self,
        metrics: Dict,
        previous_metrics: Optional[Dict] = None,
        previous_label_scores: Optional[Dict[str, float]] = None,
        neutral_steps: int = 0,
    ) -> Tuple[Dict[str, Dict[str, float]], int]:
        labels = {}
        labels["ego_tail_advantage"] = self._ego_tail_advantage(metrics)
        labels["enemy_tail_threat"] = self._enemy_tail_threat(metrics)
        labels["effective_attack_window"] = self._effective_attack_window(metrics)
        labels["enemy_missile_threat_zone"] = self._enemy_missile_threat_zone(metrics)
        labels["energy_advantage"] = self._energy_advantage(metrics)
        labels["energy_disadvantage"] = self._energy_disadvantage(metrics)
        labels["defensive_escape"] = self._defensive_escape(metrics, previous_metrics, previous_label_scores, labels)

        is_neutral = self._neutral_candidate(metrics, labels)
        current_neutral_steps = neutral_steps + 1 if is_neutral else 0
        labels["neutral_stalemate"] = self._neutral_stalemate(is_neutral, current_neutral_steps)

        for name in LABEL_NAMES:
            labels[name]["score"] = clip01(labels[name]["score"])
            labels[name]["hard"] = bool(labels[name]["hard"])
        return labels, current_neutral_steps

    def _ego_tail_advantage(self, m: Dict) -> Dict[str, float]:
        theta_tail = deg_to_rad(self.thresholds, "theta_tail_deg")
        theta_aim = deg_to_rad(self.thresholds, "theta_aim_deg")
        d_tail = float(self.thresholds["d_tail"])
        tail_score = clip01(1.0 - m["ego_tail_angle"] / theta_tail)
        aim_score = clip01(1.0 - m["ego_aim_angle"] / theta_aim)
        dist_score = upper_range_score(m["distance"], d_tail)
        hard = m["ego_tail_angle"] < theta_tail and m["ego_aim_angle"] < theta_aim and m["distance"] < d_tail
        return {"hard": hard, "score": tail_score * aim_score * dist_score}

    def _enemy_tail_threat(self, m: Dict) -> Dict[str, float]:
        theta_tail = deg_to_rad(self.thresholds, "theta_tail_deg")
        theta_aim = deg_to_rad(self.thresholds, "theta_aim_deg")
        d_threat = float(self.thresholds["d_threat"])
        tail_score = clip01(1.0 - m["enemy_tail_angle"] / theta_tail)
        aim_score = clip01(1.0 - m["enemy_aim_angle"] / theta_aim)
        dist_score = upper_range_score(m["distance"], d_threat)
        hard = m["enemy_tail_angle"] < theta_tail and m["enemy_aim_angle"] < theta_aim and m["distance"] < d_threat
        return {"hard": hard, "score": tail_score * aim_score * dist_score}

    def _effective_attack_window(self, m: Dict) -> Dict[str, float]:
        theta_launch = deg_to_rad(self.thresholds, "theta_launch_deg")
        theta_target = deg_to_rad(self.thresholds, "theta_target_deg")
        d_min = float(self.thresholds["d_attack_min"])
        d_max = float(self.thresholds["d_attack_max"])
        closing_threshold = float(self.thresholds.get("closing_rate_threshold", 0.0))
        closing_rate = -float(m["distance_rate"])
        range_score = window_range_score(m["distance"], d_min, d_max)
        aim_score = clip01(1.0 - m["ego_aim_angle"] / theta_launch)
        aspect_score = clip01(1.0 - m["ego_tail_angle"] / theta_target)
        hard = (
            d_min < m["distance"] < d_max
            and m["ego_aim_angle"] < theta_launch
            and m["ego_tail_angle"] < theta_target
            and closing_rate > closing_threshold
        )
        return {"hard": hard, "score": range_score * aim_score * aspect_score}

    def _enemy_missile_threat_zone(self, m: Dict) -> Dict[str, float]:
        theta_enemy_launch = deg_to_rad(self.thresholds, "theta_enemy_launch_deg")
        d_min = float(self.thresholds["d_missile_min"])
        d_max = float(self.thresholds["d_missile_max"])
        range_score = window_range_score(m["distance"], d_min, d_max)
        aim_score = clip01(1.0 - m["enemy_aim_angle"] / theta_enemy_launch)
        score = range_score * aim_score
        hard = d_min < m["distance"] < d_max and m["enemy_aim_angle"] < theta_enemy_launch
        if m.get("enemy_lock", False) or m.get("ego_missile_warning", False):
            hard = True
            score = max(score, float(self.thresholds.get("enemy_lock_min_score", 0.8)))
        return {"hard": hard, "score": score}

    def _energy_advantage(self, m: Dict) -> Dict[str, float]:
        delta_e = float(m["energy_difference"])
        threshold = float(self.thresholds["E_adv_threshold"])
        scale = max(float(self.thresholds["E_scale"]), 1e-8)
        score = 0.0 if delta_e <= 0 else clip01(0.5 + 0.5 * math.tanh(delta_e / scale))
        return {"hard": delta_e > threshold, "score": score}

    def _energy_disadvantage(self, m: Dict) -> Dict[str, float]:
        delta_e = float(m["energy_difference"])
        threshold = float(self.thresholds["E_dis_threshold"])
        scale = max(float(self.thresholds["E_scale"]), 1e-8)
        score = 0.0 if delta_e >= 0 else clip01(0.5 + 0.5 * math.tanh((-delta_e) / scale))
        return {"hard": delta_e < -threshold, "score": score}

    def _defensive_escape(
        self,
        m: Dict,
        previous_metrics: Optional[Dict],
        previous_label_scores: Optional[Dict[str, float]],
        current_labels: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        threat_hard = current_labels["enemy_tail_threat"]["hard"] or current_labels["enemy_missile_threat_zone"]["hard"]
        threat_score = max(
            current_labels["enemy_tail_threat"]["score"],
            current_labels["enemy_missile_threat_zone"]["score"],
        )
        separation_score = clip01(float(m["distance_rate"]) / max(float(self.thresholds["d_dot_scale"]), 1e-8))
        angle_escape_score = 0.0
        missile_score_drop = 0.0

        if previous_metrics is not None:
            angle_delta = float(m["enemy_aim_angle"] - previous_metrics.get("enemy_aim_angle", m["enemy_aim_angle"]))
            angle_scale = max(np.deg2rad(float(self.thresholds["angle_escape_scale_deg"])), 1e-8)
            angle_escape_score = clip01(angle_delta / angle_scale)

        if previous_label_scores is not None:
            prev_missile = float(previous_label_scores.get("enemy_missile_threat_zone", 0.0))
            missile_score_drop = clip01(prev_missile - current_labels["enemy_missile_threat_zone"]["score"])

        hard = (
            threat_hard
            and float(m["distance_rate"]) > float(self.thresholds["d_dot_escape_threshold"])
            and (angle_escape_score > 0.0 or missile_score_drop > 0.0)
        )
        score = threat_score * max(separation_score, angle_escape_score, missile_score_drop)
        return {"hard": hard, "score": score}

    def _neutral_candidate(self, m: Dict, labels: Dict[str, Dict[str, float]]) -> bool:
        blocking_labels = [
            "ego_tail_advantage",
            "enemy_tail_threat",
            "effective_attack_window",
            "enemy_missile_threat_zone",
            "defensive_escape",
        ]
        no_active_state = all(not labels[name]["hard"] for name in blocking_labels)
        distance_ok = float(self.thresholds["neutral_min"]) <= m["distance"] <= float(self.thresholds["neutral_max"])
        return bool(no_active_state and distance_ok)

    def _neutral_stalemate(self, is_neutral: bool, neutral_steps: int) -> Dict[str, float]:
        if not is_neutral:
            return {"hard": False, "score": 0.0}
        threshold = int(self.thresholds["neutral_time_threshold"])
        base = float(self.thresholds.get("neutral_base_score", 0.2))
        growth_steps = max(int(self.thresholds.get("neutral_growth_steps", 100)), 1)
        score = clip01(base + neutral_steps / float(growth_steps))
        return {"hard": neutral_steps >= threshold, "score": score}
