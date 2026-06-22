import json
import math
import os
import hashlib
from typing import Dict, Optional

import numpy as np

from .config import LABEL_NAMES
from .config import project_root
from .semantic_cache_retrieval import feature_vector_from_metrics
from .semantic_surrogate import SemanticSurrogatePredictor


ENERGY_LABELS = {"energy_advantage", "energy_disadvantage"}


DEFAULT_LABEL_FUSION_CONFIG = {
    "override_threshold": 0.6,
    "low_physical_threshold": 0.1,
    "high_physical_threshold": 0.8,
    "low_vlm_threshold": 0.2,
    "high_vlm_threshold": 0.8,
    "labels": {
        "ego_tail_advantage": {"policy": "hybrid", "lambda_vlm": 0.7, "physical_override": True},
        "enemy_tail_threat": {"policy": "hybrid", "lambda_vlm": 0.7, "physical_override": True},
        "effective_attack_window": {"policy": "hybrid", "lambda_vlm": 0.6, "physical_override": True},
        "enemy_missile_threat_zone": {"policy": "hybrid", "lambda_vlm": 0.4, "physical_override": True},
        "energy_advantage": {"policy": "physical", "lambda_vlm": 0.0, "physical_override": True},
        "energy_disadvantage": {"policy": "physical", "lambda_vlm": 0.0, "physical_override": True},
        "defensive_escape": {"policy": "hybrid", "lambda_vlm": 0.5, "physical_override": True},
        "neutral_stalemate": {"policy": "hybrid", "lambda_vlm": 0.5, "physical_override": True},
    },
}


def labels_to_scores(physical_labels: Dict) -> Dict[str, float]:
    return {name: float(physical_labels[name]["score"]) for name in LABEL_NAMES}


def state_cache_key(state: Dict, decimals: int = 1) -> str:
    payload = {
        "agent_id": state.get("agent_id"),
        "enemy_id": state.get("enemy_id"),
        "ego": _cache_aircraft_state(state["ego"], decimals),
        "enemy": _cache_aircraft_state(state["enemy"], decimals),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "state:" + hashlib.sha256(encoded).hexdigest()


def _cache_aircraft_state(state: Dict, decimals: int) -> Dict:
    return {
        "position": _round_list(state["position"], decimals),
        "velocity": _round_list(state["velocity"], decimals),
        "heading_vector": _round_list(state["heading_vector"], decimals + 2),
        "altitude": round(float(state["altitude"]), decimals),
        "missile_warning": bool(state.get("missile_warning", False)),
        "lock": bool(state.get("lock", False)),
    }


def _round_list(values, decimals: int):
    return [round(float(x), decimals) for x in np.asarray(values).reshape(-1)]


class SemanticScorer:
    """Semantic scorer interface with physical, mock, and cached VLM sources."""

    def __init__(
        self,
        source: str = "physical",
        seed: int = 7,
        cached_vlm_path: Optional[str] = None,
        surrogate_config: Optional[Dict] = None,
    ):
        self.source = source
        self.rng = np.random.RandomState(seed)
        self.cached_vlm_path = cached_vlm_path
        self.cache = {}
        self.surrogate_config = surrogate_config or {}
        self.surrogate = None
        self.last_info = {}

    def set_source(self, source: str) -> None:
        self.source = source

    def score(
        self,
        state: Dict,
        situation_image=None,
        physical_labels: Optional[Dict] = None,
        cache_key: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ) -> Dict[str, float]:
        if self.source == "physical":
            self.last_info = {"source": "physical", "cache_hit": False, "vlm_valid": None, "fallback_reason": None}
            return labels_to_scores(physical_labels)
        if self.source == "mock":
            self.last_info = {"source": "mock", "cache_hit": False, "vlm_valid": True, "fallback_reason": None}
            return {name: float(self.rng.rand()) for name in LABEL_NAMES}
        if self.source == "cached_vlm":
            self._ensure_cache_loaded()
            cached_entry = self._get_cached_entry(cache_key)
            if cached_entry is not None:
                self.last_info = {
                    "source": "cached_vlm",
                    "cache_hit": True,
                    "vlm_valid": bool(cached_entry.get("valid", True)),
                    "fallback_reason": cached_entry.get("fallback_reason"),
                    "cache_key": cache_key,
                    "image_id": cached_entry.get("image_id"),
                    "model": cached_entry.get("model"),
                }
                return self._sanitize_scores(cached_entry.get("scores", {}))
            cached = self._get_cached_scores(cache_key)
            if cached is not None:
                self.last_info = {"source": "cached_vlm", "cache_hit": True, "vlm_valid": True, "fallback_reason": None, "cache_key": cache_key}
                return cached
            self.last_info = {
                "source": "cached_vlm_fallback_physical",
                "cache_hit": False,
                "vlm_valid": False,
                "fallback_reason": "cache_miss",
                "cache_key": cache_key,
            }
            return labels_to_scores(physical_labels)
        if self.source == "surrogate":
            try:
                predictor = self._surrogate_predictor()
                features = feature_vector_from_metrics(metrics or {}, state)
                scores = predictor.predict_features(features)
                self.last_info = dict(predictor.last_info)
                self.last_info.update({
                    "source": "surrogate",
                    "cache_hit": None,
                    "vlm_valid": None,
                    "fallback_reason": None,
                    "surrogate_output_mode": self.surrogate_config.get("output_mode", "fused_scores_direct"),
                })
                return scores
            except Exception as exc:
                self.last_info = {
                    "source": "surrogate_fallback_physical",
                    "cache_hit": None,
                    "vlm_valid": None,
                    "fallback_reason": "surrogate_error: {}".format(exc),
                    "surrogate_output_mode": self.surrogate_config.get("output_mode", "fused_scores_direct"),
                }
                return labels_to_scores(physical_labels)
        raise ValueError("Unknown semantic source: {}".format(self.source))

    def _ensure_cache_loaded(self) -> None:
        if not self.cache and self.cached_vlm_path:
            self.cache = self._load_cache(self.cached_vlm_path)

    def _surrogate_predictor(self):
        if self.surrogate is None:
            model_dir = self.surrogate_config.get("model_dir") or self.surrogate_config.get("model_path")
            if not model_dir:
                raise ValueError("surrogate.model_dir is required when semantic_source=surrogate")
            device = self.surrogate_config.get("device", "cpu")
            self.surrogate = SemanticSurrogatePredictor(model_dir=model_dir, device=device)
        return self.surrogate

    def _load_cache(self, path: Optional[str]) -> Dict[str, Dict]:
        if not path or not os.path.exists(path):
            if path:
                candidate = os.path.abspath(os.path.join(project_root(), path))
                if os.path.exists(candidate):
                    path = candidate
                else:
                    return {}
            else:
                return {}
        cache = {}
        with open(path, "r", encoding="utf-8-sig") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    self._add_cache_aliases(cache, item)
            else:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    for key, scores in loaded.items():
                        item = scores if isinstance(scores, dict) else {}
                        item.setdefault("cache_key", str(key))
                        self._add_cache_aliases(cache, item)
        return cache

    def _add_cache_aliases(self, cache: Dict[str, Dict], item: Dict) -> None:
        if not isinstance(item, dict):
            return
        scores = item.get("scores", item)
        entry = dict(item)
        entry["scores"] = self._sanitize_scores(scores)
        aliases = [
            item.get("key"),
            item.get("cache_key"),
            item.get("image_id"),
            item.get("state_hash"),
        ]
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        aliases.extend([
            metadata.get("cache_key"),
            metadata.get("image_id"),
            metadata.get("image_hash"),
            metadata.get("state_hash"),
        ])
        for alias in aliases:
            if alias:
                cache[str(alias)] = entry

    def _get_cached_scores(self, cache_key: Optional[str]) -> Optional[Dict[str, float]]:
        entry = self._get_cached_entry(cache_key)
        if entry is not None:
            return self._sanitize_scores(entry.get("scores", {}))
        return None

    def _get_cached_entry(self, cache_key: Optional[str]) -> Optional[Dict]:
        if cache_key is None:
            return None
        return self.cache.get(str(cache_key))

    def _sanitize_scores(self, scores: Dict) -> Dict[str, float]:
        return {name: float(np.clip(scores.get(name, 0.0), 0.0, 1.0)) for name in LABEL_NAMES}


class PhysicalVerifier:
    """Verify semantic scores against physical scores."""

    def __init__(
        self,
        mode: str = "soft_gate",
        consistency_threshold: float = 0.3,
        tau: float = 0.2,
        low_confidence_value: float = 0.1,
    ):
        self.mode = mode
        self.consistency_threshold = consistency_threshold
        self.tau = max(float(tau), 1e-8)
        self.low_confidence_value = low_confidence_value

    def verify(self, physical_scores: Dict[str, float], semantic_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        confidence_scores = {}
        verified_scores = {}
        for name in LABEL_NAMES:
            p_i = float(np.clip(physical_scores.get(name, 0.0), 0.0, 1.0))
            q_i = float(np.clip(semantic_scores.get(name, 0.0), 0.0, 1.0))
            diff = abs(q_i - p_i)
            if self.mode == "hard_gate":
                confidence = 1.0 if diff < self.consistency_threshold else self.low_confidence_value
            elif self.mode == "soft_gate":
                confidence = math.exp(-diff / self.tau)
            else:
                raise ValueError("Unknown verification mode: {}".format(self.mode))
            confidence_scores[name] = float(np.clip(confidence, 0.0, 1.0))
            verified_scores[name] = confidence_scores[name] * q_i
        return {
            "semantic_scores": {name: float(np.clip(semantic_scores.get(name, 0.0), 0.0, 1.0)) for name in LABEL_NAMES},
            "physical_scores": {name: float(np.clip(physical_scores.get(name, 0.0), 0.0, 1.0)) for name in LABEL_NAMES},
            "confidence_scores": confidence_scores,
            "verified_scores": verified_scores,
        }


class LabelWiseSemanticFusion:
    """Fuse raw VLM, verified VLM, and physical scores per tactical label."""

    def __init__(self, config: Optional[Dict] = None):
        merged = _deepcopy_dict(DEFAULT_LABEL_FUSION_CONFIG)
        if config:
            _deep_update(merged, config)
        self.config = merged
        self.labels_config = merged.get("labels", {})
        self.override_threshold = float(merged.get("override_threshold", 0.6))
        self.low_physical_threshold = float(merged.get("low_physical_threshold", 0.1))
        self.high_physical_threshold = float(merged.get("high_physical_threshold", 0.8))
        self.low_vlm_threshold = float(merged.get("low_vlm_threshold", 0.2))
        self.high_vlm_threshold = float(merged.get("high_vlm_threshold", 0.8))

    def fuse(self, verification: Dict[str, Dict[str, float]]) -> Dict:
        semantic_scores = verification.get("semantic_scores", {})
        physical_scores = verification.get("physical_scores", {})
        confidence_scores = verification.get("confidence_scores", {})
        verified_scores = verification.get("verified_scores", {})
        fused_scores = {}
        fusion_policy = {}
        override_info = {}

        for name in LABEL_NAMES:
            label_config = self._label_config(name)
            policy = label_config.get("policy", "verified_vlm")
            lambda_vlm = float(np.clip(label_config.get("lambda_vlm", 1.0), 0.0, 1.0))
            physical_override = bool(label_config.get("physical_override", False))
            p_i = float(np.clip(physical_scores.get(name, 0.0), 0.0, 1.0))
            q_i = float(np.clip(semantic_scores.get(name, 0.0), 0.0, 1.0))
            v_i = float(np.clip(verified_scores.get(name, 0.0), 0.0, 1.0))
            c_i = float(np.clip(confidence_scores.get(name, 0.0), 0.0, 1.0))
            fused, base_reason = self._base_score(policy, lambda_vlm, p_i, q_i, v_i)

            override_triggered = False
            override_reason = None
            if policy == "physical":
                override_triggered = bool(name in ENERGY_LABELS or physical_override)
                override_reason = "policy_physical"
                fused = p_i
            elif physical_override:
                fused, override_triggered, override_reason = self._apply_override(name, fused, p_i, q_i)

            fused = float(np.clip(fused, 0.0, 1.0))
            fused_scores[name] = fused
            fusion_policy[name] = {
                "policy": policy,
                "lambda_vlm": lambda_vlm,
                "physical_override": physical_override,
                "base_reason": base_reason,
            }
            override_info[name] = {
                "override_triggered": bool(override_triggered),
                "override_reason": override_reason,
                "original_vlm_score": q_i,
                "physical_score": p_i,
                "verified_vlm_score": v_i,
                "confidence": c_i,
                "fused_score": fused,
            }

        output = dict(verification)
        output.update({
            "fused_scores": fused_scores,
            "fusion_policy": fusion_policy,
            "override_info": override_info,
        })
        return output

    def _label_config(self, name: str) -> Dict:
        return dict(self.labels_config.get(name, {"policy": "verified_vlm", "lambda_vlm": 1.0, "physical_override": False}))

    @staticmethod
    def _base_score(policy: str, lambda_vlm: float, physical: float, semantic: float, verified: float):
        if policy == "vlm":
            return semantic, "raw_vlm"
        if policy == "physical":
            return physical, "physical"
        if policy == "verified_vlm":
            return verified, "verified_vlm"
        if policy == "hybrid":
            return lambda_vlm * verified + (1.0 - lambda_vlm) * physical, "hybrid_verified_physical"
        if policy == "physical_override":
            return semantic, "raw_vlm_with_physical_override"
        raise ValueError("Unknown label fusion policy: {}".format(policy))

    def _apply_override(self, name: str, fused: float, physical: float, semantic: float):
        diff = abs(semantic - physical)
        if physical < self.low_physical_threshold and semantic > self.high_vlm_threshold:
            return physical, True, "physical_low_vlm_high"
        if physical > self.high_physical_threshold and semantic < self.low_vlm_threshold:
            return physical, True, "physical_high_vlm_low"
        if diff > self.override_threshold:
            return physical, True, "score_gap_gt_threshold"
        if name == "enemy_missile_threat_zone" and physical > self.high_physical_threshold and fused < physical:
            return physical, True, "missile_threat_physical_veto"
        return fused, False, None


class SemanticReward:
    def __init__(self, label_weights: Dict[str, float]):
        self.label_weights = {name: float(label_weights.get(name, 0.0)) for name in LABEL_NAMES}

    def potential(self, scores: Dict[str, float]) -> float:
        return float(sum(self.label_weights[name] * float(scores.get(name, 0.0)) for name in LABEL_NAMES))

    def weighted_physical_reward(self, physical_scores: Dict[str, float]) -> float:
        return self.potential(physical_scores)

    def shaping_reward(self, previous_phi: float, current_phi: float, gamma: float, potential_based: bool = True) -> float:
        if potential_based:
            return float(gamma * current_phi - previous_phi)
        return float(current_phi)


def _deepcopy_dict(value: Dict) -> Dict:
    return json.loads(json.dumps(value))


def _deep_update(base: Dict, override: Dict) -> Dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
