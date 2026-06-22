import copy
import json
import os
from typing import Any, Dict, Optional

import yaml


LABEL_NAMES = [
    "ego_tail_advantage",
    "enemy_tail_threat",
    "effective_attack_window",
    "enemy_missile_threat_zone",
    "energy_advantage",
    "energy_disadvantage",
    "defensive_escape",
    "neutral_stalemate",
]


DEFAULT_CONFIG = {
    "reward_shaping": {
        "enabled": False,
        "semantic_source": "physical",
        "fallback_semantic_source": "physical",
        "cached_vlm_path": None,
        "use_label_fusion": False,
        "mock_seed": 7,
        "use_potential_based": True,
        "gamma": 0.99,
        "alpha": 0.5,
        "beta": 1.0,
        "verification_mode": "soft_gate",
        "consistency_threshold": 0.3,
        "tau": 0.2,
        "low_confidence_value": 0.1,
        "physical_reward_from": "weighted_scores",
        "log_steps": True,
        "log_episodes": True,
    },
    "thresholds": {
        "theta_tail_deg": 45.0,
        "theta_aim_deg": 60.0,
        "theta_launch_deg": 30.0,
        "theta_target_deg": 45.0,
        "theta_enemy_launch_deg": 30.0,
        # LAG JSBSim positions are in meters in a local NEU frame.
        "d_tail": 5000.0,
        "d_threat": 6000.0,
        "d_attack_min": 500.0,
        "d_attack_max": 8000.0,
        "d_missile_min": 500.0,
        "d_missile_max": 14000.0,
        "closing_rate_threshold": 0.0,
        "enemy_lock_min_score": 0.8,
        # Specific energy E = 0.5 * V^2 + g * h uses m^2/s^2.
        # These defaults are deliberately broad and should be tuned per paper experiment.
        "E_adv_threshold": 20000.0,
        "E_dis_threshold": 20000.0,
        "E_scale": 50000.0,
        "d_dot_escape_threshold": 5.0,
        "d_dot_scale": 120.0,
        "angle_escape_scale_deg": 30.0,
        "neutral_min": 3000.0,
        "neutral_max": 12000.0,
        "neutral_time_threshold": 20,
        "neutral_base_score": 0.2,
        "neutral_growth_steps": 100,
    },
    "label_weights": {
        "ego_tail_advantage": 0.5,
        "enemy_tail_threat": -0.8,
        "effective_attack_window": 1.0,
        "enemy_missile_threat_zone": -1.0,
        "energy_advantage": 0.3,
        "energy_disadvantage": -0.3,
        "defensive_escape": 0.5,
        "neutral_stalemate": -0.2,
    },
    "curriculum": {
        "enabled": True,
        "mode": "by_training_steps",
        "stages": [
            {
                "name": "advantage_or_basic",
                "start_step": 0,
                "beta": 1.0,
                "alpha": 0.5,
                "semantic_source": "physical",
                "semantic_call_interval": 5,
                "opponent_level": "weak",
                "situation_level": "easy",
            },
            {
                "name": "balanced",
                "start_step": 100000,
                "beta": 0.6,
                "alpha": 0.8,
                "semantic_source": "physical",
                "semantic_call_interval": 10,
                "opponent_level": "medium",
                "situation_level": "neutral",
            },
            {
                "name": "adversarial",
                "start_step": 300000,
                "beta": 0.3,
                "alpha": 0.5,
                "semantic_source": "physical",
                "semantic_call_interval": 20,
                "opponent_level": "historical_or_strong",
                "situation_level": "hard",
            },
            {
                "name": "final",
                "start_step": 600000,
                "beta": 0.0,
                "alpha": 0.2,
                "semantic_source": "physical",
                "semantic_call_interval": 50,
                "opponent_level": "strong",
                "situation_level": "random",
            },
        ],
    },
    "situation_curriculum": {
        "enabled": False,
        "mode": "fixed",
        "stage": "offensive_advantage",
        "battle_field_center": [120.0, 60.0, 0.0],
        "distance_m": [2800.0, 3600.0],
        "lateral_m": [-300.0, 300.0],
        "altitude_ft": 20000.0,
        "ego_speed_fps": 880.0,
        "enemy_speed_fps": 760.0,
        "stages": [
            {
                "name": "stage1_offensive_heavy",
                "start_step": 0,
                "probabilities": {
                    "offensive_advantage": 0.60,
                    "neutral_merge": 0.20,
                    "defensive_disadvantage": 0.10,
                    "random": 0.10,
                },
            },
            {
                "name": "stage2_balanced",
                "start_step": 50000,
                "probabilities": {
                    "offensive_advantage": 0.40,
                    "neutral_merge": 0.30,
                    "defensive_disadvantage": 0.15,
                    "random": 0.15,
                },
            },
            {
                "name": "stage3_harder",
                "start_step": 100000,
                "probabilities": {
                    "offensive_advantage": 0.25,
                    "neutral_merge": 0.35,
                    "defensive_disadvantage": 0.20,
                    "random": 0.20,
                },
            },
            {
                "name": "stage4_randomized",
                "start_step": 150000,
                "probabilities": {
                    "offensive_advantage": 0.15,
                    "neutral_merge": 0.35,
                    "defensive_disadvantage": 0.25,
                    "random": 0.25,
                },
            },
        ],
    },
    "vlm": {
        "enabled": False,
        "model_name_or_path": "models/Qwen2.5-VL-7B-Instruct",
        "model_family": "qwen2.5-vl",
        "load_mode": "explicit_single_gpu",
        "device": "cuda:0",
        "device_map": "auto",
        "dtype": "float16",
        "local_files_only": True,
        "attn_implementation": None,
        "use_flash_attention_2": False,
        "min_pixels": 50176,
        "max_pixels": 100352,
        "pixel_presets": {
            "tiny": {"min_pixels": 50176, "max_pixels": 100352},
            "fast": {"min_pixels": 100352, "max_pixels": 200704},
            "balanced": {"min_pixels": 200704, "max_pixels": 401408},
            "accurate": {"min_pixels": 200704, "max_pixels": 802816},
        },
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "prompt_version": "compact_v1",
        "online_call": False,
        "fallback_to_physical": True,
    },
    "renderer": {
        "image_size": 512,
        "ego_centered": True,
        "show_legend": True,
        "show_distance": True,
        "show_energy": True,
        "show_speed": True,
        "show_altitude": True,
        "show_delta_energy": True,
        "show_numeric_panel": True,
        "show_attack_sector": True,
        "show_threat_sector": True,
        "view_range_m": 15000,
    },
    "cache": {
        "enabled": True,
        "path": "scripts/results/vlm_cache/qwen2_5_vl_1v1_cache.jsonl",
    },
    "surrogate": {
        "enabled": False,
        "model_dir": "scripts/results/surrogate_models/stage8_formal100_smoke",
        "model_path": None,
        "input_dim": "auto",
        "output_dim": 8,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.0,
        "target_type": "fused_scores",
        "output_mode": "fused_scores_direct",
        "normalize_features": True,
        "device": "cpu",
    },
    "dataset": {
        "output_dir": "scripts/results/aircombat_semantic_dataset",
        "num_samples": 100,
        "sample_interval": 10,
    },
    "label_fusion": {
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
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_reward_shaping_config(
    path: Optional[str] = None,
    enabled_override: Optional[bool] = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        path = resolve_config_path(path)
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.load(f, Loader=yaml.FullLoader) or {}
        deep_update(config, loaded)
    if enabled_override is not None:
        config.setdefault("reward_shaping", {})["enabled"] = bool(enabled_override)
    return config


def default_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "configs", "default.yaml")


def resolve_config_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(project_root(), path)),
        os.path.abspath(os.path.join(project_root(), "scripts", path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Reward shaping config not found: {}".format(path))


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def snapshot_config(config: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
