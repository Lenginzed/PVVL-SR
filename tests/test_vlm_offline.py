import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics
from lag_extensions.reward_shaping.config import LABEL_NAMES
from lag_extensions.reward_shaping.qwen_vl_scorer import QwenVLMScorer
from lag_extensions.reward_shaping.situation_renderer import SituationRenderer
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler
from lag_extensions.reward_shaping.vlm_cache import VLMScoreCache
from lag_extensions.reward_shaping.vlm_prompt import build_qwen_vl_prompt


def test_prompt_contains_all_labels():
    prompt = build_qwen_vl_prompt()
    for label in LABEL_NAMES:
        assert label in prompt
    assert "Output JSON only" in prompt


def test_qwen_json_parser_variants():
    good = {name: 0.2 for name in LABEL_NAMES}
    assert QwenVLMScorer.parse_scores(json.dumps(good))["valid"]
    noisy = "prefix\n{}\nsuffix".format(json.dumps(good))
    assert QwenVLMScorer.parse_scores(noisy)["valid"]
    missing = dict(good)
    missing.pop(LABEL_NAMES[0])
    assert not QwenVLMScorer.parse_scores(json.dumps(missing))["valid"]
    assert not QwenVLMScorer.parse_scores("not json")["valid"]
    clipped = dict(good)
    clipped[LABEL_NAMES[0]] = 2.0
    parsed = QwenVLMScorer.parse_scores(json.dumps(clipped))
    assert parsed["scores"][LABEL_NAMES[0]] == 1.0
    assert parsed["warnings"]


def test_renderer_and_cache(tmp_path):
    env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
    env.seed(13)
    env.reset()
    extractor = StateExtractor()
    state = extractor.extract(env, env.ego_ids[0])
    metrics = compute_aircombat_metrics(state)
    thresholds = {
        "theta_tail_deg": 45,
        "theta_aim_deg": 60,
        "theta_launch_deg": 30,
        "theta_target_deg": 45,
        "theta_enemy_launch_deg": 30,
        "d_tail": 5000,
        "d_threat": 6000,
        "d_attack_min": 500,
        "d_attack_max": 8000,
        "d_missile_min": 500,
        "d_missile_max": 14000,
        "E_adv_threshold": 20000,
        "E_dis_threshold": 20000,
        "E_scale": 50000,
        "d_dot_escape_threshold": 5,
        "d_dot_scale": 120,
        "angle_escape_scale_deg": 30,
        "neutral_min": 3000,
        "neutral_max": 12000,
        "neutral_time_threshold": 20,
    }
    labels, _ = TacticalLabeler(thresholds).evaluate(metrics)
    metadata = SituationRenderer({"image_size": 512}, thresholds).render_sample(
        state, metrics, labels, str(tmp_path), sample_id="sample", image_filename="image.png", metadata_filename="metadata.json"
    )
    assert os.path.exists(metadata["image_path"])
    assert os.path.exists(metadata["metadata_path"])

    cache = VLMScoreCache(str(tmp_path / "cache.jsonl"))
    scores = {name: labels[name]["score"] for name in LABEL_NAMES}
    entry = {"valid": True, "scores": scores, "image_id": metadata["image_id"], "metadata": metadata}
    cache.set("test-key", entry)
    assert cache.get("test-key") is not None
    assert cache.get("test-key") is not None
    assert cache.stats()["hits"] == 2
    env.close()
