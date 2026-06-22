import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics
from lag_extensions.reward_shaping.config import LABEL_NAMES
from lag_extensions.reward_shaping.reward_wrapper import RewardShapingWrapper
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PHYSICAL_CONFIG = os.path.join(ROOT, "lag_extensions", "reward_shaping", "configs", "physical_1v1.yaml")
DEFAULT_CONFIG = os.path.join(ROOT, "lag_extensions", "reward_shaping", "configs", "default.yaml")


def fixed_actions(env):
    return np.array([[20, 20, 20, 15] for _ in range(env.num_agents)])


def test_disabled_wrapper_preserves_original_reward(tmp_path):
    env_raw = SingleCombatEnv("1v1/NoWeapon/Selfplay")
    env_wrapped = RewardShapingWrapper(
        SingleCombatEnv("1v1/NoWeapon/Selfplay"),
        config_path=DEFAULT_CONFIG,
        log_dir=str(tmp_path),
        run_name="disabled",
    )
    env_raw.seed(11)
    env_wrapped.seed(11)
    env_raw.reset()
    env_wrapped.reset()
    actions = fixed_actions(env_raw)
    _, raw_rewards, raw_dones, raw_info = env_raw.step(actions)
    _, wrapped_rewards, wrapped_dones, wrapped_info = env_wrapped.step(actions)
    assert np.allclose(raw_rewards, wrapped_rewards)
    assert np.all(raw_dones == wrapped_dones)
    assert "reward_shaping" not in wrapped_info
    env_raw.close()
    env_wrapped.close()


def test_enabled_wrapper_returns_reward_details(tmp_path):
    env = RewardShapingWrapper(
        SingleCombatEnv("1v1/NoWeapon/Selfplay"),
        config_path=PHYSICAL_CONFIG,
        log_dir=str(tmp_path),
        run_name="enabled",
    )
    env.seed(3)
    env.reset()
    _, rewards, dones, info = env.step(fixed_actions(env))
    assert rewards.shape == (env.num_agents, 1)
    assert dones.shape == (env.num_agents, 1)
    assert "reward_shaping" in info
    for idx, (agent_id, detail) in enumerate(info["reward_shaping"].items()):
        assert "reward_terms" in detail
        assert "verification" in detail
        assert np.isclose(detail["reward_terms"]["total_reward"], rewards[idx][0])
    env.close()


def test_tactical_label_scores_are_bounded():
    env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
    env.seed(5)
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
    assert sorted(labels.keys()) == sorted(LABEL_NAMES)
    for label in labels.values():
        assert 0.0 <= label["score"] <= 1.0
        assert isinstance(label["hard"], bool)
    env.close()
