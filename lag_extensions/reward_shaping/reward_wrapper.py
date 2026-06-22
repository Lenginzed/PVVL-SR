import logging
import os
from typing import Dict, Optional

import numpy as np

from .aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from .config import LABEL_NAMES, default_config_path, load_reward_shaping_config
from .curriculum_scheduler import CurriculumScheduler
from .logger import RewardShapingLogger
from .semantic_reward import LabelWiseSemanticFusion, PhysicalVerifier, SemanticReward, SemanticScorer, labels_to_scores, state_cache_key
from .state_extractor import StateExtractor
from .tactical_labels import TacticalLabeler


class RewardShapingWrapper:
    """Optional 1v1 reward shaping adapter.

    When disabled, step/reset are delegated without changing rewards or info.
    When enabled, env.step() returns total shaped rewards and stores the
    detailed reward decomposition in info["reward_shaping"].
    """

    def __init__(
        self,
        env,
        config_path: Optional[str] = None,
        enabled_override: Optional[bool] = None,
        log_dir: Optional[str] = None,
        run_name: str = "env0",
    ):
        self.env = env
        self.config_path = config_path or default_config_path()
        self.config = load_reward_shaping_config(self.config_path, enabled_override=enabled_override)
        self.reward_config = self.config.get("reward_shaping", {})
        self.enabled = bool(self.reward_config.get("enabled", False))
        self.run_name = run_name

        self.extractor = StateExtractor()
        self.labeler = TacticalLabeler(self.config.get("thresholds", {}))
        self.scheduler = CurriculumScheduler(self.config.get("curriculum", {}), self.reward_config)
        self.scorer = SemanticScorer(
            source=self.reward_config.get("semantic_source", "physical"),
            seed=int(self.reward_config.get("mock_seed", 7)),
            cached_vlm_path=self.reward_config.get("cached_vlm_path") or self.config.get("cache", {}).get("path"),
            surrogate_config=self.config.get("surrogate", {}),
        )
        self.verifier = PhysicalVerifier(
            mode=self.reward_config.get("verification_mode", "soft_gate"),
            consistency_threshold=float(self.reward_config.get("consistency_threshold", 0.3)),
            tau=float(self.reward_config.get("tau", 0.2)),
            low_confidence_value=float(self.reward_config.get("low_confidence_value", 0.1)),
        )
        self.reward_model = SemanticReward(self.config.get("label_weights", {}))
        self.fusion_model = LabelWiseSemanticFusion(self.config.get("label_fusion", {}))

        self.log_dir = log_dir or os.path.abspath(os.path.join("reward_shaping_logs"))
        self.logger = RewardShapingLogger(self.log_dir, self.config, run_name) if self.enabled else None
        self.global_step = 0
        self.prev_phi = {}
        self.prev_metrics = {}
        self.prev_label_scores = {}
        self.last_semantic_scores = {}
        self.neutral_steps = {}

        if self.enabled:
            logging.info(
                "Reward shaping enabled for %s with config %s (logs: %s)",
                run_name,
                self.config_path,
                self.log_dir,
            )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        self._reset_internal_state()
        return obs

    def step(self, action):
        if not self.enabled:
            return self.env.step(action)

        obs, rewards, dones, info = self.env.step(action)
        shaped_rewards = np.array(rewards, dtype=np.float32, copy=True)
        info = dict(info)
        shaping_info = {}

        stage = self.scheduler.update(self.global_step)
        for reward_index, agent_id in enumerate(self._rl_agent_ids()):
            detail = self._compute_reward_detail(agent_id, float(rewards[reward_index][0]), stage)
            shaped_rewards[reward_index][0] = detail["reward_terms"]["total_reward"]
            shaping_info[agent_id] = detail
            if self.logger is not None and self.reward_config.get("log_steps", True):
                self.logger.log_step(agent_id, detail)

        info["reward_shaping"] = shaping_info
        info["reward_shaping_summary"] = self._step_summary(shaping_info)

        self.global_step += 1
        if np.all(dones):
            if self.logger is not None and self.reward_config.get("log_episodes", True):
                for agent_id in self._rl_agent_ids():
                    self.logger.finish_episode(self.env, agent_id)

        return obs, shaped_rewards, dones, info

    def close(self):
        return self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def _reset_internal_state(self) -> None:
        self.prev_phi.clear()
        self.prev_metrics.clear()
        self.prev_label_scores.clear()
        self.last_semantic_scores.clear()
        self.neutral_steps = {agent_id: 0 for agent_id in self._rl_agent_ids()}
        stage = self.scheduler.update(self.global_step)
        for agent_id in self._rl_agent_ids():
            context = self._compute_context(agent_id, stage, initialize=True)
            self.prev_phi[agent_id] = context["semantic_phi"]
            self.prev_metrics[agent_id] = context["raw_metrics"]
            self.prev_label_scores[agent_id] = context["physical_scores"]
            self.last_semantic_scores[agent_id] = context["verification"]["semantic_scores"]
        if self.logger is not None:
            self.logger.reset_episode(self._rl_agent_ids())

    def _compute_reward_detail(self, agent_id: str, env_reward: float, stage: Dict) -> Dict:
        context = self._compute_context(agent_id, stage, initialize=False)
        previous_phi = float(self.prev_phi.get(agent_id, context["semantic_phi"]))
        gamma = float(self.reward_config.get("gamma", 0.99))
        alpha = float(stage.get("alpha", self.reward_config.get("alpha", 0.0)))
        beta = float(stage.get("beta", self.reward_config.get("beta", 0.0)))
        potential_based = bool(self.reward_config.get("use_potential_based", True))

        shaping_reward = self.reward_model.shaping_reward(
            previous_phi=previous_phi,
            current_phi=context["semantic_phi"],
            gamma=gamma,
            potential_based=potential_based,
        )
        total_reward = env_reward + alpha * context["physical_reward"] + beta * shaping_reward

        self.prev_phi[agent_id] = context["semantic_phi"]
        self.prev_metrics[agent_id] = context["raw_metrics"]
        self.prev_label_scores[agent_id] = context["physical_scores"]
        self.last_semantic_scores[agent_id] = context["verification"]["semantic_scores"]

        return {
            "global_step": int(self.global_step),
            "env_step": int(getattr(self.env, "current_step", 0)),
            "agent_id": agent_id,
            "enemy_id": context["state"]["enemy_id"],
            "labels": context["labels"],
            "metrics": metrics_for_logging(context["raw_metrics"]),
            "verification": context["verification"],
            "semantic_info": context["semantic_info"],
            "reward_terms": {
                "env_reward": float(env_reward),
                "physical_reward": float(context["physical_reward"]),
                "semantic_phi": float(context["semantic_phi"]),
                "semantic_shaping_reward": float(shaping_reward),
                "total_reward": float(total_reward),
                "alpha": float(alpha),
                "beta": float(beta),
            },
            "curriculum": {
                "current_stage": stage.get("name", "unknown"),
                "situation_type": getattr(self.env, "current_situation_type", "default"),
                "situation_stage": getattr(self.env, "last_stage_name", None),
                "alpha": float(alpha),
                "beta": float(beta),
                "semantic_call_interval": int(stage.get("semantic_call_interval", 1)),
                "semantic_source": stage.get("semantic_source", self.reward_config.get("semantic_source", "physical")),
                "opponent_level": stage.get("opponent_level", "unchanged"),
                "situation_level": stage.get("situation_level", "unchanged"),
            },
        }

    def _compute_context(self, agent_id: str, stage: Dict, initialize: bool) -> Dict:
        state = self.extractor.extract(self.env, agent_id)
        metrics = compute_aircombat_metrics(state)
        previous_metrics = None if initialize else self.prev_metrics.get(agent_id)
        previous_scores = None if initialize else self.prev_label_scores.get(agent_id)
        labels, neutral_count = self.labeler.evaluate(
            metrics,
            previous_metrics=previous_metrics,
            previous_label_scores=previous_scores,
            neutral_steps=self.neutral_steps.get(agent_id, 0),
        )
        self.neutral_steps[agent_id] = neutral_count

        physical_scores = labels_to_scores(labels)
        semantic_source = stage.get("semantic_source", self.reward_config.get("semantic_source", "physical"))
        interval = int(stage.get("semantic_call_interval", 1))
        self.scorer.set_source(semantic_source)
        semantic_scores = self._semantic_scores(agent_id, state, metrics, labels, physical_scores, interval)
        semantic_info = dict(self.scorer.last_info)
        verification = self.verifier.verify(physical_scores, semantic_scores)
        if bool(self.reward_config.get("use_label_fusion", False)):
            if self._surrogate_fused_direct(semantic_source, semantic_info):
                verification = self._direct_surrogate_verification(verification)
            else:
                verification = self.fusion_model.fuse(verification)
            potential_scores = verification["fused_scores"]
        else:
            verification["fused_scores"] = dict(verification["verified_scores"])
            verification["fusion_policy"] = {
                name: {"policy": "verified_vlm", "lambda_vlm": 1.0, "physical_override": False}
                for name in LABEL_NAMES
            }
            verification["override_info"] = {
                name: {
                    "override_triggered": False,
                    "override_reason": None,
                    "original_vlm_score": verification["semantic_scores"][name],
                    "physical_score": verification["physical_scores"][name],
                    "verified_vlm_score": verification["verified_scores"][name],
                    "confidence": verification["confidence_scores"][name],
                    "fused_score": verification["verified_scores"][name],
                }
                for name in LABEL_NAMES
            }
            potential_scores = verification["fused_scores"]
        phi = self.reward_model.potential(potential_scores)
        physical_reward = self._physical_reward(physical_scores)
        return {
            "state": state,
            "raw_metrics": metrics,
            "labels": labels,
            "physical_scores": physical_scores,
            "verification": verification,
            "semantic_info": semantic_info,
            "semantic_phi": phi,
            "physical_reward": physical_reward,
        }

    def _physical_reward(self, physical_scores: Dict[str, float]) -> float:
        mode = self.reward_config.get("physical_reward_from", "weighted_scores")
        if mode in ("none", "off", False):
            return 0.0
        if mode == "weighted_scores":
            return self.reward_model.weighted_physical_reward(physical_scores)
        raise ValueError("Unknown physical_reward_from mode: {}".format(mode))

    def _semantic_scores(
        self,
        agent_id: str,
        state: Dict,
        metrics: Dict,
        labels: Dict,
        physical_scores: Dict[str, float],
        interval: int,
    ) -> Dict[str, float]:
        should_call = interval > 0 and (self.global_step % interval == 0 or agent_id not in self.last_semantic_scores)
        if should_call:
            cache_key = state_cache_key(state)
            return self.scorer.score(state, physical_labels=labels, cache_key=cache_key, metrics=metrics)
        self.scorer.last_info = {
            "source": "cached_interval_reuse",
            "cache_hit": None,
            "vlm_valid": None,
            "fallback_reason": None,
        }
        return {
            name: float(np.clip(self.last_semantic_scores.get(agent_id, physical_scores).get(name, physical_scores[name]), 0.0, 1.0))
            for name in LABEL_NAMES
        }

    def _surrogate_fused_direct(self, semantic_source: str, semantic_info: Dict) -> bool:
        if semantic_source != "surrogate":
            return False
        output_mode = self.config.get("surrogate", {}).get("output_mode", "fused_scores_direct")
        return output_mode == "fused_scores_direct" and semantic_info.get("source") in (
            "surrogate",
            "surrogate_fallback_physical",
            "cached_interval_reuse",
        )

    def _direct_surrogate_verification(self, verification: Dict) -> Dict:
        fused_scores = dict(verification["semantic_scores"])
        verification["fused_scores"] = fused_scores
        verification["fusion_policy"] = {
            name: {
                "policy": "surrogate_fused_direct",
                "lambda_vlm": 1.0,
                "physical_override": False,
                "base_reason": "surrogate_trained_on_fused_scores",
            }
            for name in LABEL_NAMES
        }
        verification["override_info"] = {
            name: {
                "override_triggered": False,
                "override_reason": None,
                "original_vlm_score": verification["semantic_scores"][name],
                "physical_score": verification["physical_scores"][name],
                "verified_vlm_score": verification["verified_scores"][name],
                "confidence": verification["confidence_scores"][name],
                "fused_score": fused_scores[name],
            }
            for name in LABEL_NAMES
        }
        return verification

    def _rl_agent_ids(self):
        return (self.env.ego_ids + self.env.enm_ids)[: self.env.num_agents]

    def _step_summary(self, shaping_info: Dict) -> Dict:
        if not shaping_info:
            return {}
        total = np.mean([detail["reward_terms"]["total_reward"] for detail in shaping_info.values()])
        env_reward = np.mean([detail["reward_terms"]["env_reward"] for detail in shaping_info.values()])
        mean_phi = np.mean([detail["reward_terms"]["semantic_phi"] for detail in shaping_info.values()])
        mean_conf = np.mean([
            np.mean(list(detail["verification"]["confidence_scores"].values())) for detail in shaping_info.values()
        ])
        return {
            "mean_total_reward": float(total),
            "mean_env_reward": float(env_reward),
            "mean_phi": float(mean_phi),
            "mean_confidence": float(mean_conf),
        }
