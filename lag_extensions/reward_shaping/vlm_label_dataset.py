import json
import os
from typing import Dict, Optional

import numpy as np

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv

from .aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from .config import load_reward_shaping_config
from .logger import to_jsonable
from .situation_renderer import SituationRenderer
from .state_extractor import StateExtractor
from .tactical_labels import TacticalLabeler
from .vlm_prompt import PROMPT_VERSION


def generate_aircombat_semantic_dataset(
    scenario_name: str,
    output_dir: str,
    num_samples: int = 100,
    sample_interval: int = 10,
    config_path: Optional[str] = None,
    seed: int = 1,
    agent_side: str = "ego",
) -> Dict:
    config = load_reward_shaping_config(config_path) if config_path else load_reward_shaping_config()
    os.makedirs(output_dir, exist_ok=True)
    env = SingleCombatEnv(scenario_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.reset()

    extractor = StateExtractor()
    labeler = TacticalLabeler(config.get("thresholds", {}))
    renderer_config = dict(config.get("renderer", {}))
    renderer_config.setdefault("prompt_version", config.get("vlm", {}).get("prompt_version", PROMPT_VERSION))
    renderer = SituationRenderer(renderer_config, config.get("thresholds", {}))

    sample_count = 0
    rollout_step = 0
    previous_metrics = {}
    previous_scores = {}
    neutral_steps = {}
    index_path = os.path.join(output_dir, "index.jsonl")
    agent_ids = _agent_ids(env, agent_side)

    with open(index_path, "w", encoding="utf-8") as index_file:
        while sample_count < num_samples:
            if rollout_step % sample_interval == 0:
                for agent_id in agent_ids:
                    if sample_count >= num_samples:
                        break
                    sample_id = "sample_{:06d}".format(sample_count)
                    sample_dir = os.path.abspath(os.path.join(output_dir, sample_id))
                    os.makedirs(sample_dir, exist_ok=True)
                    state = extractor.extract(env, agent_id)
                    metrics = compute_aircombat_metrics(state)
                    labels, neutral_count = labeler.evaluate(
                        metrics,
                        previous_metrics=previous_metrics.get(agent_id),
                        previous_label_scores=previous_scores.get(agent_id),
                        neutral_steps=neutral_steps.get(agent_id, 0),
                    )
                    neutral_steps[agent_id] = neutral_count
                    previous_metrics[agent_id] = metrics
                    previous_scores[agent_id] = {k: v["score"] for k, v in labels.items()}
                    metadata = renderer.render_sample(
                        state,
                        metrics,
                        labels,
                        sample_dir,
                        sample_id=sample_id,
                        image_filename="image.png",
                        metadata_filename="metadata.json",
                    )
                    _write_json(os.path.join(sample_dir, "physical_labels.json"), labels)
                    _write_json(os.path.join(sample_dir, "metrics.json"), metrics_for_logging(metrics))
                    _write_json(os.path.join(sample_dir, "env_state_snapshot.json"), _state_snapshot(state))
                    index_file.write(json.dumps(to_jsonable({
                        "sample_id": sample_id,
                        "agent_id": agent_id,
                        "image_path": metadata["image_path"],
                        "metadata_path": metadata["metadata_path"],
                        "state_hash": metadata["state_hash"],
                    }), sort_keys=True) + "\n")
                    sample_count += 1

            actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
            _, _, dones, _ = env.step(actions)
            rollout_step += 1
            if np.all(dones):
                env.reset()
                previous_metrics.clear()
                previous_scores.clear()
                neutral_steps.clear()
        env.close()

    summary = {
        "scenario_name": scenario_name,
        "output_dir": os.path.abspath(output_dir),
        "num_samples": sample_count,
        "sample_interval": sample_interval,
        "index_path": os.path.abspath(index_path),
    }
    _write_json(os.path.join(output_dir, "dataset_summary.json"), summary)
    return summary


def _agent_ids(env, agent_side: str):
    if agent_side == "ego":
        return [env.ego_ids[0]]
    if agent_side == "all":
        return (env.ego_ids + env.enm_ids)[: env.num_agents]
    raise ValueError("Unsupported agent_side: {}".format(agent_side))


def _state_snapshot(state: Dict) -> Dict:
    return {
        "agent_id": state["agent_id"],
        "enemy_id": state["enemy_id"],
        "ego": state["ego"],
        "enemy": state["enemy"],
        "relative": state["relative"],
    }


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
