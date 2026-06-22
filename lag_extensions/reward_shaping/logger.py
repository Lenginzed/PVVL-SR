import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .config import LABEL_NAMES, snapshot_config


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


class RewardShapingLogger:
    def __init__(self, log_dir: str, config: Dict, run_name: str = "env0"):
        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(self.log_dir, exist_ok=True)
        self.step_path = os.path.join(self.log_dir, "{}_steps.jsonl".format(run_name))
        self.episode_jsonl_path = os.path.join(self.log_dir, "{}_episodes.jsonl".format(run_name))
        self.episode_csv_path = os.path.join(self.log_dir, "{}_episodes.csv".format(run_name))
        self.episode_records = defaultdict(list)
        self.episode_index = defaultdict(int)
        snapshot_config(config, os.path.join(self.log_dir, "{}_config_snapshot.json".format(run_name)))

    def reset_episode(self, agent_ids: List[str]) -> None:
        for agent_id in agent_ids:
            self.episode_records[agent_id] = []

    def log_step(self, agent_id: str, record: Dict) -> None:
        self.episode_records[agent_id].append(record)
        with open(self.step_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")

    def finish_episode(self, env, agent_id: str) -> Dict:
        records = self.episode_records.get(agent_id, [])
        if not records:
            return {}
        summary = self._summarize(env, agent_id, records)
        self.episode_index[agent_id] += 1
        with open(self.episode_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(summary), sort_keys=True) + "\n")
        self._write_csv(summary)
        self.episode_records[agent_id] = []
        return summary

    def _summarize(self, env, agent_id: str, records: List[Dict]) -> Dict:
        length = len(records)
        sim = env.agents[agent_id]
        win = bool(sim.is_alive and all(not enemy.is_alive for enemy in sim.enemies))
        loss = bool(not sim.is_alive)
        draw = bool(not win and not loss)
        final_status = "win" if win else "loss" if loss else "draw"

        rewards = [r["reward_terms"] for r in records]
        metrics = [r["metrics"] for r in records]
        labels = [r["labels"] for r in records]
        verification = [r["verification"] for r in records]

        summary = {
            "run_name": self.run_name,
            "episode": self.episode_index[agent_id],
            "agent_id": agent_id,
            "episode_reward_env": float(sum(r["env_reward"] for r in rewards)),
            "episode_reward_total": float(sum(r["total_reward"] for r in rewards)),
            "episode_length": int(length),
            "final_status": final_status,
            "win": win,
            "loss": loss,
            "draw": draw,
            "missile_hit": bool(any(missile.is_success for missile in sim.launch_missiles)),
            "survival_time": float(env.current_step * getattr(env, "time_interval", 0.0)),
            "average_energy_difference": float(np.mean([m["energy_difference"] for m in metrics])),
            "average_distance": float(np.mean([m["distance"] for m in metrics])),
            "mean_phi": float(np.mean([r["semantic_phi"] for r in rewards])),
            "mean_semantic_shaping_reward": float(np.mean([r["semantic_shaping_reward"] for r in rewards])),
            "mean_physical_reward": float(np.mean([r["physical_reward"] for r in rewards])),
            "mean_confidence": float(np.mean([
                np.mean(list(v["confidence_scores"].values())) for v in verification
            ])),
            "current_stage": records[-1]["curriculum"]["current_stage"],
            "alpha": float(records[-1]["curriculum"]["alpha"]),
            "beta": float(records[-1]["curriculum"]["beta"]),
            "semantic_call_interval": int(records[-1]["curriculum"]["semantic_call_interval"]),
        }

        for name in LABEL_NAMES:
            hard_values = [float(step_labels[name]["hard"]) for step_labels in labels]
            label_scores = [float(step_labels[name]["score"]) for step_labels in labels]
            confidences = [float(v["confidence_scores"][name]) for v in verification]
            semantic_scores = [float(v["semantic_scores"][name]) for v in verification]
            summary[name + "_time_ratio"] = float(np.mean(hard_values))
            summary[name + "_mean_score"] = float(np.mean(label_scores))
            summary[name + "_mean_confidence"] = float(np.mean(confidences))
            summary[name + "_mean_semantic_score"] = float(np.mean(semantic_scores))
        return summary

    def _write_csv(self, summary: Dict) -> None:
        file_exists = os.path.exists(self.episode_csv_path)
        fieldnames = list(summary.keys())
        with open(self.episode_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)
