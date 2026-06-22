#!/usr/bin/env python
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from algorithms.ppo.ppo_policy import PPOPolicy
from config import get_config
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.reward_wrapper import RewardShapingWrapper
from lag_extensions.reward_shaping.situation_curriculum import SituationCurriculumWrapper
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler


def main():
    parser = argparse.ArgumentParser(description="Visualize representative Stage12 tactical trajectories.")
    parser.add_argument("--record", action="append", default=[], help="name,model_dir,eval_config")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--output-dir", default="scripts/results/stage12_formal_ppo/trajectory_visualizations")
    args = parser.parse_args()
    records = parse_records(args.record)
    if not records:
        records = default_records()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for record in records:
        summaries.append(run_record(args, record, out_dir / record["name"]))
    (out_dir / "trajectory_visualization_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir.resolve()), "records": summaries}, indent=2, sort_keys=True))


def run_record(args, record, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = rollout(record, args)
    selected = select_representatives(episodes)
    write_json(out_dir / "selected_episodes.json", selected)
    for kind, episode in selected.items():
        if episode is None:
            continue
        rows = episode["rows"]
        write_csv(out_dir / "{}_timeseries.csv".format(kind), rows)
        write_plot(out_dir / "{}_plot.png".format(kind), rows, "{} {}".format(record["name"], kind))
    return {
        "name": record["name"],
        "model_dir": record["model_dir"],
        "selected": {key: value["summary"] if value else None for key, value in selected.items()},
    }


def rollout(record, args):
    config = load_reward_shaping_config(record["eval_config"])
    env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    if config.get("situation_curriculum", {}).get("enabled", False):
        env = SituationCurriculumWrapper(env, config.get("situation_curriculum", {}), run_name="vis_env0")
    if record.get("use_reward_shaping"):
        env = RewardShapingWrapper(
            env,
            config_path=record.get("reward_config") or record["eval_config"],
            enabled_override=True,
            log_dir=str(Path(args.output_dir) / "reward_logs" / record["name"]),
            run_name=record["name"],
        )
    policy, all_args = load_policy(env, record["model_dir"])
    extractor = StateExtractor()
    labeler = TacticalLabeler(config.get("thresholds", {}))
    episodes = []
    for episode in range(int(args.episodes)):
        env.seed(int(args.seed) + episode)
        env.action_space.seed(int(args.seed) + episode)
        obs = env.reset()
        rnn_states = np.zeros((env.num_agents, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size), dtype=np.float32)
        masks = np.ones((env.num_agents, 1), dtype=np.float32)
        prev_metrics = {}
        prev_scores = {}
        neutral_steps = {}
        rows = []
        total_reward = 0.0
        done_env = False
        for step in range(int(args.max_steps)):
            with torch.no_grad():
                action_tensor, rnn_tensor = policy.act(obs, rnn_states, masks, deterministic=True)
            actions = action_tensor.detach().cpu().numpy()
            obs, rewards, dones, info = env.step(actions)
            total_reward += float(np.asarray(rewards).reshape(env.num_agents, -1)[0, 0])
            detail = (info.get("reward_shaping", {}) if isinstance(info, dict) else {}).get(env.ego_ids[0], {})
            reward_terms = detail.get("reward_terms", {})
            agent_id = env.ego_ids[0]
            state = extractor.extract(env, agent_id)
            metrics = compute_aircombat_metrics(state)
            labels, neutral_count = labeler.evaluate(
                metrics,
                previous_metrics=prev_metrics.get(agent_id),
                previous_label_scores=prev_scores.get(agent_id),
                neutral_steps=neutral_steps.get(agent_id, 0),
            )
            prev_metrics[agent_id] = metrics
            prev_scores[agent_id] = {name: float(labels[name]["score"]) for name in LABEL_NAMES}
            neutral_steps[agent_id] = neutral_count
            logged = metrics_for_logging(metrics)
            row = {
                "episode": episode,
                "step": step,
                "time_sec": float(step * getattr(env, "time_interval", 0.0)),
                "ego_x": float(state["ego"]["position"][0]),
                "ego_y": float(state["ego"]["position"][1]),
                "enemy_x": float(state["enemy"]["position"][0]),
                "enemy_y": float(state["enemy"]["position"][1]),
                "distance": float(logged["distance"]),
                "delta_E": float(logged["energy_difference"]),
                "ego_tail_angle_deg": float(logged["ego_tail_angle_deg"]),
                "ego_aim_angle_deg": float(logged["ego_aim_angle_deg"]),
                "semantic_phi": float(reward_terms.get("semantic_phi", 0.0)),
                "semantic_shaping_reward": float(reward_terms.get("semantic_shaping_reward", 0.0)),
            }
            for name in LABEL_NAMES:
                row[name + "_hard"] = float(labels[name]["hard"])
                row[name + "_score"] = float(labels[name]["score"])
            rows.append(row)
            rnn_states = rnn_tensor.detach().cpu().numpy()
            masks = np.ones((env.num_agents, 1), dtype=np.float32)
            masks[np.asarray(dones).reshape(env.num_agents, -1)[:, 0] == True] = 0.0
            done_env = bool(np.all(dones))
            if done_env:
                break
        sim = env.agents[env.ego_ids[0]]
        win = bool(sim.is_alive and all(not enemy.is_alive for enemy in sim.enemies))
        loss = bool(not sim.is_alive)
        summary = summarize_episode(rows, total_reward, win, loss, done_env)
        episodes.append({"rows": rows, "summary": summary})
    env.close()
    return episodes


def summarize_episode(rows, total_reward, win, loss, done_env):
    return {
        "return": float(total_reward),
        "win": bool(win),
        "loss": bool(loss),
        "draw": bool(not win and not loss),
        "done": bool(done_env),
        "length": len(rows),
        "tail_ratio": mean([row["ego_tail_advantage_hard"] for row in rows]),
        "attack_ratio": mean([row["effective_attack_window_hard"] for row in rows]),
        "enemy_threat_ratio": mean([row["enemy_missile_threat_zone_hard"] for row in rows]),
        "neutral_ratio": mean([row["neutral_stalemate_hard"] for row in rows]),
    }


def select_representatives(episodes):
    successful = [ep for ep in episodes if ep["summary"]["win"] and ep["summary"]["attack_ratio"] > 0.02]
    defensive = [ep for ep in episodes if ep["summary"]["enemy_threat_ratio"] > 0.02 and not ep["summary"]["loss"]]
    stalemate = [ep for ep in episodes if ep["summary"]["draw"] or ep["summary"]["neutral_ratio"] > 0.3]
    failure = [ep for ep in episodes if ep["summary"]["loss"]]
    return {
        "successful_offensive": best(successful, "attack_ratio"),
        "defensive_recovery": best(defensive, "enemy_threat_ratio"),
        "stalemate_or_failure": best(stalemate, "neutral_ratio") or best(failure, "length"),
    }


def best(episodes, key):
    if not episodes:
        return None
    return max(episodes, key=lambda ep: ep["summary"].get(key, 0.0))


def write_plot(path, rows, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        path.with_suffix(".plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    t = [row["time_sec"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot([row["ego_y"] for row in rows], [row["ego_x"] for row in rows], label="ego", color="blue")
    axes[0, 0].plot([row["enemy_y"] for row in rows], [row["enemy_x"] for row in rows], label="enemy", color="red")
    axes[0, 0].set_title("2D trajectory")
    axes[0, 0].set_xlabel("east m")
    axes[0, 0].set_ylabel("north m")
    axes[0, 0].legend()
    axes[0, 1].plot(t, [row["ego_tail_advantage_score"] for row in rows], label="tail")
    axes[0, 1].plot(t, [row["effective_attack_window_score"] for row in rows], label="attack")
    axes[0, 1].plot(t, [row["neutral_stalemate_score"] for row in rows], label="neutral")
    axes[0, 1].set_title("tactical scores")
    axes[0, 1].legend()
    axes[1, 0].plot(t, [row["distance"] for row in rows])
    axes[1, 0].set_title("distance")
    axes[1, 1].plot(t, [row["delta_E"] for row in rows])
    axes[1, 1].plot(t, [row["semantic_phi"] for row in rows], label="phi")
    axes[1, 1].set_title("delta_E / semantic_phi")
    axes[1, 1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_policy(env, model_dir):
    all_args = get_config().parse_args([])
    policy = PPOPolicy(all_args, env.observation_space, env.action_space, device=torch.device("cpu"))
    policy.actor.load_state_dict(torch.load(Path(model_dir) / "actor_latest.pt", map_location="cpu"))
    critic_path = Path(model_dir) / "critic_latest.pt"
    if critic_path.exists():
        policy.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
    policy.prep_rollout()
    return policy, all_args


def parse_records(records):
    parsed = []
    for item in records:
        parts = [part.strip() for part in item.split(",")]
        if len(parts) not in (3, 5):
            raise ValueError("--record must be name,model_dir,eval_config[,reward_config,use_reward_shaping]")
        parsed.append({
            "name": parts[0],
            "model_dir": parts[1],
            "eval_config": parts[2],
            "reward_config": parts[3] if len(parts) == 5 and parts[3] else None,
            "use_reward_shaping": len(parts) == 5 and parts[4].lower() in ("1", "true", "yes"),
        })
    return parsed


def default_records():
    base = "scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo"
    return [
        {
            "name": "PPO_surrogate_v2_then_fusion_with_situation_curriculum_seed0",
            "model_dir": str(Path(base) / "stage12formal_PPO_surrogate_v2_then_fusion_with_situation_curriculum_seed0" / "run1"),
            "eval_config": "lag_extensions/reward_shaping/configs/mixed_curriculum_eval_1v1.yaml",
            "reward_config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_mixed_curriculum_1v1.yaml",
            "use_reward_shaping": True,
        },
        {
            "name": "PPO_surrogate_v2_direct_beta_low_with_situation_curriculum_seed0",
            "model_dir": str(Path(base) / "stage12formal_PPO_surrogate_v2_direct_beta_low_with_situation_curriculum_seed0" / "run1"),
            "eval_config": "lag_extensions/reward_shaping/configs/mixed_curriculum_eval_1v1.yaml",
            "reward_config": "lag_extensions/reward_shaping/configs/surrogate_v2_direct_beta_low_mixed_curriculum_1v1.yaml",
            "use_reward_shaping": True,
        },
    ]


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def mean(values):
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
