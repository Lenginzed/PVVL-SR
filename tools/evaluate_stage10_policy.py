#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from algorithms.ppo.ppo_policy import PPOPolicy
from config import get_config
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.situation_curriculum import SituationCurriculumWrapper
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler


def main():
    parser = argparse.ArgumentParser(description="Evaluate a stage10 PPO checkpoint with tactical metrics.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--actor-file", default=None)
    parser.add_argument("--critic-file", default=None)
    parser.add_argument("--policy", choices=["checkpoint", "random"], default="checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/surrogate_v2_direct_1v1.yaml")
    parser.add_argument("--output-dir", default="scripts/results/stage10_policy_eval")
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def evaluate(args):
    if args.env_name != "SingleCombat":
        raise NotImplementedError("Only SingleCombat evaluation is implemented.")
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_reward_shaping_config(args.config)
    env = SingleCombatEnv(args.scenario_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    if config.get("situation_curriculum", {}).get("enabled", False):
        env = SituationCurriculumWrapper(env, config.get("situation_curriculum", {}), run_name="eval_env0")
    extractor = StateExtractor()
    labeler = TacticalLabeler(config.get("thresholds", {}))
    policy = None
    all_args = None
    device = torch.device("cpu")
    if args.policy == "checkpoint":
        if not args.model_dir:
            raise ValueError("--model-dir is required for checkpoint policy.")
        all_args = get_config().parse_args([])
        policy = PPOPolicy(all_args, env.observation_space, env.action_space, device=device)
        actor_path = args.actor_file or os.path.join(args.model_dir, "actor_latest.pt")
        critic_path = args.critic_file or os.path.join(args.model_dir, "critic_latest.pt")
        policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
        if os.path.exists(critic_path):
            policy.critic.load_state_dict(torch.load(critic_path, map_location=device))
        policy.prep_rollout()

    episode_rows = []
    for episode in range(int(args.episodes)):
        episode_seed = int(args.seed) + episode
        env.seed(episode_seed)
        env.action_space.seed(episode_seed)
        obs = env.reset()
        episode_reward = np.zeros(env.num_agents, dtype=np.float64)
        label_counts = {name: [] for name in LABEL_NAMES}
        label_scores = {name: [] for name in LABEL_NAMES}
        distances = []
        energy_diffs = []
        distance_rates = []
        ego_tail_angles = []
        ego_aim_angles = []
        attack_entry_steps = []
        curriculum_types = []
        action_saturations = []
        survival_steps = 0
        prev_metrics = {}
        prev_label_scores = {}
        neutral_steps = {}
        if all_args is not None:
            rnn_states = np.zeros((env.num_agents, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size), dtype=np.float32)
            masks = np.ones((env.num_agents, 1), dtype=np.float32)
        done_env = False
        for step in range(int(args.max_steps)):
            if args.policy == "random":
                actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
            else:
                with torch.no_grad():
                    action_tensor, rnn_tensor = policy.act(obs, rnn_states, masks, deterministic=args.deterministic)
                actions = action_tensor.detach().cpu().numpy()
                rnn_states = rnn_tensor.detach().cpu().numpy()
            action_saturations.append(_action_saturation_fraction(actions, env.action_space))
            obs, rewards, dones, _ = env.step(actions)
            rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(env.num_agents, -1)[:, 0]
            episode_reward += rewards_arr
            for agent_id in (env.ego_ids + env.enm_ids)[: env.num_agents]:
                state = extractor.extract(env, agent_id)
                metrics = compute_aircombat_metrics(state)
                labels, current_neutral_steps = labeler.evaluate(
                    metrics,
                    previous_metrics=prev_metrics.get(agent_id),
                    previous_label_scores=prev_label_scores.get(agent_id),
                    neutral_steps=neutral_steps.get(agent_id, 0),
                )
                prev_metrics[agent_id] = metrics
                prev_label_scores[agent_id] = {name: labels[name]["score"] for name in LABEL_NAMES}
                neutral_steps[agent_id] = current_neutral_steps
                for name in LABEL_NAMES:
                    label_counts[name].append(float(labels[name]["hard"]))
                    label_scores[name].append(float(labels[name]["score"]))
                logged = metrics_for_logging(metrics)
                distances.append(float(logged["distance"]))
                energy_diffs.append(float(logged["energy_difference"]))
                distance_rates.append(float(logged["distance_rate"]))
                ego_tail_angles.append(float(logged["ego_tail_angle_deg"]))
                ego_aim_angles.append(float(logged["ego_aim_angle_deg"]))
                if labels["effective_attack_window"]["hard"]:
                    attack_entry_steps.append(step)
                curriculum_types.append(getattr(env, "current_situation_type", "default"))
            survival_steps = step + 1
            done_env = bool(np.all(dones))
            if all_args is not None:
                masks = np.ones((env.num_agents, 1), dtype=np.float32)
                masks[np.asarray(dones).reshape(env.num_agents, -1)[:, 0] == True] = 0.0
            if done_env:
                break
        sim = env.agents[env.ego_ids[0]]
        win = bool(sim.is_alive and all(not enemy.is_alive for enemy in sim.enemies))
        loss = bool(not sim.is_alive)
        timeout = bool(not done_env and survival_steps >= int(args.max_steps))
        draw = bool(not win and not loss)
        row = {
            "episode": episode,
            "return_mean_agents": float(np.mean(episode_reward)),
            "return_ego": float(episode_reward[0]),
            "episode_length": int(survival_steps),
            "win": win,
            "loss": loss,
            "draw": draw,
            "timeout": timeout,
            "survival_time": float(survival_steps * getattr(env, "time_interval", 0.0)),
            "average_distance": float(np.mean(distances)) if distances else 0.0,
            "average_delta_E": float(np.mean(energy_diffs)) if energy_diffs else 0.0,
            "average_d_dot": float(np.mean(distance_rates)) if distance_rates else 0.0,
            "min_ego_tail_angle": float(np.min(ego_tail_angles)) if ego_tail_angles else 0.0,
            "min_ego_aim_angle": float(np.min(ego_aim_angles)) if ego_aim_angles else 0.0,
            "attack_window_entry_count": int(len(attack_entry_steps)),
            "attack_window_first_entry_time": float(min(attack_entry_steps) * getattr(env, "time_interval", 0.0)) if attack_entry_steps else None,
            "curriculum_type": _dominant(curriculum_types),
            "action_saturation_fraction": float(np.mean(action_saturations)) if action_saturations else 0.0,
        }
        row["defensive_escape_success_ratio"] = _safe_mean(label_counts["defensive_escape"])
        for name in LABEL_NAMES:
            row[name + "_time_ratio"] = _safe_mean(label_counts[name])
            row[name + "_score_mean"] = _safe_mean(label_scores[name])
        episode_rows.append(row)
    env.close()

    summary = _summarize(episode_rows)
    summary.update({
        "env_name": args.env_name,
        "scenario_name": args.scenario_name,
        "policy": args.policy,
        "model_dir": os.path.abspath(args.model_dir) if args.model_dir else None,
        "actor_file": os.path.abspath(args.actor_file) if args.actor_file else None,
        "episodes": int(args.episodes),
        "output_dir": os.path.abspath(args.output_dir),
    })
    _write_json(os.path.join(args.output_dir, "summary.json"), summary)
    _write_jsonl(os.path.join(args.output_dir, "episodes.jsonl"), episode_rows)
    return summary


def _action_saturation_fraction(actions, action_space):
    arr = np.asarray(actions)
    if hasattr(action_space, "low") and hasattr(action_space, "high"):
        low = np.asarray(action_space.low)
        high = np.asarray(action_space.high)
        return float(np.mean(np.logical_or(np.isclose(arr, low), np.isclose(arr, high))))
    if hasattr(action_space, "nvec"):
        nvec = np.asarray(action_space.nvec)
        return float(np.mean(np.logical_or(arr == 0, arr == (nvec - 1))))
    return 0.0


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _summarize(rows):
    if not rows:
        return {}
    summary = {}
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, bool)) and key != "episode"]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        summary[key + "_mean"] = float(np.mean(values))
        summary[key + "_std"] = float(np.std(values))
    summary["win_rate"] = float(np.mean([row["win"] for row in rows]))
    summary["loss_rate"] = float(np.mean([row["loss"] for row in rows]))
    summary["draw_rate"] = float(np.mean([row["draw"] for row in rows]))
    summary["timeout_ratio"] = float(np.mean([row["timeout"] for row in rows]))
    curriculum_types = sorted(set(row.get("curriculum_type", "default") for row in rows))
    summary["curriculum_type_counts"] = {name: int(sum(row.get("curriculum_type", "default") == name for row in rows)) for name in curriculum_types}
    summary["by_curriculum_type"] = {}
    for name in curriculum_types:
        subset = [row for row in rows if row.get("curriculum_type", "default") == name]
        if subset:
            summary["by_curriculum_type"][name] = {
                "episode_count": len(subset),
                "win_rate": float(np.mean([row["win"] for row in subset])),
                "draw_rate": float(np.mean([row["draw"] for row in subset])),
                "ego_tail_advantage_time_ratio_mean": _safe_mean([row["ego_tail_advantage_time_ratio"] for row in subset]),
                "effective_attack_window_time_ratio_mean": _safe_mean([row["effective_attack_window_time_ratio"] for row in subset]),
                "neutral_stalemate_time_ratio_mean": _safe_mean([row["neutral_stalemate_time_ratio"] for row in subset]),
                "average_distance_mean": _safe_mean([row["average_distance"] for row in subset]),
            }
    return summary


def _dominant(values):
    if not values:
        return "default"
    counts = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
