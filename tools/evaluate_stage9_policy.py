#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from config import get_config
from algorithms.ppo.ppo_policy import PPOPolicy
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler


def main():
    parser = argparse.ArgumentParser(description="Evaluate a stage9 PPO checkpoint or random policy with tactical metrics.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--policy", choices=["checkpoint", "random"], default="checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/surrogate_v1_hybrid_1v1.yaml")
    parser.add_argument("--output-dir", default="scripts/results/stage9_policy_eval")
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
        actor_path = os.path.join(args.model_dir, "actor_latest.pt")
        critic_path = os.path.join(args.model_dir, "critic_latest.pt")
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
        distances = []
        energy_diffs = []
        survival_steps = 0
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
            obs, rewards, dones, _ = env.step(actions)
            rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(env.num_agents, -1)[:, 0]
            episode_reward += rewards_arr
            for agent_id in (env.ego_ids + env.enm_ids)[: env.num_agents]:
                state = extractor.extract(env, agent_id)
                metrics = compute_aircombat_metrics(state)
                labels, _ = labeler.evaluate(metrics)
                for name in LABEL_NAMES:
                    label_counts[name].append(float(labels[name]["hard"]))
                logged = metrics_for_logging(metrics)
                distances.append(float(logged["distance"]))
                energy_diffs.append(float(logged["energy_difference"]))
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
        draw = bool(not win and not loss)
        row = {
            "episode": episode,
            "return_mean_agents": float(np.mean(episode_reward)),
            "return_ego": float(episode_reward[0]),
            "episode_length": int(survival_steps),
            "win": win,
            "loss": loss,
            "draw": draw,
            "survival_time": float(survival_steps * getattr(env, "time_interval", 0.0)),
            "average_distance": float(np.mean(distances)) if distances else 0.0,
            "average_delta_E": float(np.mean(energy_diffs)) if energy_diffs else 0.0,
        }
        for name in LABEL_NAMES:
            row[name + "_time_ratio"] = float(np.mean(label_counts[name])) if label_counts[name] else 0.0
        episode_rows.append(row)
    env.close()

    summary = _summarize(episode_rows)
    summary.update({
        "env_name": args.env_name,
        "scenario_name": args.scenario_name,
        "policy": args.policy,
        "model_dir": os.path.abspath(args.model_dir) if args.model_dir else None,
        "episodes": int(args.episodes),
        "output_dir": os.path.abspath(args.output_dir),
    })
    _write_json(os.path.join(args.output_dir, "summary.json"), summary)
    _write_jsonl(os.path.join(args.output_dir, "episodes.jsonl"), episode_rows)
    return summary


def _summarize(rows):
    if not rows:
        return {}
    summary = {}
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "episode"]
    for key in numeric_keys:
        summary[key + "_mean"] = float(np.mean([row[key] for row in rows]))
        summary[key + "_std"] = float(np.std([row[key] for row in rows]))
    summary["win_rate"] = float(np.mean([row["win"] for row in rows]))
    summary["loss_rate"] = float(np.mean([row["loss"] for row in rows]))
    summary["draw_rate"] = float(np.mean([row["draw"] for row in rows]))
    return summary


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
