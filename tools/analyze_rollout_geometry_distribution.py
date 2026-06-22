#!/usr/bin/env python
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from algorithms.ppo.ppo_policy import PPOPolicy
from config import get_config
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler
from tools.stage11_common import (
    ensure_dir,
    env_metrics_for_agents,
    labels_from_metrics_stream,
    parse_stage10_best_runs,
    summarize,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_GROUPS = [
    "random",
    "PPO_original",
    "PPO_surrogate_v2_then_fusion",
    "PPO_surrogate_v2_direct_beta_low",
    "PPO_physical",
]


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout geometry distributions for random and trained policies.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml")
    parser.add_argument("--stage10-dir", default="scripts/results/stage10_multiseed_ppo")
    parser.add_argument("--policies", nargs="*", default=DEFAULT_GROUPS)
    parser.add_argument("--rollout-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1110)
    parser.add_argument("--output-dir", default="scripts/results/stage11_rollout_geometry_distribution")
    args = parser.parse_args()

    if args.env_name != "SingleCombat":
        raise NotImplementedError("Only SingleCombat is supported.")

    out_dir = ensure_dir(args.output_dir)
    config = load_reward_shaping_config(args.config)
    thresholds = config.get("thresholds", {})
    trained_groups = [name for name in args.policies if name != "random"]
    best_runs = parse_stage10_best_runs(args.stage10_dir, trained_groups)

    all_rows = []
    summaries = {}
    for policy_name in args.policies:
        model_dir = None if policy_name == "random" else best_runs.get(policy_name)
        rows = rollout_policy(args, policy_name, model_dir, thresholds)
        all_rows.extend(rows)
        summaries[policy_name] = summarize_policy(rows, thresholds, model_dir)

    top20 = sorted(all_rows, key=lambda row: (-row["effective_attack_window_score"], row["attack_margin"]))[:20]
    write_json(os.path.join(out_dir, "summary.json"), {
        "policies": summaries,
        "thresholds": thresholds,
        "rollout_steps_per_policy": int(args.rollout_steps),
    })
    write_csv(os.path.join(out_dir, "per_policy_distribution.csv"), flatten_summary(summaries))
    write_csv(os.path.join(out_dir, "rollout_metrics.csv"), all_rows)
    write_jsonl(os.path.join(out_dir, "top20_closest_attack_window.jsonl"), top20)
    write_report(os.path.join(out_dir, "geometry_report.md"), summaries, top20)
    try_plot(out_dir, all_rows)
    print(os.path.abspath(os.path.join(out_dir, "geometry_report.md")))


def rollout_policy(args, policy_name, model_dir, thresholds):
    env = SingleCombatEnv(args.scenario_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    policy, all_args = load_policy(env, model_dir) if model_dir else (None, None)
    labeler = TacticalLabeler(thresholds)

    obs = env.reset()
    if all_args is not None:
        rnn_states = np.zeros((env.num_agents, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size), dtype=np.float32)
        masks = np.ones((env.num_agents, 1), dtype=np.float32)
    prev_metrics = {}
    prev_scores = {}
    neutral_steps = {}
    rows = []
    episode = 0
    local_step = 0
    for step in range(int(args.rollout_steps)):
        if policy is None:
            actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
        else:
            with torch.no_grad():
                action_tensor, rnn_tensor = policy.act(obs, rnn_states, masks, deterministic=True)
            actions = action_tensor.detach().cpu().numpy()
            rnn_states = rnn_tensor.detach().cpu().numpy()
        obs, rewards, dones, _ = env.step(actions)
        local_step += 1

        for metric_row in env_metrics_for_agents(env):
            agent_id = metric_row["agent_id"]
            labels, count = labeler.evaluate(
                {
                    "distance": metric_row["distance"],
                    "distance_rate": metric_row["distance_rate"],
                    "ego_aim_angle": math.radians(metric_row["ego_aim_angle_deg"]),
                    "enemy_aim_angle": math.radians(metric_row["enemy_aim_angle_deg"]),
                    "ego_tail_angle": math.radians(metric_row["ego_tail_angle_deg"]),
                    "enemy_tail_angle": math.radians(metric_row["enemy_tail_angle_deg"]),
                    "ego_specific_energy": metric_row["ego_specific_energy"],
                    "enemy_specific_energy": metric_row["enemy_specific_energy"],
                    "energy_difference": metric_row["energy_difference"],
                    "enemy_lock": False,
                    "ego_missile_warning": False,
                },
                previous_metrics=prev_metrics.get(agent_id),
                previous_label_scores=prev_scores.get(agent_id),
                neutral_steps=neutral_steps.get(agent_id, 0),
            )
            prev_metrics[agent_id] = {
                "enemy_aim_angle": math.radians(metric_row["enemy_aim_angle_deg"]),
                "distance_rate": metric_row["distance_rate"],
            }
            prev_scores[agent_id] = {name: float(labels[name]["score"]) for name in LABEL_NAMES}
            neutral_steps[agent_id] = count
            row = {
                "policy": policy_name,
                "model_dir": model_dir or "",
                "episode": episode,
                "episode_step": local_step,
                "global_step": step,
                **metric_row,
                "attack_margin": attack_margin(metric_row, thresholds),
            }
            for name in LABEL_NAMES:
                row[name + "_hard"] = float(labels[name]["hard"])
                row[name + "_score"] = float(labels[name]["score"])
            rows.append(row)

        done_env = bool(np.all(dones))
        if all_args is not None:
            masks = np.ones((env.num_agents, 1), dtype=np.float32)
            masks[np.asarray(dones).reshape(env.num_agents, -1)[:, 0] == True] = 0.0
        if done_env:
            episode += 1
            local_step = 0
            obs = env.reset()
            prev_metrics.clear()
            prev_scores.clear()
            neutral_steps.clear()
            if all_args is not None:
                rnn_states = np.zeros((env.num_agents, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size), dtype=np.float32)
                masks = np.ones((env.num_agents, 1), dtype=np.float32)
    env.close()
    return rows


def load_policy(env, model_dir):
    if not model_dir:
        return None, None
    all_args = get_config().parse_args([])
    policy = PPOPolicy(all_args, env.observation_space, env.action_space, device=torch.device("cpu"))
    actor_path = os.path.join(model_dir, "actor_latest.pt")
    critic_path = os.path.join(model_dir, "critic_latest.pt")
    policy.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    if os.path.exists(critic_path):
        policy.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
    policy.prep_rollout()
    return policy, all_args


def attack_margin(row, thresholds):
    distance = float(row["distance"])
    d_min = float(thresholds["d_attack_min"])
    d_max = float(thresholds["d_attack_max"])
    range_penalty = 0.0
    if distance < d_min:
        range_penalty = (d_min - distance) / max(d_min, 1e-8)
    elif distance > d_max:
        range_penalty = (distance - d_max) / max(d_max, 1e-8)
    aim_penalty = max(0.0, float(row["ego_aim_angle_deg"]) - float(thresholds["theta_launch_deg"])) / max(float(thresholds["theta_launch_deg"]), 1e-8)
    tail_penalty = max(0.0, float(row["ego_tail_angle_deg"]) - float(thresholds["theta_target_deg"])) / max(float(thresholds["theta_target_deg"]), 1e-8)
    closing_penalty = 0.0 if -float(row["distance_rate"]) > float(thresholds.get("closing_rate_threshold", 0.0)) else 1.0
    return float(range_penalty + aim_penalty + tail_penalty + closing_penalty)


def summarize_policy(rows, thresholds, model_dir):
    summary = {"row_count": len(rows), "model_dir": model_dir}
    for key in [
        "distance",
        "distance_rate",
        "ego_aim_angle_deg",
        "enemy_aim_angle_deg",
        "ego_tail_angle_deg",
        "enemy_tail_angle_deg",
        "energy_difference",
        "attack_margin",
    ]:
        summary[key] = summarize([row[key] for row in rows])
    for name in LABEL_NAMES:
        hards = [row[name + "_hard"] for row in rows]
        scores = [row[name + "_score"] for row in rows]
        summary[name + "_trigger_rate"] = float(np.mean(hards)) if hards else 0.0
        summary[name + "_score"] = summarize(scores)
    per_episode = {}
    for row in rows:
        per_episode.setdefault(row["episode"], []).append(row)
    summary["episode_count"] = len(per_episode)
    summary["min_ego_tail_angle_per_episode"] = summarize([min(r["ego_tail_angle_deg"] for r in ep_rows) for ep_rows in per_episode.values()])
    summary["min_ego_aim_angle_per_episode"] = summarize([min(r["ego_aim_angle_deg"] for r in ep_rows) for ep_rows in per_episode.values()])
    summary["min_distance_per_episode"] = summarize([min(r["distance"] for r in ep_rows) for ep_rows in per_episode.values()])
    summary["attack_entry_count"] = int(sum(row["effective_attack_window_hard"] > 0.0 for row in rows))
    summary["tail_entry_count"] = int(sum(row["ego_tail_advantage_hard"] > 0.0 for row in rows))
    return summary


def flatten_summary(summaries):
    rows = []
    for policy, summary in summaries.items():
        row = {"policy": policy, "row_count": summary["row_count"], "episode_count": summary["episode_count"]}
        for key in ["distance", "ego_aim_angle_deg", "ego_tail_angle_deg", "attack_margin"]:
            for stat in ["mean", "p5", "p50", "p95", "min", "max"]:
                row[key + "_" + stat] = summary[key][stat]
        for label in ["ego_tail_advantage", "effective_attack_window", "enemy_tail_threat", "enemy_missile_threat_zone", "neutral_stalemate"]:
            row[label + "_trigger_rate"] = summary[label + "_trigger_rate"]
            row[label + "_score_max"] = summary[label + "_score"]["max"]
            row[label + "_score_p95"] = summary[label + "_score"]["p95"]
        rows.append(row)
    return rows


def write_report(path, summaries, top20):
    lines = [
        "# Stage11 Rollout Geometry Distribution",
        "",
        "| policy | rows | episodes | min ego_tail p50 | min ego_aim p50 | min distance p50 | tail hard | attack hard | attack score max | neutral hard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, summary in summaries.items():
        lines.append("| {} | {} | {} | {:.2f} | {:.2f} | {:.1f} | {:.6f} | {:.6f} | {:.4f} | {:.6f} |".format(
            policy,
            summary["row_count"],
            summary["episode_count"],
            summary["min_ego_tail_angle_per_episode"]["p50"],
            summary["min_ego_aim_angle_per_episode"]["p50"],
            summary["min_distance_per_episode"]["p50"],
            summary["ego_tail_advantage_trigger_rate"],
            summary["effective_attack_window_trigger_rate"],
            summary["effective_attack_window_score"]["max"],
            summary["neutral_stalemate_trigger_rate"],
        ))
    lines.extend(["", "## Top 20 Closest Cases To Attack Window", ""])
    for row in top20[:20]:
        lines.append("- `{}` step `{}` agent `{}`: attack_score={:.4f}, margin={:.4f}, distance={:.1f}, aim={:.1f}, tail={:.1f}, d_dot={:.1f}".format(
            row["policy"],
            row["global_step"],
            row["agent_id"],
            row["effective_attack_window_score"],
            row["attack_margin"],
            row["distance"],
            row["ego_aim_angle_deg"],
            row["ego_tail_angle_deg"],
            row["distance_rate"],
        ))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def try_plot(out_dir, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    metrics = ["distance", "ego_aim_angle_deg", "ego_tail_angle_deg", "attack_margin"]
    policies = sorted({row["policy"] for row in rows})
    for metric in metrics:
        plt.figure(figsize=(8, 4))
        for policy in policies:
            values = [row[metric] for row in rows if row["policy"] == policy]
            if values:
                plt.hist(values, bins=40, alpha=0.35, label=policy, density=True)
        plt.title(metric)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, metric + "_hist.png"), dpi=140)
        plt.close()


if __name__ == "__main__":
    main()
