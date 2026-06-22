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
from lag_extensions.reward_shaping.reward_wrapper import RewardShapingWrapper
from lag_extensions.reward_shaping.situation_curriculum import SituationCurriculumWrapper
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler
from tools.stage11_common import ensure_dir, summarize, write_csv, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Audit why offensive curriculum policies keep or lose attack geometry.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--record", action="append", default=[], help="Comma format: name,model_dir,config,use_reward_shaping")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--output-dir", default="scripts/results/stage11b_offensive_reward_audit")
    args = parser.parse_args()
    records = parse_records(args.record)
    if not records:
        records = default_records()
    all_rows = []
    failure_rows = []
    summaries = {}
    for record in records:
        rows, failures, summary = audit_record(args, record)
        all_rows.extend(rows)
        failure_rows.extend(failures)
        summaries[record["name"]] = summary
    ensure_dir(args.output_dir)
    write_csv(os.path.join(args.output_dir, "per_episode_curves.csv"), all_rows)
    write_jsonl(os.path.join(args.output_dir, "typical_failure_cases.jsonl"), failure_rows[:50])
    write_json(os.path.join(args.output_dir, "summary.json"), summaries)
    write_report(os.path.join(args.output_dir, "reward_audit_report.md"), summaries)
    print(os.path.abspath(os.path.join(args.output_dir, "reward_audit_report.md")))


def audit_record(args, record):
    rows = []
    failures = []
    env = SingleCombatEnv(args.scenario_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    situation = {
        "enabled": True,
        "mode": "fixed",
        "stage": "offensive_advantage",
        "distance_m": [2800.0, 3600.0],
        "lateral_m": [-300.0, 300.0],
        "altitude_ft": 20000.0,
        "ego_speed_fps": 880.0,
        "enemy_speed_fps": 760.0,
    }
    env = SituationCurriculumWrapper(env, situation, run_name="audit_env0")
    wrapped = env
    config = load_reward_shaping_config(record["config"]) if record["config"] else load_reward_shaping_config(None)
    if record["use_reward_shaping"]:
        wrapped = RewardShapingWrapper(
            env,
            config_path=record["config"],
            enabled_override=True,
            log_dir=os.path.join(args.output_dir, "reward_shaping_logs", record["name"]),
            run_name=record["name"],
        )
    policy, all_args = load_policy(wrapped, record["model_dir"])
    extractor = StateExtractor()
    labeler = TacticalLabeler(config.get("thresholds", {}))
    for episode in range(int(args.episodes)):
        obs = wrapped.reset()
        rnn_states = np.zeros((wrapped.num_agents, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size), dtype=np.float32)
        masks = np.ones((wrapped.num_agents, 1), dtype=np.float32)
        prev_metrics = {}
        prev_scores = {}
        neutral_steps = {}
        lost_tail_step = None
        lost_attack_step = None
        initial_tail_score = None
        initial_attack_score = None
        for step in range(int(args.max_steps)):
            with torch.no_grad():
                action_tensor, rnn_tensor = policy.act(obs, rnn_states, masks, deterministic=True)
            actions = action_tensor.detach().cpu().numpy()
            action_sat = action_saturation(actions, wrapped.action_space)
            obs, rewards, dones, info = wrapped.step(actions)
            detail = None
            if record["use_reward_shaping"]:
                detail = info.get("reward_shaping", {}).get(wrapped.ego_ids[0])
            row = row_from_env(
                env=wrapped,
                extractor=extractor,
                labeler=labeler,
                prev_metrics=prev_metrics,
                prev_scores=prev_scores,
                neutral_steps=neutral_steps,
                method=record["name"],
                episode=episode,
                step=step,
                rewards=rewards,
                detail=detail,
                action_saturation=action_sat,
            )
            rows.append(row)
            if step == 0:
                initial_tail_score = row["ego_tail_advantage_score"]
                initial_attack_score = row["effective_attack_window_score"]
            if lost_tail_step is None and row["ego_tail_advantage_hard"] <= 0.0:
                lost_tail_step = step
            if lost_attack_step is None and row["effective_attack_window_hard"] <= 0.0:
                lost_attack_step = step
            rnn_states = rnn_tensor.detach().cpu().numpy()
            masks = np.ones((wrapped.num_agents, 1), dtype=np.float32)
            masks[np.asarray(dones).reshape(wrapped.num_agents, -1)[:, 0] == True] = 0.0
            if bool(np.all(dones)):
                break
        if (lost_tail_step is not None and lost_tail_step < 20) or (lost_attack_step is not None and lost_attack_step < 20):
            failures.append({
                "method": record["name"],
                "episode": episode,
                "lost_tail_step": lost_tail_step,
                "lost_attack_step": lost_attack_step,
                "initial_tail_score": initial_tail_score,
                "initial_attack_score": initial_attack_score,
                "reason_hint": infer_reason([row for row in rows if row["method"] == record["name"] and row["episode"] == episode]),
            })
    wrapped.close()
    return rows, failures, summarize_method(rows)


def row_from_env(env, extractor, labeler, prev_metrics, prev_scores, neutral_steps, method, episode, step, rewards, detail, action_saturation):
    agent_id = env.ego_ids[0]
    state = extractor.extract(env, agent_id)
    metrics = compute_aircombat_metrics(state)
    labels, count = labeler.evaluate(
        metrics,
        previous_metrics=prev_metrics.get(agent_id),
        previous_label_scores=prev_scores.get(agent_id),
        neutral_steps=neutral_steps.get(agent_id, 0),
    )
    prev_metrics[agent_id] = metrics
    prev_scores[agent_id] = {name: float(labels[name]["score"]) for name in LABEL_NAMES}
    neutral_steps[agent_id] = count
    logged = metrics_for_logging(metrics)
    reward_terms = detail.get("reward_terms", {}) if detail else {}
    verification = detail.get("verification", {}) if detail else {}
    fused = verification.get("fused_scores", {})
    row = {
        "method": method,
        "episode": int(episode),
        "step": int(step),
        "time_sec": float(step * getattr(env, "time_interval", 0.0)),
        "situation_type": getattr(env, "current_situation_type", "offensive_advantage"),
        "env_reward": float(np.asarray(rewards).reshape(env.num_agents, -1)[0, 0]),
        "physical_reward": float(reward_terms.get("physical_reward", 0.0)),
        "semantic_phi": float(reward_terms.get("semantic_phi", 0.0)),
        "semantic_shaping_reward": float(reward_terms.get("semantic_shaping_reward", 0.0)),
        "total_reward": float(reward_terms.get("total_reward", np.asarray(rewards).reshape(env.num_agents, -1)[0, 0])),
        "action_saturation": float(action_saturation),
        **logged,
    }
    for name in LABEL_NAMES:
        row[name + "_hard"] = float(labels[name]["hard"])
        row[name + "_score"] = float(labels[name]["score"])
        row[name + "_fused_score"] = float(fused.get(name, labels[name]["score"]))
    return row


def summarize_method(rows):
    out = {"row_count": len(rows)}
    for key in [
        "ego_tail_advantage_score",
        "effective_attack_window_score",
        "enemy_missile_threat_zone_score",
        "defensive_escape_score",
        "neutral_stalemate_score",
        "semantic_phi",
        "semantic_shaping_reward",
        "total_reward",
        "action_saturation",
    ]:
        out[key] = summarize([row[key] for row in rows])
    for name in ["ego_tail_advantage", "effective_attack_window", "neutral_stalemate", "enemy_missile_threat_zone"]:
        out[name + "_hard_rate"] = float(np.mean([row[name + "_hard"] for row in rows])) if rows else 0.0
    loss_steps = first_loss_steps(rows, "ego_tail_advantage_hard")
    out["tail_loss_step_mean"] = float(np.mean(loss_steps)) if loss_steps else None
    attack_loss_steps = first_loss_steps(rows, "effective_attack_window_hard")
    out["attack_loss_step_mean"] = float(np.mean(attack_loss_steps)) if attack_loss_steps else None
    out["diagnosis"] = diagnose(out)
    return out


def first_loss_steps(rows, key):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    losses = []
    for ep_rows in by_episode.values():
        ep_rows = sorted(ep_rows, key=lambda row: row["step"])
        for row in ep_rows:
            if row[key] <= 0.0:
                losses.append(row["step"])
                break
    return losses


def infer_reason(ep_rows):
    if not ep_rows:
        return "unknown"
    early = ep_rows[: min(50, len(ep_rows))]
    missile = np.mean([row["enemy_missile_threat_zone_score"] for row in early])
    defense = np.mean([row["defensive_escape_score"] for row in early])
    neutral = np.mean([row["neutral_stalemate_score"] for row in early])
    attack = np.mean([row["effective_attack_window_score"] for row in early])
    if neutral > 0.4:
        return "neutral_stalemate dominates after losing attack geometry"
    if missile > attack and missile > 0.05:
        return "enemy_missile_threat_zone penalty may dominate attack reward"
    if defense > attack and defense > 0.02:
        return "defensive_escape reward may encourage separation"
    return "likely action/control failure or attack reward too weak"


def diagnose(summary):
    attack = summary["effective_attack_window_hard_rate"]
    tail = summary["ego_tail_advantage_hard_rate"]
    neutral = summary["neutral_stalemate_hard_rate"]
    missile = summary["enemy_missile_threat_zone_score"]["mean"]
    shaping_std = summary["semantic_shaping_reward"]["std"]
    if tail < 0.1 and attack < 0.05 and neutral > 0.3:
        return "Conservative drift: loses offensive geometry and spends time neutral."
    if tail < 0.1 and missile > 0.05:
        return "Threat penalty may be discouraging close pursuit."
    if shaping_std < 1e-4:
        return "Semantic shaping is nearly flat in offensive episodes."
    return "No single reward-pathology signal dominates."


def load_policy(env, model_dir):
    all_args = get_config().parse_args([])
    policy = PPOPolicy(all_args, env.observation_space, env.action_space, device=torch.device("cpu"))
    actor_path = os.path.join(model_dir, "actor_latest.pt")
    critic_path = os.path.join(model_dir, "critic_latest.pt")
    policy.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    if os.path.exists(critic_path):
        policy.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
    policy.prep_rollout()
    return policy, all_args


def action_saturation(actions, action_space):
    arr = np.asarray(actions)
    if hasattr(action_space, "nvec"):
        nvec = np.asarray(action_space.nvec)
        return float(np.mean(np.logical_or(arr == 0, arr == (nvec - 1))))
    return 0.0


def parse_records(records):
    parsed = []
    for item in records:
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 4:
            raise ValueError("--record must be name,model_dir,config,use_reward_shaping")
        parsed.append({
            "name": parts[0],
            "model_dir": parts[1],
            "config": parts[2] if parts[2] not in ("None", "none", "") else None,
            "use_reward_shaping": parts[3].lower() in ("1", "true", "yes"),
        })
    return parsed


def default_records():
    base = "scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo"
    return [
        {
            "name": "PPO_original_with_curriculum",
            "model_dir": os.path.join(base, "stage11_offensive_original_seed0", "run1"),
            "config": "lag_extensions/reward_shaping/configs/offensive_curriculum_original_1v1.yaml",
            "use_reward_shaping": False,
        },
        {
            "name": "PPO_surrogate_v2_then_fusion_with_curriculum",
            "model_dir": os.path.join(base, "stage11_offensive_surrogate_v2_then_fusion_seed0", "run1"),
            "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_offensive_curriculum_1v1.yaml",
            "use_reward_shaping": True,
        },
        {
            "name": "PPO_surrogate_v2_direct_beta_low_with_curriculum",
            "model_dir": os.path.join(base, "stage11_offensive_surrogate_v2_direct_beta_low_seed0", "run1"),
            "config": "lag_extensions/reward_shaping/configs/surrogate_v2_direct_beta_low_offensive_curriculum_1v1.yaml",
            "use_reward_shaping": True,
        },
    ]


def write_report(path, summaries):
    lines = [
        "# Stage11B Offensive Curriculum Reward Audit",
        "",
        "| method | tail hard | attack hard | neutral hard | missile score mean | phi std | shaping std | diagnosis |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method, summary in summaries.items():
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            method,
            summary.get("ego_tail_advantage_hard_rate", 0.0),
            summary.get("effective_attack_window_hard_rate", 0.0),
            summary.get("neutral_stalemate_hard_rate", 0.0),
            summary.get("enemy_missile_threat_zone_score", {}).get("mean", 0.0),
            summary.get("semantic_phi", {}).get("std", 0.0),
            summary.get("semantic_shaping_reward", {}).get("std", 0.0),
            summary.get("diagnosis", ""),
        ))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
