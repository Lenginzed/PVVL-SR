#!/usr/bin/env python
import argparse
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.tasks.singlecombat_task import PursueAgent
from lag_extensions.reward_shaping.aircombat_metrics import compute_aircombat_metrics, metrics_for_logging
from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.situation_renderer import SituationRenderer
from lag_extensions.reward_shaping.state_extractor import StateExtractor
from lag_extensions.reward_shaping.tactical_labels import TacticalLabeler
from tools.stage11_common import ensure_dir, summarize, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description="Test whether scripted pursuit can produce tail/attack geometry.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--output-dir", default="scripts/results/stage11_scripted_pursuit_geometry")
    args = parser.parse_args()
    if args.env_name != "SingleCombat":
        raise NotImplementedError("Only SingleCombat is supported.")

    out_dir = ensure_dir(args.output_dir)
    config = load_reward_shaping_config(args.config)
    modes = ["enemy_straight", "enemy_weak_turn", "default_opponent", "offensive_init_hold"]
    summaries = {}
    all_rows = []
    for mode in modes:
        rows = run_mode(args, config, mode, os.path.join(out_dir, mode))
        all_rows.extend(rows)
        summaries[mode] = summarize_mode(rows)

    write_json(os.path.join(out_dir, "summary.json"), summaries)
    write_csv(os.path.join(out_dir, "trajectory_metrics.csv"), all_rows)
    write_report(os.path.join(out_dir, "scripted_pursuit_report.md"), summaries)
    try_plot(out_dir, all_rows)
    print(os.path.abspath(os.path.join(out_dir, "scripted_pursuit_report.md")))


def run_mode(args, config, mode, mode_dir):
    ensure_dir(mode_dir)
    env = SingleCombatEnv(args.scenario_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    obs = env.reset()
    if mode == "offensive_init_hold":
        obs = reset_with_curriculum_state(env, "offensive_advantage")

    labeler = TacticalLabeler(config.get("thresholds", {}))
    extractor = StateExtractor()
    pursue_agent = try_make_pursue_agent()
    enemy_agent = try_make_pursue_agent() if mode == "default_opponent" else None
    renderer = SituationRenderer(config.get("renderer", {}), config.get("thresholds", {}))
    prev_metrics = {}
    prev_scores = {}
    neutral_steps = {}
    first_rendered = False
    rows = []
    for step in range(int(args.steps)):
        actions = scripted_actions(env, mode, pursue_agent, enemy_agent)
        obs, rewards, dones, _ = env.step(actions)
        for agent_id in (env.ego_ids + env.enm_ids)[: env.num_agents]:
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
            row = {"mode": mode, "step": step, "agent_id": agent_id, **logged}
            for name in LABEL_NAMES:
                row[name + "_hard"] = float(labels[name]["hard"])
                row[name + "_score"] = float(labels[name]["score"])
            rows.append(row)
            if (not first_rendered) and agent_id == env.ego_ids[0] and (
                labels["ego_tail_advantage"]["hard"] or labels["effective_attack_window"]["hard"] or mode == "offensive_init_hold"
            ):
                renderer.render_sample(state, metrics_for_logging(metrics), labels, mode_dir, sample_id="typical_{}".format(mode))
                first_rendered = True
        if bool(np.all(dones)):
            break
    env.close()
    return rows


def scripted_actions(env, mode, pursue_agent, enemy_agent):
    actions = []
    agent_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    for idx, agent_id in enumerate(agent_ids):
        sim = env.agents[agent_id]
        if idx == 0:
            action = pursuit_discrete_action(sim, pursue_agent)
        elif mode == "enemy_straight" or mode == "offensive_init_hold":
            action = np.array([20, 20, 20, 24], dtype=np.int64)
        elif mode == "enemy_weak_turn":
            action = np.array([25, 20, 24, 24], dtype=np.int64)
        elif mode == "default_opponent":
            action = pursuit_discrete_action(sim, enemy_agent)
        else:
            action = env.action_space.sample()
        actions.append(action)
    return np.asarray(actions)


def try_make_pursue_agent():
    try:
        agent = PursueAgent()
        agent.reset()
        return agent
    except Exception:
        return None


def pursuit_discrete_action(sim, pursue_agent):
    if pursue_agent is not None:
        try:
            norm = np.asarray(pursue_agent.get_action(sim), dtype=np.float64)
            return normalized_to_discrete(norm)
        except Exception:
            pass
    # Fallback: a simple full-throttle action with weak rudder/aileron toward the target side.
    ego_pos = np.asarray(sim.get_position()[:2], dtype=np.float64)
    enemy_pos = np.asarray(sim.enemies[0].get_position()[:2], dtype=np.float64)
    ego_vel = np.asarray(sim.get_velocity()[:2], dtype=np.float64)
    rel = enemy_pos - ego_pos
    side = np.sign(np.cross(ego_vel, rel))
    aileron = 24 if side > 0 else 16 if side < 0 else 20
    rudder = 24 if side > 0 else 16 if side < 0 else 20
    return np.array([aileron, 20, rudder, 29], dtype=np.int64)


def normalized_to_discrete(norm):
    norm = np.asarray(norm, dtype=np.float64)
    if norm.shape[0] < 4:
        return np.array([20, 20, 20, 24], dtype=np.int64)
    return np.array([
        int(np.clip(round((norm[0] + 1.0) * 20.0), 0, 40)),
        int(np.clip(round((norm[1] + 1.0) * 20.0), 0, 40)),
        int(np.clip(round((norm[2] + 1.0) * 20.0), 0, 40)),
        int(np.clip(round((norm[3] - 0.4) * 58.0), 0, 29)),
    ], dtype=np.int64)


def reset_with_curriculum_state(env, situation):
    states = build_curriculum_init_states(situation)
    env.current_step = 0
    for sim, state in zip(env.agents.values(), states):
        sim.reload(state.copy())
    env._tempsims.clear()
    env.task.reset(env)
    return env._pack(env.get_obs())


def build_curriculum_init_states(situation):
    base_lon = 120.0
    base_lat = 60.0
    altitude_ft = 20000
    if situation == "offensive_advantage":
        ego_ne = (0.0, 0.0)
        enemy_ne = (3000.0, 0.0)
        ego_heading = 0.0
        enemy_heading = 0.0
        ego_speed = 900.0
        enemy_speed = 760.0
    else:
        raise ValueError("Unknown situation: {}".format(situation))
    return [
        state_from_ne(base_lon, base_lat, ego_ne, altitude_ft, ego_heading, ego_speed),
        state_from_ne(base_lon, base_lat, enemy_ne, altitude_ft, enemy_heading, enemy_speed),
    ]


def state_from_ne(base_lon, base_lat, ne, altitude_ft, heading_deg, speed_fps):
    north, east = ne
    lat = base_lat + north / 111000.0
    lon = base_lon + east / (111000.0 * max(math.cos(math.radians(base_lat)), 1e-6))
    return {
        "ic_long_gc_deg": lon,
        "ic_lat_geod_deg": lat,
        "ic_h_sl_ft": altitude_ft,
        "ic_psi_true_deg": heading_deg % 360.0,
        "ic_u_fps": speed_fps,
    }


def summarize_mode(rows):
    summary = {"row_count": len(rows)}
    for key in ["distance", "ego_aim_angle_deg", "ego_tail_angle_deg", "distance_rate"]:
        summary[key] = summarize([row[key] for row in rows])
    for name in ["ego_tail_advantage", "effective_attack_window", "enemy_tail_threat", "enemy_missile_threat_zone"]:
        hits = [float(row[name + "_hard"]) for row in rows]
        scores = [float(row[name + "_score"]) for row in rows]
        summary[name + "_time_ratio"] = float(np.mean(hits)) if hits else 0.0
        summary[name + "_first_step"] = first_hit_step(rows, name)
        summary[name + "_score_max"] = float(np.max(scores)) if scores else 0.0
    return summary


def first_hit_step(rows, label):
    for row in rows:
        if row[label + "_hard"] > 0.0:
            return int(row["step"])
    return None


def write_report(path, summaries):
    lines = [
        "# Stage11 Scripted Pursuit Geometry Test",
        "",
        "| mode | rows | tail ratio | attack ratio | first tail | first attack | tail score max | attack score max | min distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in summaries.items():
        lines.append("| {} | {} | {:.6f} | {:.6f} | {} | {} | {:.4f} | {:.4f} | {:.1f} |".format(
            mode,
            summary["row_count"],
            summary["ego_tail_advantage_time_ratio"],
            summary["effective_attack_window_time_ratio"],
            summary["ego_tail_advantage_first_step"],
            summary["effective_attack_window_first_step"],
            summary["ego_tail_advantage_score_max"],
            summary["effective_attack_window_score_max"],
            summary["distance"]["min"],
        ))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def try_plot(out_dir, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode and row["agent_id"].startswith("A")]
        if not mode_rows:
            continue
        plt.figure(figsize=(8, 4))
        plt.plot([row["step"] for row in mode_rows], [row["distance"] for row in mode_rows], label="distance")
        plt.plot([row["step"] for row in mode_rows], [row["ego_tail_angle_deg"] for row in mode_rows], label="ego_tail_deg")
        plt.plot([row["step"] for row in mode_rows], [row["ego_aim_angle_deg"] for row in mode_rows], label="ego_aim_deg")
        plt.legend()
        plt.title(mode)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, mode + "_trajectory.png"), dpi=140)
        plt.close()


if __name__ == "__main__":
    main()
