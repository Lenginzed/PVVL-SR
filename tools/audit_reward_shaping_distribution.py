#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv

from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.logger import to_jsonable
from lag_extensions.reward_shaping.reward_wrapper import RewardShapingWrapper


def main():
    parser = argparse.ArgumentParser(description="Audit reward shaping distribution without PPO updates.")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollout-steps", type=int, default=5000)
    parser.add_argument("--policy", choices=["random"], default="random")
    parser.add_argument("--output-dir", default="scripts/results/stage9_reward_audit_surrogate_v1_random5000")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    summary = audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def audit(args):
    if args.env_name != "SingleCombat":
        raise NotImplementedError("Only SingleCombat audit is implemented.")
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_reward_shaping_config(args.config, enabled_override=True)
    env = SingleCombatEnv(args.scenario_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    wrapped = RewardShapingWrapper(
        env,
        config_path=args.config,
        enabled_override=True,
        log_dir=os.path.join(args.output_dir, "reward_shaping_logs"),
        run_name="audit",
    )
    wrapped.reset()

    rows = []
    for step in range(int(args.rollout_steps)):
        actions = np.array([wrapped.action_space.sample() for _ in range(wrapped.num_agents)])
        _, _, dones, info = wrapped.step(actions)
        shaping = info.get("reward_shaping", {})
        for agent_id, detail in shaping.items():
            row = _row_from_detail(step, agent_id, detail)
            rows.append(row)
        if np.all(dones):
            wrapped.reset()
    wrapped.close()

    step_csv = os.path.join(args.output_dir, "step_rewards.csv")
    _write_csv(step_csv, rows)
    per_label_rows = _per_label_distribution(rows)
    _write_csv(os.path.join(args.output_dir, "per_label_distribution.csv"), per_label_rows)
    summary = _summary(rows, config)
    summary.update({
        "env_name": args.env_name,
        "scenario_name": args.scenario_name,
        "config": os.path.abspath(args.config),
        "policy": args.policy,
        "rollout_steps": int(args.rollout_steps),
        "num_records": len(rows),
        "step_rewards_csv": os.path.abspath(step_csv),
        "per_label_distribution_csv": os.path.abspath(os.path.join(args.output_dir, "per_label_distribution.csv")),
        "output_dir": os.path.abspath(args.output_dir),
    })
    _write_json(os.path.join(args.output_dir, "summary.json"), summary)
    _try_histograms(args.output_dir, rows)
    return summary


def _row_from_detail(step, agent_id, detail):
    rewards = detail["reward_terms"]
    verification = detail.get("verification", {})
    semantic = verification.get("semantic_scores", {})
    fused = verification.get("fused_scores", verification.get("verified_scores", {}))
    physical = verification.get("physical_scores", {})
    row = {
        "rollout_step": int(step),
        "agent_id": agent_id,
        "global_step": detail.get("global_step"),
        "env_reward": rewards.get("env_reward", 0.0),
        "physical_reward": rewards.get("physical_reward", 0.0),
        "semantic_phi": rewards.get("semantic_phi", 0.0),
        "semantic_shaping_reward": rewards.get("semantic_shaping_reward", 0.0),
        "total_reward": rewards.get("total_reward", 0.0),
        "alpha": rewards.get("alpha", 0.0),
        "beta": rewards.get("beta", 0.0),
        "semantic_source": detail.get("semantic_info", {}).get("source"),
        "semantic_inference_time_sec": detail.get("semantic_info", {}).get("inference_time_sec"),
        "distance": detail.get("metrics", {}).get("distance"),
        "energy_difference": detail.get("metrics", {}).get("energy_difference"),
    }
    env_abs = abs(float(row["env_reward"]))
    row["abs_semantic_shaping_over_abs_env"] = abs(float(row["semantic_shaping_reward"])) / max(env_abs, 1e-8)
    row["nan_or_inf"] = int(any(not np.isfinite(float(row[key])) for key in [
        "env_reward", "physical_reward", "semantic_phi", "semantic_shaping_reward", "total_reward"
    ]))
    for name in LABEL_NAMES:
        row[name + "_semantic_score"] = float(semantic.get(name, 0.0))
        row[name + "_fused_score"] = float(fused.get(name, 0.0))
        row[name + "_physical_score"] = float(physical.get(name, 0.0))
    return row


def _summary(rows, config):
    keys = ["env_reward", "physical_reward", "semantic_phi", "semantic_shaping_reward", "total_reward", "abs_semantic_shaping_over_abs_env"]
    summary = {key: _stats([row[key] for row in rows]) for key in keys}
    values = [row["semantic_shaping_reward"] for row in rows]
    summary["semantic_reward_spike_count_abs_gt_10"] = int(sum(abs(x) > 10.0 for x in values))
    summary["semantic_reward_spike_count_abs_gt_100"] = int(sum(abs(x) > 100.0 for x in values))
    summary["nan_or_inf_count"] = int(sum(row["nan_or_inf"] for row in rows))
    summary["alpha_values"] = sorted(set(float(row["alpha"]) for row in rows))
    summary["beta_values"] = sorted(set(float(row["beta"]) for row in rows))
    summary["semantic_sources"] = dict(_counts(row["semantic_source"] for row in rows))
    saturated = 0
    total_scores = 0
    for row in rows:
        for name in LABEL_NAMES:
            score = float(row[name + "_fused_score"])
            saturated += int(score <= 1e-4 or score >= 1.0 - 1e-4)
            total_scores += 1
    summary["fused_score_saturation_ratio"] = saturated / total_scores if total_scores else 0.0
    summary["diagnosis"] = _diagnosis(summary, config)
    return summary


def _diagnosis(summary, config):
    ratio_p95 = summary["abs_semantic_shaping_over_abs_env"]["p95"]
    if summary["nan_or_inf_count"] > 0:
        return "FAIL: NaN or inf rewards detected."
    if ratio_p95 > 20.0:
        return "WARN: semantic shaping reward is much larger than env reward; reduce beta or label weights."
    if summary["semantic_reward_spike_count_abs_gt_100"] > 0:
        return "WARN: extreme semantic reward spikes detected."
    if summary["semantic_phi"]["std"] < 1e-6:
        return "WARN: semantic_phi is nearly constant; surrogate may provide weak training signal."
    return "PASS: reward magnitudes look numerically stable for this rollout."


def _per_label_distribution(rows):
    output = []
    for name in LABEL_NAMES:
        for kind in ["semantic", "fused", "physical"]:
            values = [row[name + "_" + kind + "_score"] for row in rows]
            item = {"label": name, "score_type": kind}
            item.update(_stats(values))
            output.append(item)
    return output


def _stats(values):
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _counts(values):
    counts = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return counts


def _try_histograms(output_dir, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    keys = ["env_reward", "physical_reward", "semantic_phi", "semantic_shaping_reward", "total_reward"]
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 12))
    for ax, key in zip(axes, keys):
        ax.hist([row[key] for row in rows], bins=50)
        ax.set_title(key)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_histograms.png"), dpi=140)
    plt.close(fig)


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
