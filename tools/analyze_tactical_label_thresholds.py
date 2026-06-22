#!/usr/bin/env python
import argparse
import copy
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from tools.stage11_common import (
    collect_reward_step_logs,
    ensure_dir,
    iter_reward_metrics,
    labels_from_metrics_stream,
    summarize,
    write_csv,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze tactical-label trigger sensitivity to thresholds.")
    parser.add_argument("--stage10-dir", default="scripts/results/stage10_multiseed_ppo")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml")
    parser.add_argument("--groups", nargs="*", default=[
        "PPO_physical",
        "PPO_surrogate_v2_then_fusion",
        "PPO_surrogate_v2_direct_beta_low",
        "PPO_surrogate_v2_direct",
    ])
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--per-log-limit", type=int, default=5000)
    parser.add_argument("--output-dir", default="scripts/results/stage11_threshold_analysis")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    config = load_reward_shaping_config(args.config)
    base_thresholds = config.get("thresholds", {})
    paths = collect_reward_step_logs(args.stage10_dir, args.groups)
    metrics_rows = collect_metrics_balanced(paths, args.limit, args.per_log_limit)
    if not metrics_rows:
        raise RuntimeError("No reward shaping step metrics found. Check --stage10-dir or --groups.")

    variants = build_variants(base_thresholds)
    sweep_rows = []
    label_rows = []
    for variant_name, thresholds in variants:
        labeled = labels_from_metrics_stream(metrics_rows, thresholds)
        row = {"variant": variant_name, "sample_count": len(labeled)}
        for label in LABEL_NAMES:
            hards = [float(item[label + "_hard"]) for item in labeled]
            scores = [float(item[label + "_score"]) for item in labeled]
            row[label + "_trigger_rate"] = float(np.mean(hards)) if hards else 0.0
            score_summary = summarize(scores)
            row[label + "_score_mean"] = score_summary["mean"]
            row[label + "_score_p95"] = score_summary["p95"]
            row[label + "_score_max"] = score_summary["max"]
            label_rows.append({
                "variant": variant_name,
                "label": label,
                "trigger_rate": row[label + "_trigger_rate"],
                "score_mean": row[label + "_score_mean"],
                "score_p95": row[label + "_score_p95"],
                "score_max": row[label + "_score_max"],
            })
        sweep_rows.append(row)

    summary = {
        "sample_count": len(metrics_rows),
        "source_log_count": len(paths),
        "groups": args.groups,
        "base_variant": _compact_variant_row(sweep_rows[0]),
        "recommendation": make_recommendation(sweep_rows),
    }
    write_csv(os.path.join(out_dir, "threshold_sweep.csv"), sweep_rows)
    write_csv(os.path.join(out_dir, "per_label_sweep.csv"), label_rows)
    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_report(os.path.join(out_dir, "threshold_report.md"), summary, sweep_rows)
    print(os.path.abspath(os.path.join(out_dir, "threshold_report.md")))


def build_variants(base):
    variants = [("base", copy.deepcopy(base))]
    for value in [30, 45, 60, 75]:
        cfg = copy.deepcopy(base)
        cfg["theta_tail_deg"] = value
        variants.append(("theta_tail_{}".format(value), cfg))
    for value in [30, 45, 60, 75]:
        cfg = copy.deepcopy(base)
        cfg["theta_aim_deg"] = value
        variants.append(("theta_aim_{}".format(value), cfg))
    for value in [20, 30, 45, 60]:
        cfg = copy.deepcopy(base)
        cfg["theta_launch_deg"] = value
        variants.append(("theta_launch_{}".format(value), cfg))
    for name, lo_mult, hi_mult in [("attack_range_current", 1.0, 1.0), ("attack_range_wide20", 0.8, 1.2), ("attack_range_wide50", 0.5, 1.5)]:
        cfg = copy.deepcopy(base)
        cfg["d_attack_min"] = float(base["d_attack_min"]) * lo_mult
        cfg["d_attack_max"] = float(base["d_attack_max"]) * hi_mult
        variants.append((name, cfg))
    combined = copy.deepcopy(base)
    combined["theta_tail_deg"] = 60
    combined["theta_aim_deg"] = 75
    combined["theta_launch_deg"] = 45
    combined["theta_target_deg"] = 60
    combined["d_attack_min"] = float(base["d_attack_min"]) * 0.8
    combined["d_attack_max"] = float(base["d_attack_max"]) * 1.2
    variants.append(("combined_lenient_geometry", combined))
    return variants


def _compact_variant_row(row):
    keep = {}
    for key in [
        "ego_tail_advantage_trigger_rate",
        "ego_tail_advantage_score_mean",
        "ego_tail_advantage_score_p95",
        "ego_tail_advantage_score_max",
        "effective_attack_window_trigger_rate",
        "effective_attack_window_score_mean",
        "effective_attack_window_score_p95",
        "effective_attack_window_score_max",
        "enemy_tail_threat_trigger_rate",
        "enemy_missile_threat_zone_trigger_rate",
        "neutral_stalemate_trigger_rate",
    ]:
        keep[key] = row.get(key, 0.0)
    return keep


def make_recommendation(rows):
    base = rows[0]
    attack_max = base.get("effective_attack_window_score_max", 0.0)
    tail_max = base.get("ego_tail_advantage_score_max", 0.0)
    attack_hard = base.get("effective_attack_window_trigger_rate", 0.0)
    tail_hard = base.get("ego_tail_advantage_trigger_rate", 0.0)
    combined = next((row for row in rows if row["variant"] == "combined_lenient_geometry"), None)
    notes = []
    if attack_hard == 0.0 and tail_hard == 0.0 and attack_max < 0.05 and tail_max < 0.05:
        notes.append("Base rollout almost never approaches offensive geometry; this is more likely a state-distribution/control issue than a single hard-threshold issue.")
    elif attack_hard == 0.0 or tail_hard == 0.0:
        notes.append("Hard triggers are zero while some soft score exists; inspect threshold margins before changing training.")
    if combined:
        notes.append(
            "Combined lenient trigger rates: tail={:.6f}, attack={:.6f}. If these remain near zero, do not loosen thresholds just to create positives.".format(
                combined.get("ego_tail_advantage_trigger_rate", 0.0),
                combined.get("effective_attack_window_trigger_rate", 0.0),
            )
        )
    return notes


def write_report(path, summary, rows):
    lines = [
        "# Stage11 Tactical Threshold Analysis",
        "",
        "- source_log_count: `{}`".format(summary["source_log_count"]),
        "- sample_count: `{}`".format(summary["sample_count"]),
        "- groups: `{}`".format(", ".join(summary["groups"])),
        "",
        "## Recommendation",
        "",
    ]
    for note in summary["recommendation"]:
        lines.append("- " + note)
    lines.extend([
        "",
        "## Key Variants",
        "",
        "| variant | ego_tail hard | ego_tail max | attack hard | attack max | enemy_threat hard | missile hard | neutral hard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row["variant"] == "base" or row["variant"].startswith("combined") or "wide" in row["variant"]:
            lines.append("| {variant} | {tail:.6f} | {tailmax:.4f} | {attack:.6f} | {attackmax:.4f} | {enemy:.6f} | {missile:.6f} | {neutral:.6f} |".format(
                variant=row["variant"],
                tail=row.get("ego_tail_advantage_trigger_rate", 0.0),
                tailmax=row.get("ego_tail_advantage_score_max", 0.0),
                attack=row.get("effective_attack_window_trigger_rate", 0.0),
                attackmax=row.get("effective_attack_window_score_max", 0.0),
                enemy=row.get("enemy_tail_threat_trigger_rate", 0.0),
                missile=row.get("enemy_missile_threat_zone_trigger_rate", 0.0),
                neutral=row.get("neutral_stalemate_trigger_rate", 0.0),
            ))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def collect_metrics_balanced(paths, global_limit, per_log_limit):
    rows = []
    for path in paths:
        remaining = 0 if not per_log_limit else int(per_log_limit)
        for row in iter_reward_metrics([path], limit=per_log_limit):
            rows.append(row)
            if remaining:
                remaining -= 1
                if remaining <= 0:
                    break
            if global_limit and len(rows) >= int(global_limit):
                return rows
    return rows


if __name__ == "__main__":
    main()
