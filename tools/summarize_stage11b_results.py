#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tools.summarize_stage10_results import summarize_reward_log, summarize_train_metrics


EVAL_TYPES = ["default_eval", "offensive_eval", "mixed_eval"]
KEYS = [
    "win_rate",
    "loss_rate",
    "draw_rate",
    "timeout_ratio",
    "return_ego_mean",
    "episode_length_mean",
    "survival_time_mean",
    "ego_tail_advantage_time_ratio_mean",
    "effective_attack_window_time_ratio_mean",
    "attack_window_entry_count_mean",
    "attack_window_first_entry_time_mean",
    "min_ego_tail_angle_mean",
    "min_ego_aim_angle_mean",
    "enemy_tail_threat_time_ratio_mean",
    "enemy_missile_threat_zone_time_ratio_mean",
    "defensive_escape_time_ratio_mean",
    "neutral_stalemate_time_ratio_mean",
    "average_distance_mean",
    "average_delta_E_mean",
    "average_d_dot_mean",
]


def main():
    parser = argparse.ArgumentParser(description="Summarize stage11B medium PPO results.")
    parser.add_argument("--results-dir", default="scripts/results/stage11b_medium_ppo")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = summarize(Path(args.results_dir), Path(args.output_dir) if args.output_dir else Path(args.results_dir))
    print(json.dumps(out, indent=2, sort_keys=True))


def summarize(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = load_all_run_records(results_dir)
    seed_rows = []
    reward_rows = []
    for record in run_summary.get("results", []):
        seed_rows.extend(rows_for_record(record))
        reward_rows.append(reward_row_for_record(record))
    by_eval = {}
    for eval_type in EVAL_TYPES:
        rows = [row for row in seed_rows if row["eval_type"] == eval_type]
        aggregate = aggregate_rows(rows)
        by_eval[eval_type] = aggregate
        write_csv(output_dir / "stage11b_{}.csv".format(eval_type.replace("_eval", "_eval")), aggregate)
    write_csv(output_dir / "stage11b_seed_table.csv", seed_rows)
    write_csv(output_dir / "stage11b_reward_summary.csv", reward_rows)
    write_main_table(output_dir / "stage11b_main_table.md", by_eval)
    write_stalemate_report(output_dir / "stage11b_stalemate_report.md", by_eval)
    write_offensive_report(output_dir / "stage11b_offensive_retention_report.md", by_eval)
    write_reward_report(output_dir / "stage11b_reward_audit_report.md", reward_rows)
    payload = {"seed_rows": seed_rows, "aggregate_by_eval": by_eval, "reward_rows": reward_rows}
    (output_dir / "stage11b_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "main_table": str((output_dir / "stage11b_main_table.md").resolve()),
        "default_eval_csv": str((output_dir / "stage11b_default_eval.csv").resolve()),
        "offensive_eval_csv": str((output_dir / "stage11b_offensive_eval.csv").resolve()),
        "mixed_eval_csv": str((output_dir / "stage11b_mixed_eval.csv").resolve()),
        "summary_json": str((output_dir / "stage11b_summary.json").resolve()),
    }


def load_all_run_records(results_dir):
    records = []
    seen = set()
    summary_path = results_dir / "run_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for record in payload.get("results", []):
            key = (record.get("group"), int(record.get("seed", -1)))
            seen.add(key)
            records.append(record)
    for path in sorted(results_dir.glob("*/*/run_record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record.get("group"), int(record.get("seed", -1)))
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return {"results": records}


def rows_for_record(record):
    rows = []
    trial = Path(record["stdout"]).parent
    for eval_type in EVAL_TYPES:
        summary_path = trial / eval_type / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "group": record["group"],
            "seed": int(record["seed"]),
            "eval_type": eval_type,
            "success": bool(record.get("success")),
            "run_dir": record.get("run_dir"),
        }
        for key in KEYS:
            row[key] = summary.get(key)
        row["conservative_stalemate_score"] = (
            float(summary.get("draw_rate", 0.0))
            + float(summary.get("timeout_ratio", 0.0))
            + float(summary.get("neutral_stalemate_time_ratio_mean", 0.0))
            - float(summary.get("effective_attack_window_time_ratio_mean", 0.0))
            - float(summary.get("ego_tail_advantage_time_ratio_mean", 0.0))
        )
        rows.append(row)
    return rows


def reward_row_for_record(record):
    trial = Path(record["stdout"]).parent
    row = {
        "group": record["group"],
        "seed": int(record["seed"]),
        "success": bool(record.get("success")),
        "duration_sec": record.get("duration_sec"),
        "run_dir": record.get("run_dir"),
    }
    reward_path = trial / "reward_shaping" / "train_env0_steps.jsonl"
    row.update(summarize_reward_log(reward_path))
    if record.get("run_dir"):
        row.update(summarize_train_metrics(Path(record["run_dir"]) / "train_metrics.jsonl"))
    return row


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    out = []
    for group, group_rows in sorted(grouped.items()):
        record = {"group": group, "seed_count": len(group_rows), "success_count": sum(1 for row in group_rows if row.get("success"))}
        for key in KEYS + ["conservative_stalemate_score"]:
            values = [row.get(key) for row in group_rows if finite(row.get(key))]
            if values:
                record[key + "_mean"] = mean(values)
                record[key + "_std"] = std(values)
        out.append(record)
    return out


def write_main_table(path, by_eval):
    cols = [
        "group",
        "seed_count",
        "win_rate_mean",
        "draw_rate_mean",
        "ego_tail_advantage_time_ratio_mean_mean",
        "effective_attack_window_time_ratio_mean_mean",
        "enemy_missile_threat_zone_time_ratio_mean_mean",
        "neutral_stalemate_time_ratio_mean_mean",
        "conservative_stalemate_score_mean",
    ]
    lines = ["# Stage11B Main Table", ""]
    for eval_type in EVAL_TYPES:
        lines.extend(["## {}".format(eval_type), "", markdown_table(by_eval.get(eval_type, []), cols), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stalemate_report(path, by_eval):
    rows = by_eval.get("default_eval", []) + by_eval.get("mixed_eval", [])
    rows = sorted(rows, key=lambda row: row.get("conservative_stalemate_score_mean", 0.0), reverse=True)
    cols = ["group", "seed_count", "draw_rate_mean", "neutral_stalemate_time_ratio_mean_mean", "effective_attack_window_time_ratio_mean_mean", "ego_tail_advantage_time_ratio_mean_mean", "conservative_stalemate_score_mean"]
    path.write_text("# Stage11B Stalemate Report\n\n" + markdown_table(rows, cols), encoding="utf-8")


def write_offensive_report(path, by_eval):
    rows = by_eval.get("offensive_eval", [])
    rows = sorted(rows, key=lambda row: (row.get("effective_attack_window_time_ratio_mean_mean", 0.0), row.get("ego_tail_advantage_time_ratio_mean_mean", 0.0)), reverse=True)
    cols = ["group", "seed_count", "win_rate_mean", "draw_rate_mean", "ego_tail_advantage_time_ratio_mean_mean", "effective_attack_window_time_ratio_mean_mean", "attack_window_entry_count_mean_mean", "min_ego_tail_angle_mean_mean", "min_ego_aim_angle_mean_mean", "neutral_stalemate_time_ratio_mean_mean"]
    path.write_text("# Stage11B Offensive Retention Report\n\n" + markdown_table(rows, cols), encoding="utf-8")


def write_reward_report(path, rows):
    cols = ["group", "seed", "reward_log_count", "reward_nan_inf_count", "reward_spike_abs_gt_10", "semantic_phi_mean", "semantic_phi_std", "semantic_shaping_reward_mean", "semantic_shaping_reward_std", "total_reward_mean", "total_reward_std"]
    path.write_text("# Stage11B Reward Audit Report\n\n" + markdown_table(rows, cols), encoding="utf-8")


def markdown_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def mean(values):
    values = [float(v) for v in values if finite(v)]
    return float(sum(values) / len(values)) if values else None


def std(values):
    values = [float(v) for v in values if finite(v)]
    if not values:
        return None
    m = mean(values)
    return float(math.sqrt(sum((v - m) ** 2 for v in values) / len(values)))


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return str(value)


if __name__ == "__main__":
    main()
