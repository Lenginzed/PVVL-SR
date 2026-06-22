#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tools.summarize_stage10_results import summarize_reward_log, summarize_train_metrics


EVAL_TYPES = ["default_eval", "offensive_eval", "mixed_eval", "unseen_eval"]
KEYS = [
    "win_rate",
    "loss_rate",
    "draw_rate",
    "timeout_ratio",
    "return_ego_mean",
    "return_mean_agents_mean",
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
    "energy_advantage_time_ratio_mean",
    "energy_disadvantage_time_ratio_mean",
    "neutral_stalemate_time_ratio_mean",
    "average_distance_mean",
    "average_delta_E_mean",
    "average_d_dot_mean",
]


TACTICAL_KEYS = [
    "ego_tail_advantage_time_ratio_mean",
    "effective_attack_window_time_ratio_mean",
    "attack_window_entry_count_mean",
    "attack_window_first_entry_time_mean",
    "min_ego_tail_angle_mean",
    "min_ego_aim_angle_mean",
    "enemy_tail_threat_time_ratio_mean",
    "enemy_missile_threat_zone_time_ratio_mean",
    "defensive_escape_time_ratio_mean",
    "energy_advantage_time_ratio_mean",
    "energy_disadvantage_time_ratio_mean",
    "neutral_stalemate_time_ratio_mean",
    "average_distance_mean",
    "average_delta_E_mean",
    "average_d_dot_mean",
]


def main():
    parser = argparse.ArgumentParser(description="Summarize Stage12 formal PPO results.")
    parser.add_argument("--results-dir", default="scripts/results/stage12_formal_ppo")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = summarize(Path(args.results_dir), Path(args.output_dir) if args.output_dir else Path(args.results_dir))
    print(json.dumps(out, indent=2, sort_keys=True))


def summarize(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(results_dir)
    seed_rows = []
    reward_rows = []
    train_rows = []
    checkpoint_rows = []
    for record in records:
        seed_rows.extend(rows_for_record(record))
        reward_rows.append(reward_row_for_record(record))
        train_rows.extend(train_curve_rows(record))
        checkpoint_rows.extend(checkpoint_curve_rows(record))

    by_eval = {}
    for eval_type in EVAL_TYPES:
        rows = [row for row in seed_rows if row["eval_type"] == eval_type]
        aggregate = aggregate_rows(rows)
        by_eval[eval_type] = aggregate
        write_csv(output_dir / "stage12_{}.csv".format(eval_type), aggregate)
        write_eval_table(output_dir / "stage12_main_{}_table.md".format(eval_type), eval_type, aggregate)

    write_csv(output_dir / "stage12_seed_table.csv", seed_rows)
    write_csv(output_dir / "stage12_reward_distribution_summary.csv", reward_rows)
    write_csv(output_dir / "stage12_training_curve_summary.csv", train_rows + checkpoint_rows)
    write_csv(output_dir / "stage12_tactical_metrics_summary.csv", tactical_rows(by_eval))
    write_ablation_table(output_dir / "stage12_ablation_table.md", by_eval)
    sample_rows = sample_efficiency(checkpoint_rows)
    write_csv(output_dir / "sample_efficiency_table.csv", sample_rows)
    write_sample_efficiency_report(output_dir / "sample_efficiency_report.md", sample_rows)
    write_final_recommendation(output_dir / "stage12_final_recommendation.md", by_eval, reward_rows, sample_rows)
    payload = {
        "records": records,
        "seed_rows": seed_rows,
        "aggregate_by_eval": by_eval,
        "reward_rows": reward_rows,
        "sample_efficiency": sample_rows,
    }
    (output_dir / "stage12_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "default_table": str((output_dir / "stage12_main_default_eval_table.md").resolve()),
        "offensive_table": str((output_dir / "stage12_main_offensive_eval_table.md").resolve()),
        "mixed_table": str((output_dir / "stage12_main_mixed_eval_table.md").resolve()),
        "unseen_table": str((output_dir / "stage12_main_unseen_eval_table.md").resolve()),
        "ablation_table": str((output_dir / "stage12_ablation_table.md").resolve()),
        "final_recommendation": str((output_dir / "stage12_final_recommendation.md").resolve()),
        "summary_json": str((output_dir / "stage12_summary.json").resolve()),
    }


def load_records(results_dir):
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
    return records


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
            "num_env_steps": int(record.get("num_env_steps", 0)),
            "run_dir": record.get("run_dir"),
        }
        for key in KEYS:
            row[key] = summary.get(key)
        row["enemy_threat_exposure"] = max(
            float_or_zero(summary.get("enemy_tail_threat_time_ratio_mean")),
            float_or_zero(summary.get("enemy_missile_threat_zone_time_ratio_mean")),
        )
        row["conservative_stalemate_score"] = (
            float_or_zero(summary.get("draw_rate"))
            + float_or_zero(summary.get("timeout_ratio"))
            + float_or_zero(summary.get("neutral_stalemate_time_ratio_mean"))
            - float_or_zero(summary.get("effective_attack_window_time_ratio_mean"))
            - float_or_zero(summary.get("ego_tail_advantage_time_ratio_mean"))
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
        "num_env_steps": int(record.get("num_env_steps", 0)),
        "run_dir": record.get("run_dir"),
    }
    row.update(summarize_reward_log(trial / "reward_shaping" / "train_env0_steps.jsonl"))
    if record.get("run_dir"):
        row.update(summarize_train_metrics(Path(record["run_dir"]) / "train_metrics.jsonl"))
    return row


def train_curve_rows(record):
    run_dir = record.get("run_dir")
    if not run_dir:
        return []
    path = Path(run_dir) / "train_metrics.jsonl"
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["row_type"] = "train_metric"
            rec["group"] = record["group"]
            rec["seed"] = int(record["seed"])
            rows.append(rec)
    return rows


def checkpoint_curve_rows(record):
    trial = Path(record["stdout"]).parent
    path = trial / "eval_by_checkpoint.jsonl"
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            summary = rec.get("summary", {})
            row = {
                "row_type": "checkpoint_eval",
                "group": record["group"],
                "seed": int(record["seed"]),
                "eval_type": rec.get("eval_type"),
                "update": rec.get("update"),
                "approx_env_steps": rec.get("approx_env_steps"),
            }
            for key in [
                "win_rate",
                "draw_rate",
                "ego_tail_advantage_time_ratio_mean",
                "effective_attack_window_time_ratio_mean",
                "neutral_stalemate_time_ratio_mean",
                "enemy_missile_threat_zone_time_ratio_mean",
            ]:
                row[key] = summary.get(key)
            rows.append(row)
    return rows


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    out = []
    for group, group_rows in sorted(grouped.items()):
        record = {
            "group": group,
            "seed_count": len(group_rows),
            "success_count": sum(1 for row in group_rows if row.get("success")),
        }
        numeric_keys = sorted({
            key
            for row in group_rows
            for key, value in row.items()
            if key not in {"seed", "success"} and finite(value)
        })
        for key in numeric_keys:
            values = [float(row[key]) for row in group_rows if finite(row.get(key))]
            st = stats(values)
            record[key + "_mean"] = st["mean"]
            record[key + "_std"] = st["std"]
        out.append(record)
    return out


def tactical_rows(by_eval):
    rows = []
    for eval_type, aggregates in by_eval.items():
        for row in aggregates:
            out = {"eval_type": eval_type, "group": row["group"], "seed_count": row.get("seed_count")}
            for key in TACTICAL_KEYS + ["enemy_threat_exposure", "conservative_stalemate_score"]:
                out[key + "_mean"] = row.get(key + "_mean")
                out[key + "_std"] = row.get(key + "_std")
            rows.append(out)
    return rows


def sample_efficiency(checkpoint_rows):
    thresholds = [
        ("mixed_win_ge_0.30", "mixed_eval", "win_rate", ">=", 0.30),
        ("offensive_attack_ge_0.08", "offensive_eval", "effective_attack_window_time_ratio_mean", ">=", 0.08),
        ("mixed_neutral_le_0.30", "mixed_eval", "neutral_stalemate_time_ratio_mean", "<=", 0.30),
        ("mixed_enemy_threat_le_0.12", "mixed_eval", "enemy_missile_threat_zone_time_ratio_mean", "<=", 0.12),
    ]
    by_group_seed = defaultdict(list)
    for row in checkpoint_rows:
        by_group_seed[(row["group"], row["seed"])].append(row)
    per_seed = []
    for (group, seed), rows in by_group_seed.items():
        for name, eval_type, metric, op, value in thresholds:
            candidates = sorted([row for row in rows if row.get("eval_type") == eval_type], key=lambda row: int(row.get("approx_env_steps") or 0))
            reached = None
            for row in candidates:
                metric_value = row.get(metric)
                if satisfies(metric_value, op, value):
                    reached = int(row.get("approx_env_steps") or 0)
                    break
            per_seed.append({
                "group": group,
                "seed": seed,
                "threshold": name,
                "eval_type": eval_type,
                "metric": metric,
                "target": value,
                "op": op,
                "reached_step": reached,
                "reached": reached is not None,
            })
    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["group"], row["threshold"])].append(row)
    out = []
    for (group, threshold), rows in sorted(grouped.items()):
        reached = [row["reached_step"] for row in rows if row["reached_step"] is not None]
        out.append({
            "group": group,
            "threshold": threshold,
            "seed_count": len(rows),
            "reached_count": len(reached),
            "mean_reached_step": mean(reached) if reached else None,
            "std_reached_step": std(reached) if reached else None,
            "status": "reached" if reached else "not reached",
        })
    return out


def write_eval_table(path, eval_type, rows):
    cols = [
        "group",
        "seed_count",
        "win_rate",
        "draw_rate",
        "return_ego_mean",
        "ego_tail_advantage_time_ratio_mean",
        "effective_attack_window_time_ratio_mean",
        "enemy_threat_exposure",
        "neutral_stalemate_time_ratio_mean",
        "conservative_stalemate_score",
    ]
    path.write_text("# Stage12 {}\n\n".format(eval_type) + markdown_mean_std_table(rows, cols), encoding="utf-8")


def write_ablation_table(path, by_eval):
    rows = by_eval.get("mixed_eval", [])
    ablation_names = [
        "PPO_original",
        "PPO_physical",
        "PPO_original_with_situation_curriculum",
        "PPO_surrogate_v2_then_fusion_with_situation_curriculum",
        "PPO_surrogate_v2_direct_beta_low_with_situation_curriculum",
        "PPO_surrogate_v2_no_potential_with_situation_curriculum",
    ]
    rows = [row for row in rows if row["group"] in ablation_names]
    cols = [
        "group",
        "seed_count",
        "win_rate",
        "draw_rate",
        "ego_tail_advantage_time_ratio_mean",
        "effective_attack_window_time_ratio_mean",
        "neutral_stalemate_time_ratio_mean",
        "enemy_threat_exposure",
        "conservative_stalemate_score",
    ]
    path.write_text("# Stage12 Ablation Table (mixed_eval)\n\n" + markdown_mean_std_table(rows, cols), encoding="utf-8")


def write_sample_efficiency_report(path, rows):
    cols = ["group", "threshold", "seed_count", "reached_count", "mean_reached_step", "std_reached_step", "status"]
    path.write_text("# Stage12 Sample Efficiency\n\n" + markdown_table(rows, cols), encoding="utf-8")


def write_final_recommendation(path, by_eval, reward_rows, sample_rows):
    mixed = {row["group"]: row for row in by_eval.get("mixed_eval", [])}
    offensive = {row["group"]: row for row in by_eval.get("offensive_eval", [])}
    candidates = [
        "PPO_surrogate_v2_then_fusion_with_situation_curriculum",
        "PPO_surrogate_v2_direct_beta_low_with_situation_curriculum",
    ]
    lines = ["# Stage12 Final Recommendation", ""]
    for group in candidates:
        m = mixed.get(group, {})
        o = offensive.get(group, {})
        lines.append("## {}".format(group))
        lines.append("")
        lines.append("- mixed win: {}".format(fmt_pair(m.get("win_rate_mean"), m.get("win_rate_std"))))
        lines.append("- mixed attack: {}".format(fmt_pair(m.get("effective_attack_window_time_ratio_mean_mean"), m.get("effective_attack_window_time_ratio_mean_std"))))
        lines.append("- mixed neutral: {}".format(fmt_pair(m.get("neutral_stalemate_time_ratio_mean_mean"), m.get("neutral_stalemate_time_ratio_mean_std"))))
        lines.append("- offensive attack: {}".format(fmt_pair(o.get("effective_attack_window_time_ratio_mean_mean"), o.get("effective_attack_window_time_ratio_mean_std"))))
        lines.append("- offensive tail: {}".format(fmt_pair(o.get("ego_tail_advantage_time_ratio_mean_mean"), o.get("ego_tail_advantage_time_ratio_mean_std"))))
        lines.append("")
    lines.append("Interpretation should remain cautious for 3 seeds; prefer the method that improves mixed/offensive tactical metrics without default/unseen collapse.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_mean_std_table(rows, base_cols):
    lines = ["| " + " | ".join(base_cols) + " |", "| " + " | ".join(["---"] * len(base_cols)) + " |"]
    for row in rows:
        cells = []
        for col in base_cols:
            if col in {"group", "seed_count"}:
                cells.append(str(row.get(col, "")))
            else:
                cells.append(fmt_pair(row.get(col + "_mean"), row.get(col + "_std")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def markdown_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col)) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = [float(v) for v in values if finite(v)]
    if not values:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def mean(values):
    values = [float(v) for v in values if finite(v)]
    return float(np.mean(values)) if values else None


def std(values):
    values = [float(v) for v in values if finite(v)]
    return float(np.std(values)) if values else None


def satisfies(value, op, target):
    if not finite(value):
        return False
    value = float(value)
    return value >= float(target) if op == ">=" else value <= float(target)


def finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def float_or_zero(value):
    return float(value) if finite(value) else 0.0


def format_value(value):
    if value is None:
        return "not reached"
    if isinstance(value, bool):
        return str(value)
    if finite(value):
        return "{:.4f}".format(float(value))
    return str(value)


def fmt_pair(mean_value, std_value):
    if not finite(mean_value):
        return ""
    if not finite(std_value):
        return "{:.4f}".format(float(mean_value))
    return "{:.4f} +/- {:.4f}".format(float(mean_value), float(std_value))


if __name__ == "__main__":
    main()

