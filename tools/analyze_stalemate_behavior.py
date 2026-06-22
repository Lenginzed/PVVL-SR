#!/usr/bin/env python
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Analyze conservative stalemate behavior in stage10 results.")
    parser.add_argument("--results-dir", default="scripts/results/stage10_multiseed_ppo")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output = analyze(Path(args.results_dir), Path(args.output_dir) if args.output_dir else Path(args.results_dir) / "stalemate_analysis")
    print(json.dumps(output, indent=2, sort_keys=True))


def analyze(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = results_dir / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(run_summary_path)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    seed_rows = []
    for record in run_summary.get("results", []):
        trial = Path(record["stdout"]).parent
        summary_path = trial / "final_eval" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = make_row(record["group"], record["seed"], summary)
        seed_rows.append(row)
    aggregate_rows = aggregate(seed_rows)
    write_csv(output_dir / "stalemate_seed_table.csv", seed_rows)
    write_csv(output_dir / "stalemate_summary.csv", aggregate_rows)
    payload = {"seed_rows": seed_rows, "aggregate": aggregate_rows}
    (output_dir / "stalemate_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "stalemate_report.md").write_text(markdown_report(aggregate_rows), encoding="utf-8")
    return {
        "seed_table": str((output_dir / "stalemate_seed_table.csv").resolve()),
        "summary_csv": str((output_dir / "stalemate_summary.csv").resolve()),
        "summary_json": str((output_dir / "stalemate_summary.json").resolve()),
        "report": str((output_dir / "stalemate_report.md").resolve()),
    }


def make_row(group, seed, summary):
    draw = float(summary.get("draw_rate", 0.0))
    timeout = float(summary.get("timeout_ratio", 0.0))
    neutral = float(summary.get("neutral_stalemate_time_ratio_mean", 0.0))
    attack = float(summary.get("effective_attack_window_time_ratio_mean", 0.0))
    tail = float(summary.get("ego_tail_advantage_time_ratio_mean", 0.0))
    conservative_score = draw + timeout + neutral - attack - tail
    return {
        "group": group,
        "seed": int(seed),
        "draw_rate": draw,
        "timeout_ratio": timeout,
        "neutral_stalemate_time_ratio": neutral,
        "effective_attack_window_time_ratio": attack,
        "ego_tail_advantage_time_ratio": tail,
        "enemy_missile_threat_zone_time_ratio": float(summary.get("enemy_missile_threat_zone_time_ratio_mean", 0.0)),
        "average_distance": float(summary.get("average_distance_mean", 0.0)),
        "average_delta_E": float(summary.get("average_delta_E_mean", 0.0)),
        "survival_time": float(summary.get("survival_time_mean", 0.0)),
        "win_rate": float(summary.get("win_rate", 0.0)),
        "loss_rate": float(summary.get("loss_rate", 0.0)),
        "conservative_stalemate_score": conservative_score,
        "flag_conservative": bool(conservative_score > 0.8 and attack < 0.02 and tail < 0.02),
    }


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    out = []
    for group, group_rows in sorted(grouped.items()):
        record = {"group": group, "seed_count": len(group_rows)}
        record["flagged_seed_count"] = sum(1 for row in group_rows if row["flag_conservative"])
        for key in [
            "draw_rate",
            "timeout_ratio",
            "neutral_stalemate_time_ratio",
            "effective_attack_window_time_ratio",
            "ego_tail_advantage_time_ratio",
            "enemy_missile_threat_zone_time_ratio",
            "average_distance",
            "average_delta_E",
            "survival_time",
            "win_rate",
            "loss_rate",
            "conservative_stalemate_score",
        ]:
            values = [float(row[key]) for row in group_rows]
            record[key + "_mean"] = mean(values)
            record[key + "_std"] = std(values)
        out.append(record)
    return out


def markdown_report(rows):
    cols = [
        "group", "seed_count", "flagged_seed_count", "conservative_stalemate_score_mean",
        "draw_rate_mean", "timeout_ratio_mean", "neutral_stalemate_time_ratio_mean",
        "effective_attack_window_time_ratio_mean", "ego_tail_advantage_time_ratio_mean",
        "average_distance_mean", "survival_time_mean",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
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


def mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def std(values):
    if not values:
        return 0.0
    m = mean(values)
    return float(math.sqrt(sum((v - m) ** 2 for v in values) / len(values)))


def fmt(value):
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return str(value)


if __name__ == "__main__":
    main()
