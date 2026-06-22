#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


EVAL_TYPES = ["default_eval", "offensive_eval", "mixed_eval", "unseen_eval"]
METRICS = [
    "win_rate",
    "draw_rate",
    "ego_tail_advantage_time_ratio_mean",
    "effective_attack_window_time_ratio_mean",
    "enemy_threat_exposure",
    "neutral_stalemate_time_ratio_mean",
    "conservative_stalemate_score",
]


def main():
    parser = argparse.ArgumentParser(description="Compare Stage12 300k and Stage13 500k summaries.")
    parser.add_argument("--stage12-summary", default="scripts/results/stage12_formal_ppo/stage12_summary.json")
    parser.add_argument("--stage13-summary", default="scripts/results/stage13_500k_confirmation/stage13_summary.json")
    parser.add_argument("--output-dir", default="scripts/results/stage13_500k_confirmation")
    args = parser.parse_args()
    out = compare(args)
    print(json.dumps(out, indent=2, sort_keys=True))


def compare(args):
    stage12 = json.loads(Path(args.stage12_summary).read_text(encoding="utf-8"))
    stage13 = json.loads(Path(args.stage13_summary).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for eval_type in EVAL_TYPES:
        old = {row["group"]: row for row in stage12.get("aggregate_by_eval", {}).get(eval_type, [])}
        new = {row["group"]: row for row in stage13.get("aggregate_by_eval", {}).get(eval_type, [])}
        for group in sorted(set(old) & set(new)):
            for metric in METRICS:
                old_mean = old[group].get(metric + "_mean")
                new_mean = new[group].get(metric + "_mean")
                if old_mean is None or new_mean is None:
                    continue
                rows.append({
                    "eval_type": eval_type,
                    "group": group,
                    "metric": metric,
                    "stage12_300k": float(old_mean),
                    "stage13_500k": float(new_mean),
                    "delta_500k_minus_300k": float(new_mean) - float(old_mean),
                })
    write_json(out_dir / "stage13_vs_stage12_comparison.json", rows)
    write_report(out_dir / "stage13_vs_stage12_report.md", rows)
    return {
        "comparison_json": str((out_dir / "stage13_vs_stage12_comparison.json").resolve()),
        "report": str((out_dir / "stage13_vs_stage12_report.md").resolve()),
    }


def write_report(path, rows):
    lines = [
        "# Stage13 500k vs Stage12 300k",
        "",
        "Stage13 continues from Stage12 actor/critic weights for an additional 200k environment steps. PPO optimizer state is not restored because the original runner does not save it.",
        "",
    ]
    for eval_type in EVAL_TYPES:
        lines.extend(["## {}".format(eval_type), ""])
        lines.append("| group | metric | 300k | 500k | delta |")
        lines.append("|---|---|---:|---:|---:|")
        for row in rows:
            if row["eval_type"] != eval_type:
                continue
            lines.append("| {group} | {metric} | {old:.4f} | {new:.4f} | {delta:+.4f} |".format(
                group=row["group"],
                metric=row["metric"],
                old=row["stage12_300k"],
                new=row["stage13_500k"],
                delta=row["delta_500k_minus_300k"],
            ))
        lines.append("")
    lines.extend(["## Diagnostic Reading", ""])
    lines.append("- Positive win/attack/tail deltas are improvements.")
    lines.append("- Positive draw/neutral/conservative-score deltas indicate possible conservatism.")
    lines.append("- Negative enemy-threat deltas indicate safer behavior.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
