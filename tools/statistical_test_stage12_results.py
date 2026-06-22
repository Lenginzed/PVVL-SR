#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METHODS = [
    "PPO_original",
    "PPO_original_with_situation_curriculum",
    "PPO_physical",
    "PPO_surrogate_v2_then_fusion_with_situation_curriculum",
    "PPO_surrogate_v2_direct_beta_low_with_situation_curriculum",
    "PPO_surrogate_v2_no_potential_with_situation_curriculum",
]

METRICS = [
    "win_rate",
    "effective_attack_window_time_ratio_mean",
    "ego_tail_advantage_time_ratio_mean",
    "neutral_stalemate_time_ratio_mean",
    "enemy_threat_exposure",
]


def main():
    parser = argparse.ArgumentParser(description="Cautious Stage12 statistics for multiseed PPO results.")
    parser.add_argument("--summary", default="scripts/results/stage12_formal_ppo/stage12_summary.json")
    parser.add_argument("--eval-type", default="mixed_eval")
    parser.add_argument("--baseline", default="PPO_original_with_situation_curriculum")
    parser.add_argument("--output-dir", default="scripts/results/stage12_formal_ppo/statistical_tests")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    rows = run(args)
    print(json.dumps({"rows": rows, "output_dir": str(Path(args.output_dir).resolve())}, indent=2, sort_keys=True))


def run(args):
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    seed_rows = [row for row in summary.get("seed_rows", []) if row.get("eval_type") == args.eval_type]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_rows = [row for row in seed_rows if row.get("group") == args.baseline]
    for method in METHODS:
        method_rows = [row for row in seed_rows if row.get("group") == method]
        if not method_rows:
            continue
        for metric in METRICS:
            x = values(method_rows, metric)
            b = values(baseline_rows, metric)
            diff = [xi - bi for xi, bi in zip(x, b)] if len(x) == len(b) and len(x) > 0 else []
            ci = bootstrap_ci(diff, args.bootstrap) if diff else (None, None)
            rows.append({
                "eval_type": args.eval_type,
                "method": method,
                "baseline": args.baseline,
                "metric": metric,
                "seed_count": len(x),
                "method_mean": mean(x),
                "method_std": std(x),
                "baseline_mean": mean(b),
                "baseline_std": std(b),
                "paired_diff_mean": mean(diff),
                "paired_diff_ci_low": ci[0],
                "paired_diff_ci_high": ci[1],
                "interpretation": interpretation(len(x)),
            })
    write_csv(out_dir / "stage12_statistical_tests.csv", rows)
    write_report(out_dir / "stage12_statistical_tests.md", rows, args)
    return rows


def values(rows, metric):
    out = []
    for row in sorted(rows, key=lambda item: int(item.get("seed", 0))):
        value = row.get(metric)
        if finite(value):
            out.append(float(value))
    return out


def bootstrap_ci(values_, n):
    values_ = [float(v) for v in values_ if finite(v)]
    if not values_:
        return None, None
    rng = np.random.default_rng(0)
    samples = []
    arr = np.asarray(values_, dtype=np.float64)
    for _ in range(int(n)):
        picked = rng.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.mean(picked)))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def write_report(path, rows, args):
    lines = [
        "# Stage12 Statistical Test",
        "",
        "Eval type: `{}`".format(args.eval_type),
        "Baseline: `{}`".format(args.baseline),
        "",
        "With 3 seeds, treat these as descriptive intervals, not strong significance claims.",
        "",
        "| method | metric | mean +/- std | baseline mean +/- std | paired diff | 95% bootstrap CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            row["method"],
            row["metric"],
            fmt_pair(row["method_mean"], row["method_std"]),
            fmt_pair(row["baseline_mean"], row["baseline_std"]),
            fmt(row["paired_diff_mean"]),
            "[{}, {}]".format(fmt(row["paired_diff_ci_low"]), fmt(row["paired_diff_ci_high"])),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)


def mean(values_):
    values_ = [float(v) for v in values_ if finite(v)]
    return float(np.mean(values_)) if values_ else None


def std(values_):
    values_ = [float(v) for v in values_ if finite(v)]
    return float(np.std(values_)) if values_ else None


def finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def fmt(value):
    return "" if value is None or not finite(value) else "{:.4f}".format(float(value))


def fmt_pair(mean_value, std_value):
    if not finite(mean_value):
        return ""
    return "{} +/- {}".format(fmt(mean_value), fmt(std_value))


def interpretation(seed_count):
    if seed_count < 5:
        return "descriptive_only_low_n"
    return "bootstrap_ci_reported"


if __name__ == "__main__":
    main()

