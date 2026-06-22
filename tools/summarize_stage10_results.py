#!/usr/bin/env python
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


TACTICAL_KEYS = [
    "ego_tail_advantage_time_ratio_mean",
    "enemy_tail_threat_time_ratio_mean",
    "effective_attack_window_time_ratio_mean",
    "enemy_missile_threat_zone_time_ratio_mean",
    "energy_advantage_time_ratio_mean",
    "energy_disadvantage_time_ratio_mean",
    "defensive_escape_time_ratio_mean",
    "neutral_stalemate_time_ratio_mean",
    "average_distance_mean",
    "average_delta_E_mean",
    "average_d_dot_mean",
]

FINAL_KEYS = [
    "win_rate",
    "loss_rate",
    "draw_rate",
    "timeout_ratio",
    "return_ego_mean",
    "return_mean_agents_mean",
    "episode_length_mean",
    "survival_time_mean",
] + TACTICAL_KEYS


def main():
    parser = argparse.ArgumentParser(description="Summarize stage10 multiseed PPO results.")
    parser.add_argument("--results-dir", default="scripts/results/stage10_multiseed_ppo")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output = summarize(Path(args.results_dir), Path(args.output_dir) if args.output_dir else Path(args.results_dir))
    print(json.dumps(output, indent=2, sort_keys=True))


def summarize(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = results_dir / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(run_summary_path)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    rows = []
    training_curve_rows = []
    for record in run_summary.get("results", []):
        row = summarize_record(record)
        rows.append(row)
        training_curve_rows.extend(read_train_metrics(record))
    aggregate = aggregate_rows(rows)
    write_csv(output_dir / "stage10_seed_table.csv", rows)
    write_csv(output_dir / "stage10_summary.csv", aggregate)
    write_csv(output_dir / "stage10_training_curves.csv", training_curve_rows)
    payload = {
        "results_dir": str(results_dir.resolve()),
        "seed_rows": rows,
        "aggregate": aggregate,
    }
    (output_dir / "stage10_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_tables(output_dir, aggregate)
    return {
        "summary_csv": str((output_dir / "stage10_summary.csv").resolve()),
        "summary_json": str((output_dir / "stage10_summary.json").resolve()),
        "seed_table_csv": str((output_dir / "stage10_seed_table.csv").resolve()),
        "training_curves_csv": str((output_dir / "stage10_training_curves.csv").resolve()),
        "main_table_md": str((output_dir / "stage10_main_table.md").resolve()),
        "ablation_table_md": str((output_dir / "stage10_ablation_table.md").resolve()),
    }


def summarize_record(record):
    group = record["group"]
    seed = record["seed"]
    trial = Path(record["stdout"]).parent
    final_summary_path = trial / "final_eval" / "summary.json"
    final = json.loads(final_summary_path.read_text(encoding="utf-8")) if final_summary_path.exists() else {}
    reward_stats = summarize_reward_log(trial / "reward_shaping" / "train_env0_steps.jsonl")
    train_stats = summarize_train_metrics(Path(record["run_dir"]) / "train_metrics.jsonl") if record.get("run_dir") else {}
    row = {
        "group": group,
        "seed": seed,
        "success": record.get("success", False),
        "duration_sec": record.get("duration_sec"),
        "run_dir": record.get("run_dir"),
    }
    for key in FINAL_KEYS:
        row[key] = final.get(key)
    row.update(reward_stats)
    row.update(train_stats)
    return row


def summarize_reward_log(path):
    if not path.exists():
        return {
            "reward_log_count": 0,
            "reward_nan_inf_count": None,
            "reward_spike_abs_gt_10": None,
        }
    terms = defaultdict(list)
    fused = defaultdict(list)
    semantic = defaultdict(list)
    sources = Counter()
    modes = Counter()
    nan_inf = 0
    spikes = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rt = rec.get("reward_terms", {})
            for key in ["env_reward", "physical_reward", "semantic_phi", "semantic_shaping_reward", "total_reward"]:
                value = rt.get(key)
                if value is None:
                    continue
                try:
                    value = float(value)
                    terms[key].append(value)
                    if not math.isfinite(value):
                        nan_inf += 1
                    if key == "semantic_shaping_reward" and abs(value) > 10:
                        spikes += 1
                except Exception:
                    nan_inf += 1
            info = rec.get("semantic_info", {})
            sources[str(info.get("source"))] += 1
            modes[str(info.get("surrogate_output_mode"))] += 1
            ver = rec.get("verification", {})
            for label, value in ver.get("fused_scores", {}).items():
                fused[label].append(float(value))
            for label, value in ver.get("semantic_scores", {}).items():
                semantic[label].append(float(value))
    out = {
        "reward_log_count": len(terms.get("env_reward", [])),
        "reward_nan_inf_count": nan_inf,
        "reward_spike_abs_gt_10": spikes,
        "semantic_source_counts": json.dumps(dict(sources), sort_keys=True),
        "surrogate_output_mode_counts": json.dumps(dict(modes), sort_keys=True),
    }
    for key, values in terms.items():
        st = stats(values)
        out["{}_mean".format(key)] = st["mean"]
        out["{}_std".format(key)] = st["std"]
        out["{}_min".format(key)] = st["min"]
        out["{}_max".format(key)] = st["max"]
    for label, values in fused.items():
        out["fused_{}_mean".format(label)] = stats(values)["mean"]
    for label, values in semantic.items():
        out["semantic_{}_mean".format(label)] = stats(values)["mean"]
    return out


def summarize_train_metrics(path):
    if not path.exists():
        return {}
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    out = {"train_metric_updates": len(rows)}
    if not rows:
        return out
    for key in ["policy_loss", "value_loss", "policy_entropy_loss", "actor_grad_norm", "critic_grad_norm", "ratio", "average_episode_rewards", "episodes_in_buffer"]:
        values = [row.get(key) for row in rows if row.get(key) is not None and _finite(row.get(key))]
        if values:
            st = stats([float(v) for v in values])
            out["train_{}_mean".format(key)] = st["mean"]
            out["train_{}_last".format(key)] = float(values[-1])
    return out


def read_train_metrics(record):
    run_dir = record.get("run_dir")
    if not run_dir:
        return []
    path = Path(run_dir) / "train_metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["group"] = record["group"]
            rec["seed"] = record["seed"]
            rows.append(rec)
    return rows


def aggregate_rows(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    aggregate = []
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and key not in {"seed"}})
    for group, group_rows in sorted(by_group.items()):
        out = {"group": group, "seed_count": len(group_rows)}
        out["success_count"] = sum(1 for row in group_rows if row.get("success"))
        for key in numeric_keys:
            values = [row.get(key) for row in group_rows if row.get(key) is not None and _finite(row.get(key))]
            if values:
                st = stats([float(v) for v in values])
                out[key + "_mean"] = st["mean"]
                out[key + "_std"] = st["std"]
        aggregate.append(out)
    return aggregate


def write_markdown_tables(output_dir, aggregate):
    main_cols = [
        "group", "seed_count", "success_count", "win_rate_mean", "win_rate_std", "draw_rate_mean", "draw_rate_std",
        "return_ego_mean_mean", "episode_length_mean_mean", "survival_time_mean_mean",
        "ego_tail_advantage_time_ratio_mean_mean", "effective_attack_window_time_ratio_mean_mean",
        "enemy_missile_threat_zone_time_ratio_mean_mean", "neutral_stalemate_time_ratio_mean_mean",
        "semantic_shaping_reward_mean_mean", "semantic_shaping_reward_std_mean",
    ]
    ablation_cols = [
        "group", "seed_count", "win_rate_mean", "draw_rate_mean", "timeout_ratio_mean",
        "semantic_phi_mean_mean", "semantic_shaping_reward_mean_mean", "semantic_shaping_reward_std_mean",
        "reward_nan_inf_count_mean", "reward_spike_abs_gt_10_mean",
    ]
    (output_dir / "stage10_main_table.md").write_text(markdown_table(aggregate, main_cols), encoding="utf-8")
    (output_dir / "stage10_ablation_table.md").write_text(markdown_table(aggregate, ablation_cols), encoding="utf-8")


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
    values = [float(v) for v in values if _finite(v)]
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(sum(values) / len(values)),
        "std": float(np_std(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def np_std(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return str(value)


if __name__ == "__main__":
    main()
