#!/usr/bin/env python
import argparse
import csv
import json
import math
import re
from pathlib import Path


GROUP_TO_METHOD = {
    "PPO_original_with_situation_curriculum": "PPO-Cur",
    "PPO_physical": "PPO-Phys",
    "PPO_surrogate_v2_then_fusion_with_situation_curriculum": "PPO-PVVL-SR",
}


TABLE_SPECS = {
    "table1_mixed_eval_stage13.csv": {
        "eval_type": "mixed_eval",
        "metrics": {
            "win_rate": "win_rate",
            "draw_rate": "draw_rate",
            "return_ego_mean": "return_ego_mean",
            "effective_attack_window_time_ratio_mean": "effective_attack_window_time_ratio_mean",
            "ego_tail_advantage_time_ratio_mean": "ego_tail_advantage_time_ratio_mean",
            "enemy_threat_exposure": "enemy_threat_exposure",
            "neutral_stalemate_time_ratio_mean": "neutral_stalemate_time_ratio_mean",
        },
    },
    "table2_offensive_eval_stage13.csv": {
        "eval_type": "offensive_eval",
        "metrics": {
            "win_rate": "win_rate",
            "effective_attack_window_time_ratio_mean": "effective_attack_window_time_ratio_mean",
            "ego_tail_advantage_time_ratio_mean": "ego_tail_advantage_time_ratio_mean",
            "attack_window_first_entry_time_mean": "attack_window_first_entry_time_mean",
            "neutral_stalemate_time_ratio_mean": "neutral_stalemate_time_ratio_mean",
        },
    },
    "table3_unseen_eval_stage13.csv": {
        "eval_type": "unseen_eval",
        "metrics": {
            "win_rate": "win_rate",
            "effective_attack_window_time_ratio_mean": "effective_attack_window_time_ratio_mean",
            "neutral_stalemate_time_ratio_mean": "neutral_stalemate_time_ratio_mean",
            "enemy_threat_exposure": "enemy_threat_exposure",
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description="Check TAES package result consistency.")
    parser.add_argument("--package-dir", default="scripts/results/taes_paper_package")
    parser.add_argument("--stage13-summary", default="scripts/results/stage13_500k_confirmation/stage13_summary.json")
    parser.add_argument("--stage12-summary", default="scripts/results/stage12_formal_ppo/stage12_summary.json")
    args = parser.parse_args()
    report = check(args)
    print(json.dumps(report, indent=2, sort_keys=True))


def check(args):
    package_dir = Path(args.package_dir)
    stage13 = load_json(args.stage13_summary)
    issues = []
    warnings = []

    check_expected_files(package_dir, issues)
    check_tables(package_dir, stage13, issues)
    check_fig_sources(package_dir, stage13, issues)
    check_nan_inf(package_dir, issues)
    check_text_claims(package_dir, issues, warnings)
    check_dodge(package_dir, issues)
    check_stage_labels(package_dir, issues)

    status = "PASS" if not issues else "FAIL"
    report_path = package_dir / "result_consistency_report.md"
    write_report(report_path, status, issues, warnings)
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "report": str(report_path.resolve()),
    }


def check_expected_files(package_dir, issues):
    expected = [
        "paper_experiment_summary.md",
        "paper_claims_and_limitations.md",
        "experiment_protocol_summary.md",
        "method_ablation_summary.md",
        "reproducibility_checklist.md",
        "taes_experiment_package_report.md",
        "main_results_tables/table1_mixed_eval_stage13.csv",
        "main_results_tables/table2_offensive_eval_stage13.csv",
        "main_results_tables/table3_unseen_eval_stage13.csv",
        "main_results_tables/table4_ablation_study.csv",
        "main_results_tables/table5_dodgemissile_preliminary.csv",
        "figures/fig1_method_architecture.pdf",
        "figures/fig2_label_quality_improvement.pdf",
        "figures/fig3_surrogate_model_accuracy.pdf",
        "figures/fig4_mixed_evaluation_performance.pdf",
        "figures/fig5_300k_to_500k_trend.pdf",
        "figures/fig6_sample_efficiency.pdf",
        "figures/fig7_representative_trajectories.pdf",
        "figures/fig8_dodgemissile_preliminary.pdf",
    ]
    for rel in expected:
        path = package_dir / rel
        if not path.exists():
            issues.append(f"Missing expected file: {rel}")
        elif path.is_file() and path.stat().st_size == 0:
            issues.append(f"Empty expected file: {rel}")


def check_tables(package_dir, stage13, issues):
    inv_map = {v: k for k, v in GROUP_TO_METHOD.items()}
    for filename, spec in TABLE_SPECS.items():
        path = package_dir / "main_results_tables" / filename
        if not path.exists():
            continue
        rows = read_csv(path)
        for row in rows:
            method = row["Method"]
            group = inv_map.get(method)
            if not group:
                issues.append(f"{filename}: unknown method {method}")
                continue
            source = find_group(stage13, spec["eval_type"], group)
            for col, metric in spec["metrics"].items():
                mean, std = parse_mean_std(row.get(col, ""))
                src_mean = metric_mean(source, metric)
                src_std = metric_std(source, metric)
                if not close(mean, src_mean, 5e-4):
                    issues.append(f"{filename}: {method} {col} mean mismatch table={mean} source={src_mean}")
                if not close(std, src_std, 5e-4):
                    issues.append(f"{filename}: {method} {col} std mismatch table={std} source={src_std}")


def check_fig_sources(package_dir, stage13, issues):
    src = package_dir / "figure_sources" / "data" / "fig4_mixed_performance_source.csv"
    if src.exists():
        rows = read_csv(src)
        for row in rows:
            method = row["method"]
            group = {v: k for k, v in GROUP_TO_METHOD.items()}.get(method)
            if not group:
                continue
            source = find_group(stage13, "mixed_eval", group)
            mean = float(row["mean"])
            src_mean = metric_mean(source, row["metric"])
            if not close(mean, src_mean, 5e-8):
                issues.append(f"fig4 source mismatch for {method}/{row['metric']}: {mean} vs {src_mean}")


def check_nan_inf(package_dir, issues):
    for path in package_dir.rglob("*.csv"):
        rows = read_csv(path)
        for i, row in enumerate(rows, start=2):
            for key, value in row.items():
                if isinstance(value, str) and value.strip().lower() in ("nan", "inf", "-inf"):
                    issues.append(f"NaN/Inf literal in {path}: row {i}, column {key}")
                try:
                    num = float(value)
                    if math.isnan(num) or math.isinf(num):
                        issues.append(f"NaN/Inf numeric in {path}: row {i}, column {key}")
                except Exception:
                    pass


def check_text_claims(package_dir, issues, warnings):
    claims = read_text(package_dir / "paper_claims_and_limitations.md")
    report = read_text(package_dir / "taes_experiment_package_report.md")
    ablation = read_text(package_dir / "method_ablation_summary.md")
    protocol = read_text(package_dir / "experiment_protocol_summary.md")
    if "PPO-Phys remains a strong" not in claims:
        issues.append("physical strong baseline is not explicitly stated in claims/limitations.")
    if "not full weapon combat success" not in claims:
        issues.append("DodgeMissile preliminary boundary is not explicit in claims.")
    if "directly controls" not in claims or "Claims to Avoid" not in claims:
        issues.append("Claims-to-avoid section does not mention VLM direct control.")
    if "not uniformly worse" not in ablation:
        issues.append("no_potential conclusion may be over-simplified; expected cautious wording not found.")
    if "optimizer state is reset" not in protocol and "optimizer state is reset" not in report:
        issues.append("Stage13 optimizer-reset caveat missing.")
    if "outperforms all baselines" in (claims + report).lower():
        issues.append("Overclaim phrase detected: outperforms all baselines.")
    if "state-of-the-art" in (claims + report).lower():
        warnings.append("SOTA phrase detected; verify it is in claims-to-avoid only.")


def check_dodge(package_dir, issues):
    table = package_dir / "main_results_tables" / "table5_dodgemissile_preliminary.csv"
    report = read_text(package_dir / "taes_experiment_package_report.md") + read_text(package_dir / "paper_claims_and_limitations.md")
    if "preliminary" not in report.lower():
        issues.append("DodgeMissile not marked as preliminary in package text.")
    if table.exists():
        for row in read_csv(table):
            win_loss = row.get("Win/Loss", "")
            if not win_loss.startswith("0.0000/1.0000"):
                issues.append(f"DodgeMissile row does not show all-loss preliminary result: {row}")


def check_stage_labels(package_dir, issues):
    trend = read_text(package_dir / "figure_captions.md") + read_text(package_dir / "taes_experiment_package_report.md")
    if "300k" not in trend or "500k" not in trend:
        issues.append("300k/500k distinction missing from captions/report.")


def write_report(path, status, issues, warnings):
    lines = ["# TAES Result Consistency Report", "", f"Status: **{status}**", ""]
    lines.append("## Checks")
    lines.extend([
        "",
        "- Main table values traced to Stage13 summary.",
        "- Figure source data checked against table/source summaries.",
        "- NaN/inf literals checked in generated CSV files.",
        "- DodgeMissile preliminary language checked.",
        "- Physical strong baseline and no-potential caution checked.",
        "- 300k/500k source distinction checked.",
    ])
    lines.append("")
    lines.append("## Issues")
    if issues:
        lines.extend([f"- {item}" for item in issues])
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Warnings")
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text(path):
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def find_group(summary, eval_type, group):
    for row in summary.get("aggregate_by_eval", {}).get(eval_type, []):
        if row.get("group") == group:
            return row
    return {}


def metric_mean(row, metric):
    return row.get(metric, row.get(metric + "_mean"))


def metric_std(row, metric):
    return row.get(metric + "_std", 0.0)


def parse_mean_std(text):
    match = re.match(r"\s*([-+0-9.eE]+)\s*\+/-\s*([-+0-9.eE]+)", text or "")
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


if __name__ == "__main__":
    main()
