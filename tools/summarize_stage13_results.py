#!/usr/bin/env python
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tools.summarize_stage12_formal_results import summarize as summarize_stage12_style
from tools.compare_stage12_stage13_results import compare as compare_stage12_stage13


def main():
    parser = argparse.ArgumentParser(description="Summarize Stage13 500k confirmation results.")
    parser.add_argument("--results-dir", default="scripts/results/stage13_500k_confirmation")
    parser.add_argument("--stage12-summary", default="scripts/results/stage12_formal_ppo/stage12_summary.json")
    args = parser.parse_args()
    out = summarize(args)
    print(json.dumps(out, indent=2, sort_keys=True))


def summarize(args):
    results_dir = Path(args.results_dir)
    out = summarize_stage12_style(results_dir, results_dir)
    rename_outputs(results_dir)
    compare_args = argparse.Namespace(
        stage12_summary=args.stage12_summary,
        stage13_summary=str(results_dir / "stage13_summary.json"),
        output_dir=str(results_dir),
    )
    compare_out = compare_stage12_stage13(compare_args)
    write_claims(results_dir)
    write_risks(results_dir)
    write_final_recommendation(results_dir)
    return {
        "default_table": str((results_dir / "stage13_default_eval_table.md").resolve()),
        "offensive_table": str((results_dir / "stage13_offensive_eval_table.md").resolve()),
        "mixed_table": str((results_dir / "stage13_mixed_eval_table.md").resolve()),
        "unseen_table": str((results_dir / "stage13_unseen_eval_table.md").resolve()),
        "vs_stage12_report": compare_out["report"],
        "claim_candidates": str((results_dir / "stage13_main_claim_candidates.md").resolve()),
        "risk_report": str((results_dir / "stage13_risk_report.md").resolve()),
    }


def rename_outputs(results_dir):
    mapping = {
        "stage12_main_default_eval_table.md": "stage13_default_eval_table.md",
        "stage12_main_offensive_eval_table.md": "stage13_offensive_eval_table.md",
        "stage12_main_mixed_eval_table.md": "stage13_mixed_eval_table.md",
        "stage12_main_unseen_eval_table.md": "stage13_unseen_eval_table.md",
        "stage12_summary.json": "stage13_summary.json",
        "stage12_reward_distribution_summary.csv": "stage13_reward_distribution_summary.csv",
        "stage12_tactical_metrics_summary.csv": "stage13_tactical_metrics_summary.csv",
        "stage12_training_curve_summary.csv": "stage13_training_curve_summary.csv",
        "stage12_ablation_table.md": "stage13_ablation_table.md",
        "stage12_final_recommendation.md": "stage13_final_recommendation.md",
        "sample_efficiency_report.md": "stage13_sample_efficiency_report.md",
        "sample_efficiency_table.csv": "stage13_sample_efficiency_table.csv",
    }
    for src, dst in mapping.items():
        src_path = results_dir / src
        if src_path.exists():
            shutil.copyfile(src_path, results_dir / dst)
            dst_path = results_dir / dst
            if dst_path.suffix.lower() == ".md":
                text = dst_path.read_text(encoding="utf-8")
                dst_path.write_text(text.replace("# Stage12", "# Stage13"), encoding="utf-8")


def load_summary(results_dir):
    return json.loads((results_dir / "stage13_summary.json").read_text(encoding="utf-8"))


def metric(summary, eval_type, group, key):
    for row in summary.get("aggregate_by_eval", {}).get(eval_type, []):
        if row.get("group") == group:
            return row.get(key + "_mean")
    return None


def write_claims(results_dir):
    summary = load_summary(results_dir)
    main = "PPO_surrogate_v2_then_fusion_with_situation_curriculum"
    baseline = "PPO_original_with_situation_curriculum"
    physical = "PPO_physical"
    mixed_win = metric(summary, "mixed_eval", main, "win_rate")
    mixed_threat = metric(summary, "mixed_eval", main, "enemy_threat_exposure")
    mixed_attack = metric(summary, "mixed_eval", main, "effective_attack_window_time_ratio_mean")
    unseen_win = metric(summary, "unseen_eval", main, "win_rate")
    unseen_threat = metric(summary, "unseen_eval", main, "enemy_threat_exposure")
    base_win = metric(summary, "mixed_eval", baseline, "win_rate")
    phys_win = metric(summary, "mixed_eval", physical, "win_rate")
    phys_threat = metric(summary, "mixed_eval", physical, "enemy_threat_exposure")
    phys_unseen_win = metric(summary, "unseen_eval", physical, "win_rate")
    phys_unseen_threat = metric(summary, "unseen_eval", physical, "enemy_threat_exposure")
    lines = ["# Stage13 Main Claim Candidates", ""]
    lines.append("## A. Strong Claim")
    lines.append("")
    lines.append("Use only if then_fusion is clearly better than both curriculum-only and physical on mixed/unseen outcome plus tactical/safety metrics.")
    lines.append("")
    lines.append("## B. Medium Claim")
    lines.append("")
    lines.append("Recommended if then_fusion is competitive in win rate and improves safety, sample efficiency, or unseen robustness.")
    lines.append("")
    lines.append("## C. Conservative Claim")
    lines.append("")
    lines.append("Use if physical remains stronger overall, while VLM/surrogate reward is still valuable for interpretable semantic diagnostics.")
    lines.append("")
    lines.append("## Current Automatic Reading")
    lines.append("")
    lines.append("- then_fusion mixed win: {}".format(fmt(mixed_win)))
    lines.append("- original_curriculum mixed win: {}".format(fmt(base_win)))
    lines.append("- physical mixed win: {}".format(fmt(phys_win)))
    lines.append("- then_fusion mixed enemy threat: {}".format(fmt(mixed_threat)))
    lines.append("- physical mixed enemy threat: {}".format(fmt(phys_threat)))
    lines.append("- then_fusion mixed attack-window ratio: {}".format(fmt(mixed_attack)))
    lines.append("- then_fusion unseen win: {}".format(fmt(unseen_win)))
    lines.append("- physical unseen win: {}".format(fmt(phys_unseen_win)))
    lines.append("- then_fusion unseen enemy threat: {}".format(fmt(unseen_threat)))
    lines.append("- physical unseen enemy threat: {}".format(fmt(phys_unseen_threat)))
    if (
        mixed_win is not None and base_win is not None and phys_win is not None
        and mixed_threat is not None and phys_threat is not None
        and unseen_win is not None and phys_unseen_win is not None
        and mixed_win >= base_win and mixed_win >= phys_win
        and mixed_threat <= phys_threat
        and unseen_win >= phys_unseen_win
    ):
        lines.append("- Suggested claim tier: A. then_fusion is better across mixed outcome, safety, and unseen robustness.")
    elif mixed_win is not None and base_win is not None and mixed_win >= base_win:
        lines.append("- Suggested claim tier: B. then_fusion improves mixed outcome over curriculum-only and remains competitive, but safety/unseen claims need qualifiers.")
    elif mixed_threat is not None and phys_threat is not None and mixed_threat < phys_threat:
        lines.append("- Suggested claim tier: B. Emphasize safer semantic shaping and threat reduction, not universal dominance.")
    else:
        lines.append("- Suggested claim tier: C. Keep claims conservative.")
    (results_dir / "stage13_main_claim_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_risks(results_dir):
    summary = load_summary(results_dir)
    main = "PPO_surrogate_v2_then_fusion_with_situation_curriculum"
    lines = ["# Stage13 Risk Report", ""]
    for eval_type in ["default_eval", "offensive_eval", "mixed_eval", "unseen_eval"]:
        neutral = metric(summary, eval_type, main, "neutral_stalemate_time_ratio_mean")
        attack = metric(summary, eval_type, main, "effective_attack_window_time_ratio_mean")
        threat = metric(summary, eval_type, main, "enemy_threat_exposure")
        lines.append("## {}".format(eval_type))
        lines.append("")
        lines.append("- neutral: {}".format(fmt(neutral)))
        lines.append("- attack: {}".format(fmt(attack)))
        lines.append("- enemy threat: {}".format(fmt(threat)))
        if neutral is not None and neutral > 0.5:
            lines.append("- risk: neutral/stalemate tendency remains high.")
        if attack is not None and attack < 0.03:
            lines.append("- risk: attack-window occupancy is weak.")
        lines.append("")
    (results_dir / "stage13_risk_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_recommendation(results_dir):
    summary = load_summary(results_dir)
    main = "PPO_surrogate_v2_then_fusion_with_situation_curriculum"
    baseline = "PPO_original_with_situation_curriculum"
    physical = "PPO_physical"
    lines = ["# Stage13 Final Recommendation", ""]
    lines.append("Stage13 ran the 500k confirmation subset only: curriculum-only, physical shaping, and then_fusion curriculum.")
    lines.append("The Stage13 continuation uses Stage12 actor/critic weights plus 200k additional steps; PPO optimizer state is reset because the original runner does not save it.")
    lines.append("")
    lines.append("## Main Candidate")
    lines.append("")
    lines.append("`{}` remains the main NoWeapon candidate, with important qualifiers.".format(main))
    lines.append("")
    for eval_type in ["default_eval", "offensive_eval", "mixed_eval", "unseen_eval"]:
        lines.append("### {}".format(eval_type))
        lines.append("")
        for group in [baseline, physical, main]:
            lines.append("- `{}`: win {}, attack {}, threat {}, neutral {}".format(
                group,
                fmt(metric(summary, eval_type, group, "win_rate")),
                fmt(metric(summary, eval_type, group, "effective_attack_window_time_ratio_mean")),
                fmt(metric(summary, eval_type, group, "enemy_threat_exposure")),
                fmt(metric(summary, eval_type, group, "neutral_stalemate_time_ratio_mean")),
            ))
        lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- then_fusion is strongest on mixed_eval win and attack-window ratio among the Stage13 500k subset.")
    lines.append("- physical remains the safer baseline on enemy threat exposure, especially in mixed/unseen eval.")
    lines.append("- default_eval still has zero attack-window occupancy for all methods, consistent with the difficult natural initial distribution diagnosed in Stage11.")
    lines.append("- Recommended paper claim tier: medium, not strong universal dominance.")
    (results_dir / "stage13_final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    return "n/a" if value is None else "{:.4f}".format(float(value))


if __name__ == "__main__":
    main()
