#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


LABELS = [
    "ego_tail_advantage",
    "enemy_tail_threat",
    "effective_attack_window",
    "enemy_missile_threat_zone",
    "energy_advantage",
    "energy_disadvantage",
    "defensive_escape",
    "neutral_stalemate",
]

LABEL_SHORT = {
    "ego_tail_advantage": "Ego tail",
    "enemy_tail_threat": "Enemy tail",
    "effective_attack_window": "Attack win.",
    "enemy_missile_threat_zone": "Missile thr.",
    "energy_advantage": "Energy adv.",
    "energy_disadvantage": "Energy dis.",
    "defensive_escape": "Def. escape",
    "neutral_stalemate": "Neutral",
}

GROUP_TO_METHOD = {
    "PPO_original": "PPO",
    "PPO_original_with_situation_curriculum": "PPO-Cur",
    "PPO_physical": "PPO-Phys",
    "PPO_surrogate_v2_then_fusion_with_situation_curriculum": "PPO-PVVL-SR",
    "PPO_surrogate_v2_no_potential_with_situation_curriculum": "PPO-PVVL-SR-noPot",
    "PPO_surrogate_v2_direct_beta_low_with_situation_curriculum": "PPO-PVVL-SR-direct",
}

METHOD_STYLE = {
    "PPO": {"color": "#4D4D4D", "marker": "o", "linestyle": "-"},
    "PPO-Cur": {"color": "#607D8B", "marker": "s", "linestyle": "-"},
    "PPO-Phys": {"color": "#D08C3B", "marker": "^", "linestyle": "--"},
    "PPO-PVVL-SR": {"color": "#2F6FAE", "marker": "D", "linestyle": "-"},
    "PPO-PVVL-SR-noPot": {"color": "#7B5EA7", "marker": "v", "linestyle": ":"},
    "PPO-PVVL-SR-direct": {"color": "#8C3F4D", "marker": "x", "linestyle": "--"},
}

FINAL_TITLE = "PVVL-SR: Physics-Verified Vision-Language Semantic Reward Shaping for UAV Air Combat Reinforcement Learning"


def main():
    parser = argparse.ArgumentParser(description="Generate TAES paper experiment package from existing result files.")
    parser.add_argument("--output-dir", default="scripts/results/taes_paper_package")
    parser.add_argument("--stage13-dir", default="scripts/results/stage13_500k_confirmation")
    parser.add_argument("--stage12-dir", default="scripts/results/stage12_formal_ppo")
    parser.add_argument("--dodge-dir", default="scripts/results/stage13_dodgemissile_preliminary")
    args = parser.parse_args()
    result = generate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def generate(args):
    out = Path(args.output_dir)
    mkdirs(out)
    paths = collect_paths(args)
    data = load_all(paths)

    table_paths = write_tables(out, data)
    figure_paths = write_figures(out, data, paths)
    write_markdown_package(out, data, paths, table_paths, figure_paths)
    write_manifest(out, paths, table_paths, figure_paths)
    copy_figure_script(out)

    return {
        "output_dir": str(out.resolve()),
        "tables": {k: {kk: str(vv.resolve()) for kk, vv in bundle.items()} for k, bundle in table_paths.items()},
        "figures": {k: [str(path.resolve()) for path in v] for k, v in figure_paths.items()},
    }


def mkdirs(out):
    for sub in [
        "",
        "main_results_tables",
        "figures",
        "figure_sources",
        "figure_sources/data",
        "result_analysis_text",
        "appendix_material",
    ]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def collect_paths(args):
    root = Path("scripts/results")
    return {
        "stage13_dir": Path(args.stage13_dir),
        "stage12_dir": Path(args.stage12_dir),
        "dodge_dir": Path(args.dodge_dir),
        "stage13_summary": Path(args.stage13_dir) / "stage13_summary.json",
        "stage12_summary": Path(args.stage12_dir) / "stage12_summary.json",
        "stage13_vs_stage12": Path(args.stage13_dir) / "stage13_vs_stage12_comparison.json",
        "stage13_sample_efficiency": Path(args.stage13_dir) / "stage13_sample_efficiency_table.csv",
        "stage7_fusion_summary": root / "stage7_fusion_eval_formal100" / "summary.json",
        "stage7_fusion_per_label": root / "stage7_fusion_eval_formal100" / "per_label_metrics.csv",
        "stage7_cache_coverage": root / "stage7_cache_coverage_random1000" / "coverage_summary.json",
        "stage8_surrogate_v1_summary": root / "surrogate_eval" / "stage8_surrogate_v1" / "summary.json",
        "stage8_surrogate_v1_per_label": root / "surrogate_eval" / "stage8_surrogate_v1" / "per_label_metrics.csv",
        "stage9_surrogate_v2_summary": root / "surrogate_eval" / "stage9_surrogate_v2" / "summary.json",
        "stage9_surrogate_v2_per_label": root / "surrogate_eval" / "stage9_surrogate_v2" / "per_label_metrics.csv",
        "stage9_surrogate_benchmark": root / "surrogate_benchmark" / "stage9_benchmark_surrogate_v2_cpu.json",
        "stage11_metric_sanity": root / "stage11_metric_sanity" / "sanity_report.md",
        "stage11_threshold_report": root / "stage11_threshold_analysis" / "threshold_report.md",
        "stage11_geometry_report": root / "stage11_rollout_geometry_distribution" / "geometry_report.md",
        "stage12_ablation": Path(args.stage12_dir) / "stage12_ablation_table.md",
        "stage12_weapon": root / "stage12_weapon_scenario_check" / "weapon_scenario_availability.md",
        "dodge_summary": Path(args.dodge_dir) / "weapon_preliminary_summary.json",
        "dodge_report": Path(args.dodge_dir) / "weapon_preliminary_report.md",
        "pvvl_success_ts": root / "stage12_formal_ppo" / "trajectory_visualizations" / "PPO_surrogate_v2_then_fusion_with_situation_curriculum_seed0" / "successful_offensive_timeseries.csv",
        "pvvl_failure_ts": root / "taes_stage13_trajectory_sources" / "PPO_PVVL_SR_500k_seed0" / "stalemate_or_failure_timeseries.csv",
        "phys_safe_ts": root / "taes_stage13_trajectory_sources" / "PPO_Phys_500k_seed0" / "defensive_recovery_timeseries.csv",
    }


def load_all(paths):
    return {
        "stage13": load_json(paths["stage13_summary"]),
        "stage12": load_json(paths["stage12_summary"]),
        "stage13_vs_stage12": load_json(paths["stage13_vs_stage12"]),
        "sample_eff": read_csv(paths["stage13_sample_efficiency"]),
        "fusion_summary": load_json(paths["stage7_fusion_summary"]),
        "fusion_per_label": read_csv(paths["stage7_fusion_per_label"]),
        "cache_coverage": load_json(paths["stage7_cache_coverage"]),
        "surrogate_v1_summary": load_json(paths["stage8_surrogate_v1_summary"]),
        "surrogate_v1_per_label": read_csv(paths["stage8_surrogate_v1_per_label"]),
        "surrogate_v2_summary": load_json(paths["stage9_surrogate_v2_summary"]),
        "surrogate_v2_per_label": read_csv(paths["stage9_surrogate_v2_per_label"]),
        "surrogate_benchmark": load_json(paths["stage9_surrogate_benchmark"]),
        "dodge": load_json(paths["dodge_summary"]),
    }


def write_tables(out, data):
    table_dir = out / "main_results_tables"
    paths = {}
    paths["table1"] = write_eval_table(
        table_dir,
        "table1_mixed_eval_stage13",
        "Table I. Overall comparison in mixed evaluation (Stage13 500k confirmation).",
        data["stage13"],
        "mixed_eval",
        ["win_rate", "draw_rate", "return_ego_mean", "effective_attack_window_time_ratio_mean",
         "ego_tail_advantage_time_ratio_mean", "enemy_threat_exposure", "neutral_stalemate_time_ratio_mean"],
        ["PPO_original_with_situation_curriculum", "PPO_physical", "PPO_surrogate_v2_then_fusion_with_situation_curriculum"],
    )
    paths["table2"] = write_eval_table(
        table_dir,
        "table2_offensive_eval_stage13",
        "Table II. Offensive evaluation (Stage13 500k confirmation).",
        data["stage13"],
        "offensive_eval",
        ["win_rate", "effective_attack_window_time_ratio_mean", "ego_tail_advantage_time_ratio_mean",
         "attack_window_first_entry_time_mean", "neutral_stalemate_time_ratio_mean"],
        ["PPO_original_with_situation_curriculum", "PPO_physical", "PPO_surrogate_v2_then_fusion_with_situation_curriculum"],
    )
    paths["table3"] = write_eval_table(
        table_dir,
        "table3_unseen_eval_stage13",
        "Table III. Unseen evaluation (Stage13 500k confirmation).",
        data["stage13"],
        "unseen_eval",
        ["win_rate", "effective_attack_window_time_ratio_mean", "neutral_stalemate_time_ratio_mean", "enemy_threat_exposure"],
        ["PPO_original_with_situation_curriculum", "PPO_physical", "PPO_surrogate_v2_then_fusion_with_situation_curriculum"],
    )
    paths["table4"] = write_ablation_table(table_dir, data)
    paths["table5"] = write_dodge_table(table_dir, data)
    return paths


def write_eval_table(table_dir, stem, caption, summary, eval_type, metrics, groups):
    rows = []
    for group in groups:
        row = {"Method": GROUP_TO_METHOD[group]}
        source = find_group(summary, eval_type, group)
        for metric in metrics:
            row[metric] = fmt_mean_std(metric_mean(source, metric), metric_std(source, metric))
        rows.append(row)
    csv_path = table_dir / f"{stem}.csv"
    md_path = table_dir / f"{stem}.md"
    tex_path = table_dir / f"{stem}.tex"
    headers = ["Method"] + metrics
    write_csv_rows(csv_path, rows, headers)
    write_md_table(md_path, caption, rows, headers)
    write_latex_table(tex_path, caption, rows, headers)
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def write_ablation_table(table_dir, data):
    fsum = data["fusion_summary"]
    csum = data["cache_coverage"]
    v1 = data["surrogate_v1_summary"]
    v2 = data["surrogate_v2_summary"]
    bench = data["surrogate_benchmark"]
    st12 = data["stage12"]
    then_row = find_group(st12, "mixed_eval", "PPO_surrogate_v2_then_fusion_with_situation_curriculum")
    nop_row = find_group(st12, "mixed_eval", "PPO_surrogate_v2_no_potential_with_situation_curriculum")
    cur_row = find_group(st12, "mixed_eval", "PPO_original_with_situation_curriculum")
    ppo_row = find_group(st12, "mixed_eval", "PPO_original")
    rows = [
        {
            "Ablation": "Raw VLM -> label-wise fused",
            "Metric": "overall label MAE; high-diff cases",
            "Baseline/Before": f"{fsum['overall_vlm_mae']:.4f}; {fsum['high_diff_cases_before_fusion']}/{fsum['num_samples']}",
            "Variant/After": f"{fsum['overall_fused_mae']:.4f}; {fsum['high_diff_cases_after_fusion']}/{fsum['num_samples']}",
            "Conclusion": "Fusion substantially improves label reliability.",
        },
        {
            "Ablation": "Exact cached VLM -> surrogate",
            "Metric": "exact cache hit; CPU inference time",
            "Baseline/Before": f"{csum['exact_cache_hit_rate']:.3f} hit; {csum['fallback_physical_rate']:.3f} fallback",
            "Variant/After": f"{bench['avg_inference_time_sec']:.2e} s/query",
            "Conclusion": "Surrogate is needed for continuous PPO states.",
        },
        {
            "Ablation": "Surrogate v1 -> surrogate v2",
            "Metric": "overall MAE; high-diff ratio",
            "Baseline/Before": f"{v1['overall_mae']:.4f}; {100*v1['high_diff_ratio']:.1f}%",
            "Variant/After": f"{v2['overall_mae']:.4f}; {100*v2['high_diff_ratio']:.1f}%",
            "Conclusion": "Boundary augmentation improves surrogate accuracy.",
        },
        {
            "Ablation": "No potential -> potential shaping",
            "Metric": "Stage12 mixed win; enemy threat",
            "Baseline/Before": f"{metric_mean(nop_row, 'win_rate'):.4f}; {metric_mean(nop_row, 'enemy_threat_exposure'):.4f}",
            "Variant/After": f"{metric_mean(then_row, 'win_rate'):.4f}; {metric_mean(then_row, 'enemy_threat_exposure'):.4f}",
            "Conclusion": "Potential shaping improves outcome/threat control, not every tactical metric.",
        },
        {
            "Ablation": "No curriculum -> mixed curriculum",
            "Metric": "Stage12 mixed win; attack-window ratio",
            "Baseline/Before": f"{metric_mean(ppo_row, 'win_rate'):.4f}; {metric_mean(ppo_row, 'effective_attack_window_time_ratio_mean'):.4f}",
            "Variant/After": f"{metric_mean(cur_row, 'win_rate'):.4f}; {metric_mean(cur_row, 'effective_attack_window_time_ratio_mean'):.4f}",
            "Conclusion": "Curriculum is necessary for attack-window exposure, but does not solve all metrics by itself.",
        },
    ]
    headers = ["Ablation", "Metric", "Baseline/Before", "Variant/After", "Conclusion"]
    caption = "Table IV. Ablation and diagnostic evidence supporting the reward-modeling pipeline."
    csv_path = table_dir / "table4_ablation_study.csv"
    md_path = table_dir / "table4_ablation_study.md"
    tex_path = table_dir / "table4_ablation_study.tex"
    write_csv_rows(csv_path, rows, headers)
    write_md_table(md_path, caption, rows, headers)
    write_latex_table(tex_path, caption, rows, headers)
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def write_dodge_table(table_dir, data):
    rows = []
    for run in data["dodge"].get("preliminary_runs", []):
        s = load_json(run.get("eval_summary", ""))
        rows.append({
            "Method": GROUP_TO_METHOD.get(run["group"], run["group"].replace("PPO_surrogate_v2_then_fusion", "PPO-PVVL-SR")),
            "Return": fmt_num(s.get("return_ego_mean")),
            "Episode length": fmt_num(s.get("episode_length_mean")),
            "Survival time": fmt_num(s.get("survival_time_mean")),
            "Missile threat ratio": fmt_num(s.get("enemy_missile_threat_zone_time_ratio_mean")),
            "Defensive escape ratio": fmt_num(s.get("defensive_escape_time_ratio_mean")),
            "Win/Loss": f"{fmt_num(s.get('win_rate'))}/{fmt_num(s.get('loss_rate'))}",
        })
    headers = ["Method", "Return", "Episode length", "Survival time", "Missile threat ratio", "Defensive escape ratio", "Win/Loss"]
    caption = "Table V. DodgeMissile preliminary validation. This is a compatibility and weak-transfer check only; all methods still lose under 50k single-seed training."
    csv_path = table_dir / "table5_dodgemissile_preliminary.csv"
    md_path = table_dir / "table5_dodgemissile_preliminary.md"
    tex_path = table_dir / "table5_dodgemissile_preliminary.tex"
    write_csv_rows(csv_path, rows, headers)
    write_md_table(md_path, caption, rows, headers)
    write_latex_table(tex_path, caption, rows, headers)
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def write_figures(out, data, paths):
    ensure_matplotlib()
    fig_dir = out / "figures"
    src_dir = out / "figure_sources" / "data"
    outputs = {}
    outputs["fig1_architecture"] = fig_architecture(fig_dir)
    outputs["fig2_label_quality"] = fig_label_quality(fig_dir, src_dir, data)
    outputs["fig3_surrogate_accuracy"] = fig_surrogate_accuracy(fig_dir, src_dir, data)
    outputs["fig4_mixed_performance"] = fig_mixed_performance(fig_dir, src_dir, data)
    outputs["fig5_300k_500k_trend"] = fig_stage_trend(fig_dir, src_dir, data)
    outputs["fig6_sample_efficiency"] = fig_sample_efficiency(fig_dir, src_dir, data)
    outputs["fig7_representative_trajectories"] = fig_trajectories(fig_dir, src_dir, paths)
    outputs["fig8_dodgemissile_preliminary"] = fig_dodge(fig_dir, src_dir, data)
    write_figure_captions(out, outputs)
    return outputs


def ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def fig_architecture(fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.16, 2.4))
    ax.axis("off")
    labels = [
        "LAG state",
        "Situation image",
        "Qwen2.5-VL\noffline teacher",
        "Physical\nverification",
        "Label-wise\nfusion",
        "Surrogate\nMLP",
        "Potential-based\nshaping",
        "PPO update",
    ]
    x_positions = [0.02, 0.15, 0.30, 0.45, 0.58, 0.70, 0.82, 0.93]
    widths = [0.10, 0.12, 0.13, 0.12, 0.10, 0.10, 0.11, 0.08]
    y, h = 0.42, 0.26
    for i, (x, w, text) in enumerate(zip(x_positions, widths, labels)):
        color = "#E8EEF7" if i not in (3, 4) else "#F7EFE3"
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.015",
                             facecolor=color, edgecolor="#333333", linewidth=0.9)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")
        if i < len(labels) - 1:
            x2 = x_positions[i + 1]
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x2, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#333333"))
    ax.text(0.30, 0.20, "VLM is used only offline; PPO deployment uses the surrogate reward model.", ha="center", va="center")
    ax.text(0.58, 0.78, "physics-guided semantic label reliability", ha="center", va="center", color="#444444")
    return save_figure(fig, fig_dir / "fig1_method_architecture")


def fig_label_quality(fig_dir, src_dir, data):
    import matplotlib.pyplot as plt
    rows = []
    for row in data["fusion_per_label"]:
        rows.append({"label": row["label"], "raw_vlm_mae": fnum(row["vlm_mae"]), "fused_mae": fnum(row["fused_mae"])})
    write_csv_rows(src_dir / "fig2_label_quality_source.csv", rows, ["label", "raw_vlm_mae", "fused_mae"])
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    width = 0.36
    ax.bar([i - width / 2 for i in x], [r["raw_vlm_mae"] for r in rows], width, label="Raw VLM", color="#8C6BB1")
    ax.bar([i + width / 2 for i in x], [r["fused_mae"] for r in rows], width, label="Physics-verified fused", color="#2F6FAE")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_SHORT[r["label"]] for r in rows], rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, fig_dir / "fig2_label_quality_improvement")


def fig_surrogate_accuracy(fig_dir, src_dir, data):
    import matplotlib.pyplot as plt
    v1 = {r["label"]: fnum(r["mae"]) for r in data["surrogate_v1_per_label"]}
    v2 = {r["label"]: fnum(r["mae"]) for r in data["surrogate_v2_per_label"]}
    rows = [{"label": label, "surrogate_v1_mae": v1.get(label, 0.0), "surrogate_v2_mae": v2.get(label, 0.0)} for label in LABELS]
    write_csv_rows(src_dir / "fig3_surrogate_accuracy_source.csv", rows, ["label", "surrogate_v1_mae", "surrogate_v2_mae"])
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    width = 0.36
    ax.bar([i - width / 2 for i in x], [r["surrogate_v1_mae"] for r in rows], width, label="Surrogate v1", color="#7B5EA7")
    ax.bar([i + width / 2 for i in x], [r["surrogate_v2_mae"] for r in rows], width, label="Surrogate v2", color="#2F6FAE")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_SHORT[r["label"]] for r in rows], rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, fig_dir / "fig3_surrogate_model_accuracy")


def fig_mixed_performance(fig_dir, src_dir, data):
    import matplotlib.pyplot as plt
    methods = [
        ("PPO-Cur", "PPO_original_with_situation_curriculum"),
        ("PPO-Phys", "PPO_physical"),
        ("PPO-PVVL-SR", "PPO_surrogate_v2_then_fusion_with_situation_curriculum"),
    ]
    metrics = [
        ("win_rate", "Win rate"),
        ("draw_rate", "Draw rate"),
        ("effective_attack_window_time_ratio_mean", "Attack-window"),
        ("neutral_stalemate_time_ratio_mean", "Neutral"),
        ("enemy_threat_exposure", "Enemy threat"),
    ]
    rows = []
    for mname, group in methods:
        row = find_group(data["stage13"], "mixed_eval", group)
        for metric, label in metrics:
            rows.append({"method": mname, "metric": metric, "mean": metric_mean(row, metric), "std": metric_std(row, metric)})
    write_csv_rows(src_dir / "fig4_mixed_performance_source.csv", rows, ["method", "metric", "mean", "std"])
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0))
    axes = axes.flatten()
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        means = [r["mean"] for r in rows if r["metric"] == metric]
        stds = [r["std"] for r in rows if r["metric"] == metric]
        colors = [METHOD_STYLE[m[0]]["color"] for m in methods]
        ax.bar(range(len(methods)), means, yerr=stds, capsize=2, color=colors, edgecolor="#333333", linewidth=0.4)
        ax.set_title(label)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m[0] for m in methods], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    fig.tight_layout()
    return save_figure(fig, fig_dir / "fig4_mixed_evaluation_performance")


def fig_stage_trend(fig_dir, src_dir, data):
    import matplotlib.pyplot as plt
    group = "PPO_surrogate_v2_then_fusion_with_situation_curriculum"
    metric_names = [
        ("win_rate", "Win"),
        ("effective_attack_window_time_ratio_mean", "Attack"),
        ("ego_tail_advantage_time_ratio_mean", "Tail"),
        ("neutral_stalemate_time_ratio_mean", "Neutral"),
        ("enemy_threat_exposure", "Enemy threat"),
    ]
    rows = []
    for item in data["stage13_vs_stage12"]:
        if item["group"] == group and item["eval_type"] == "mixed_eval" and item["metric"] in [m[0] for m in metric_names]:
            rows.append({
                "metric": item["metric"],
                "stage12_300k": item["stage12_300k"],
                "stage13_500k": item["stage13_500k"],
                "delta": item["delta_500k_minus_300k"],
            })
    write_csv_rows(src_dir / "fig5_300k_500k_trend_source.csv", rows, ["metric", "stage12_300k", "stage13_500k", "delta"])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = [0, 1]
    for metric, label in metric_names:
        row = next((r for r in rows if r["metric"] == metric), None)
        if not row:
            continue
        ax.plot(x, [row["stage12_300k"], row["stage13_500k"]], marker=METHOD_STYLE["PPO-PVVL-SR"]["marker"],
                linewidth=1.4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["300k", "500k"])
    ax.set_ylabel("Ratio / rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    return save_figure(fig, fig_dir / "fig5_300k_to_500k_trend")


def fig_sample_efficiency(fig_dir, src_dir, data):
    import matplotlib.pyplot as plt
    wanted = ["PPO_physical", "PPO_surrogate_v2_then_fusion_with_situation_curriculum"]
    rows = []
    for row in data["sample_eff"]:
        if row.get("threshold") == "mixed_win_ge_0.30" and row.get("group") in wanted:
            rows.append({
                "method": GROUP_TO_METHOD[row["group"]],
                "mean_reached_step": fnum(row["mean_reached_step"]) if row["mean_reached_step"] != "not reached" else math.nan,
                "std_reached_step": fnum(row["std_reached_step"]) if row["std_reached_step"] != "not reached" else math.nan,
                "status": row["status"],
            })
    write_csv_rows(src_dir / "fig6_sample_efficiency_source.csv", rows, ["method", "mean_reached_step", "std_reached_step", "status"])
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    colors = [METHOD_STYLE[r["method"]]["color"] for r in rows]
    ax.bar([r["method"] for r in rows], [r["mean_reached_step"] / 1000.0 for r in rows],
           yerr=[r["std_reached_step"] / 1000.0 for r in rows], color=colors, capsize=2, edgecolor="#333333", linewidth=0.4)
    ax.set_ylabel("Steps to mixed win >= 0.30 (k)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, fig_dir / "fig6_sample_efficiency")


def fig_trajectories(fig_dir, src_dir, paths):
    import matplotlib.pyplot as plt
    sources = [
        ("PPO-PVVL-SR offensive", paths["pvvl_success_ts"]),
        ("PPO-Phys safe/defensive", paths["phys_safe_ts"]),
        ("PPO-PVVL-SR stalemate", paths["pvvl_failure_ts"]),
    ]
    source_rows = []
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.2))
    for idx, (title, path) in enumerate(sources):
        rows = read_csv(path) if Path(path).exists() else []
        for row in rows:
            source_rows.append({
                "case": title,
                "time_sec": row.get("time_sec", ""),
                "ego_x": row.get("ego_x", ""),
                "ego_y": row.get("ego_y", ""),
                "enemy_x": row.get("enemy_x", ""),
                "enemy_y": row.get("enemy_y", ""),
                "distance": row.get("distance", ""),
                "tail_score": row.get("ego_tail_advantage_score", ""),
                "attack_score": row.get("effective_attack_window_score", ""),
                "semantic_phi": row.get("semantic_phi", ""),
            })
        ax_traj, ax_ts = axes[idx, 0], axes[idx, 1]
        if not rows:
            ax_traj.text(0.5, 0.5, "missing source", ha="center", va="center")
            ax_ts.axis("off")
            continue
        ego_x = [fnum(r["ego_y"]) for r in rows]
        ego_y = [fnum(r["ego_x"]) for r in rows]
        enemy_x = [fnum(r["enemy_y"]) for r in rows]
        enemy_y = [fnum(r["enemy_x"]) for r in rows]
        t = [fnum(r["time_sec"]) for r in rows]
        ax_traj.plot(ego_x, ego_y, color="#2F6FAE", linewidth=1.3, label="Ego")
        ax_traj.plot(enemy_x, enemy_y, color="#8C3F4D", linewidth=1.3, linestyle="--", label="Enemy")
        ax_traj.set_title(title)
        ax_traj.set_xlabel("East (m)")
        ax_traj.set_ylabel("North (m)")
        ax_traj.legend(frameon=False, loc="best")
        ax_ts.plot(t, [fnum(r["distance"]) / 1000.0 for r in rows], label="Distance (km)", color="#4D4D4D")
        ax_ts.plot(t, [fnum(r["ego_tail_advantage_score"]) for r in rows], label="Tail score", color="#2F6FAE")
        ax_ts.plot(t, [fnum(r["effective_attack_window_score"]) for r in rows], label="Attack score", color="#D08C3B")
        ax_ts.plot(t, [fnum(r.get("semantic_phi", 0.0)) for r in rows], label="Phi", color="#7B5EA7", linestyle=":")
        ax_ts.set_xlabel("Time (s)")
        ax_ts.grid(alpha=0.25)
        ax_ts.legend(frameon=False, ncol=2)
    write_csv_rows(src_dir / "fig7_representative_trajectories_source.csv", source_rows,
                   ["case", "time_sec", "ego_x", "ego_y", "enemy_x", "enemy_y", "distance", "tail_score", "attack_score", "semantic_phi"])
    fig.tight_layout()
    return save_figure(fig, fig_dir / "fig7_representative_trajectories")


def fig_dodge(fig_dir, src_dir, data):
    import matplotlib.pyplot as plt
    rows = []
    for run in data["dodge"].get("preliminary_runs", []):
        s = load_json(run.get("eval_summary", ""))
        name = GROUP_TO_METHOD.get(run["group"], run["group"].replace("PPO_surrogate_v2_then_fusion", "PPO-PVVL-SR"))
        rows.append({
            "method": name,
            "return": s.get("return_ego_mean"),
            "survival_time": s.get("survival_time_mean"),
            "defensive_escape": s.get("defensive_escape_time_ratio_mean"),
        })
    write_csv_rows(src_dir / "fig8_dodgemissile_source.csv", rows, ["method", "return", "survival_time", "defensive_escape"])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    metrics = [("return", "Return"), ("survival_time", "Survival (s)"), ("defensive_escape", "Def. escape")]
    for ax, (metric, label) in zip(axes, metrics):
        ax.bar([r["method"] for r in rows], [float(r[metric]) for r in rows],
               color=[METHOD_STYLE.get(r["method"], METHOD_STYLE["PPO-PVVL-SR"])["color"] for r in rows],
               edgecolor="#333333", linewidth=0.4)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("DodgeMissile preliminary validation only")
    fig.tight_layout()
    return save_figure(fig, fig_dir / "fig8_dodgemissile_preliminary")


def write_figure_captions(out, outputs):
    captions = [
        ("Fig. 1", "Overview of the PVVL-SR pipeline. The VLM teacher is used offline to annotate standardized air-combat situations; deployment-time PPO uses a lightweight surrogate reward model together with physical verification and potential-based shaping."),
        ("Fig. 2", "Comparison of raw VLM scores and physics-verified fused scores on the semantic label dataset. The fusion mechanism substantially reduces errors in energy- and threat-related labels, where raw VLM scores are less reliable."),
        ("Fig. 3", "Per-label surrogate prediction error for surrogate v1 and surrogate v2. Boundary and geometry augmentation reduces the overall surrogate error and the number of high-difference cases."),
        ("Fig. 4", "Mixed-evaluation performance after the Stage13 500k confirmation. PPO-PVVL-SR achieves the highest win rate and attack-window occupancy and the lowest draw/stalemate tendency among the compared Stage13 methods, while PPO-Phys retains lower enemy-threat exposure."),
        ("Fig. 5", "Trend of PPO-PVVL-SR from 300k to 500k in mixed evaluation. Attack-window and tail-advantage ratios increase and neutral stalemate decreases, but enemy-threat exposure increases relative to 300k."),
        ("Fig. 6", "Sample efficiency for reaching mixed-evaluation win rate >= 0.30. This auxiliary comparison is based on checkpoint evaluation and should be interpreted with checkpoint variance in mind."),
        ("Fig. 7", "Representative trajectory examples. The panels show a PVVL-SR offensive case, a PPO-Phys safe/defensive case, and a PVVL-SR stalemate case, together with distance, tactical scores, and semantic potential."),
        ("Fig. 8", "DodgeMissile preliminary validation. The results indicate wrapper compatibility and weak transfer signals only; they do not constitute full weapon-enabled combat validation."),
    ]
    (out / "figure_captions.md").write_text("\n\n".join([f"**{k}.** {v}" for k, v in captions]) + "\n", encoding="utf-8")


def save_figure(fig, stem):
    paths = []
    for suffix, kwargs in [
        (".pdf", {}),
        (".eps", {}),
        (".png", {"dpi": 600}),
    ]:
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    try:
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        paths.append(stem.with_suffix(".svg"))
    except Exception:
        pass
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def write_markdown_package(out, data, paths, table_paths, figure_paths):
    write_experiment_summary(out, data, table_paths, figure_paths)
    write_claims(out)
    write_protocol(out, paths)
    write_method_ablation(out, data)
    write_analysis_texts(out, data)
    write_appendix(out, paths)
    write_reproducibility(out, paths)
    write_report(out, data, table_paths, figure_paths)


def write_experiment_summary(out, data, table_paths, figure_paths):
    stage13 = data["stage13"]
    pvvl_mixed = find_group(stage13, "mixed_eval", "PPO_surrogate_v2_then_fusion_with_situation_curriculum")
    phys_mixed = find_group(stage13, "mixed_eval", "PPO_physical")
    cur_mixed = find_group(stage13, "mixed_eval", "PPO_original_with_situation_curriculum")
    lines = [
        "# TAES Paper Experiment Summary",
        "",
        f"Recommended title: **{FINAL_TITLE}**",
        "",
        "Recommended method acronym: **PVVL-SR**, standing for Physics-Verified Vision-Language Semantic Reward Shaping.",
        "",
        "The acronym is useful here because the paper is centered on a reward-shaping module rather than a new policy architecture. It is short enough for figures and tables, while the full title remains descriptive and avoids overclaiming.",
        "",
        "## Method Name Mapping",
        "",
        "| Experiment file name | Paper method name | Meaning |",
        "|---|---|---|",
        "| PPO_original | PPO | Original PPO, no reward shaping and no curriculum. |",
        "| PPO_original_with_situation_curriculum | PPO-Cur | PPO with mixed situation curriculum. |",
        "| PPO_physical | PPO-Phys | PPO with physical reward shaping. |",
        "| PPO_surrogate_v2_then_fusion_with_situation_curriculum | PPO-PVVL-SR | Main method: surrogate semantic reward, label-wise fusion, potential shaping, mixed curriculum. |",
        "| PPO_surrogate_v2_no_potential_with_situation_curriculum | PPO-PVVL-SR-noPot | Potential-shaping ablation. |",
        "| PPO_surrogate_v2_direct_beta_low_with_situation_curriculum | PPO-PVVL-SR-direct | Diagnostic direct-output variant, not the main method. |",
        "",
        "## Main Stage13 500k Evidence",
        "",
        f"- PPO-PVVL-SR mixed win rate: {fmt_mean_std(metric_mean(pvvl_mixed, 'win_rate'), metric_std(pvvl_mixed, 'win_rate'))}.",
        f"- PPO-Cur mixed win rate: {fmt_mean_std(metric_mean(cur_mixed, 'win_rate'), metric_std(cur_mixed, 'win_rate'))}.",
        f"- PPO-Phys mixed win rate: {fmt_mean_std(metric_mean(phys_mixed, 'win_rate'), metric_std(phys_mixed, 'win_rate'))}.",
        f"- PPO-PVVL-SR mixed attack-window ratio: {fmt_mean_std(metric_mean(pvvl_mixed, 'effective_attack_window_time_ratio_mean'), metric_std(pvvl_mixed, 'effective_attack_window_time_ratio_mean'))}.",
        f"- PPO-PVVL-SR mixed enemy-threat exposure: {fmt_mean_std(metric_mean(pvvl_mixed, 'enemy_threat_exposure'), metric_std(pvvl_mixed, 'enemy_threat_exposure'))}; PPO-Phys remains lower at {fmt_mean_std(metric_mean(phys_mixed, 'enemy_threat_exposure'), metric_std(phys_mixed, 'enemy_threat_exposure'))}.",
        "",
        "## Generated Main Tables",
    ]
    for key, bundle in table_paths.items():
        lines.append(f"- {key}: `{rel(bundle['md'])}`")
    lines.extend(["", "## Generated Figures"])
    for key, files in figure_paths.items():
        lines.append(f"- {key}: `{rel(files[0])}`")
    (out / "paper_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claims(out):
    lines = [
        "# Claims and Limitations",
        "",
        "## Supported Claims",
        "",
        "- The proposed PPO-PVVL-SR method improves mixed-evaluation win rate and reduces draw/stalemate behavior compared with PPO-Cur and PPO-Phys in the Stage13 500k NoWeapon setting.",
        "- PPO-PVVL-SR improves attack-window occupancy in mixed and offensive evaluations relative to PPO-Phys in the Stage13 500k subset.",
        "- Label-wise physics-verified fusion substantially improves raw VLM semantic label reliability.",
        "- The surrogate reward network enables fast semantic reward inference and avoids online VLM calls during PPO training or deployment.",
        "- DodgeMissile preliminary experiments indicate wrapper compatibility and weak transfer signals, but not full weapon combat success.",
        "",
        "## Claims to Avoid",
        "",
        "- The method fully outperforms physical reward shaping.",
        "- The method solves UAV air combat.",
        "- The method validates missile combat capability.",
        "- The VLM directly controls the UAV.",
        "- The method achieves state-of-the-art performance without comparison to other strong baselines.",
        "- NoWeapon results prove weapon-enabled air-combat performance.",
        "",
        "## Limitations and Honest Interpretation",
        "",
        "- PPO-Phys remains a strong threat-control baseline and has lower enemy-threat exposure than PPO-PVVL-SR in Stage13 mixed and unseen evaluations.",
        "- Default initial-state evaluation still rarely produces attack-window occupancy, consistent with the reachability diagnosis in Stage11.",
        "- DodgeMissile validation is single-seed, 50k-step, and preliminary; all methods still lose in the reported DodgeMissile evaluation.",
        "- The Stage13 500k confirmation resumes actor/critic weights from 300k, but PPO optimizer state is reset because the original runner does not save optimizer checkpoints.",
        "- The VLM teacher is offline and surrogate-based; it is not an online controller and does not output actions.",
    ]
    (out / "paper_claims_and_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol(out, paths):
    lines = [
        "# Experiment Protocol Summary",
        "",
        f"- Simulator: LAG-master.",
        "- Main scenario: `1v1/NoWeapon/Selfplay`.",
        "- Weapon preliminary scenario: `1v1/DodgeMissile/Selfplay`.",
        "- RL algorithm: PPO.",
        "- VLM teacher: Qwen2.5-VL-7B-Instruct.",
        "- VLM usage: offline semantic teacher only; no online VLM calls in PPO step loops.",
        "- Semantic surrogate: lightweight MLP mapping air-combat state features to fused semantic label scores.",
        "- Reward: environment reward plus physical reward and potential-based semantic shaping.",
        "- Curriculum: mixed situation curriculum with offensive, neutral, defensive, and random initial situations.",
        "- Stage13 seeds: 0, 1, 2 for the 500k confirmation subset.",
        "- Stage13 training: Stage12 300k actor/critic weights plus 200k continuation steps.",
        "- Stage13 final evaluation: default_eval, offensive_eval, mixed_eval, and unseen_eval.",
        "- Stage13 final evaluation episodes: 200 per evaluation type.",
        "- Important caveat: Stage13 continuation restores actor/critic weights only; PPO optimizer state is reset.",
        "- Hardware: local Windows workstation; GPU available in prior Qwen checks: NVIDIA GeForce RTX 4080 SUPER.",
        "",
        "## Key Source Files",
        "",
    ]
    for key in ["stage13_summary", "stage12_summary", "stage7_fusion_summary", "stage7_cache_coverage", "dodge_summary"]:
        lines.append(f"- {key}: `{rel(paths[key])}`")
    (out / "experiment_protocol_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_ablation(out, data):
    fsum = data["fusion_summary"]
    v1 = data["surrogate_v1_summary"]
    v2 = data["surrogate_v2_summary"]
    cache = data["cache_coverage"]
    bench = data["surrogate_benchmark"]
    lines = [
        "# Method Ablation Summary",
        "",
        "## Label-wise Fusion",
        "",
        f"Raw VLM overall MAE on formal100 is {fsum['overall_vlm_mae']:.4f}, while fused MAE is {fsum['overall_fused_mae']:.4f}. High-difference cases decrease from {fsum['high_diff_cases_before_fusion']}/{fsum['num_samples']} to {fsum['high_diff_cases_after_fusion']}/{fsum['num_samples']}.",
        "",
        "## Cached VLM versus Surrogate",
        "",
        f"Exact cached-VLM hit rate during random PPO rollout is {cache['exact_cache_hit_rate']:.4f}, with fallback rate {cache['fallback_physical_rate']:.4f}. This motivates surrogate learning rather than exact image-cache lookup.",
        "",
        "## Surrogate v1 versus v2",
        "",
        f"Surrogate v1 overall MAE is {v1['overall_mae']:.4f} with high-diff ratio {100*v1['high_diff_ratio']:.1f}%. Surrogate v2 improves to MAE {v2['overall_mae']:.4f} and high-diff ratio {100*v2['high_diff_ratio']:.1f}%.",
        "",
        "## Inference Speed",
        "",
        f"Surrogate v2 CPU inference averages {bench['avg_inference_time_sec']:.2e} s over {bench['repeat']} repeats, compared with a Qwen reference time of {bench['qwen_reference_time_sec']:.2f} s per image.",
        "",
        "## Potential-Based Shaping",
        "",
        "The no-potential ablation is not uniformly worse on every tactical metric, but Stage12 shows weaker outcome/threat control than potential-based PPO-PVVL-SR. The paper should phrase this as a stability and outcome-control benefit, not as universal dominance.",
    ]
    (out / "method_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_texts(out, data):
    text_dir = out / "result_analysis_text"
    texts = {
        "experimental_setup.md": experimental_setup_text(),
        "semantic_label_quality.md": semantic_label_quality_text(data),
        "surrogate_reward_network.md": surrogate_reward_text(data),
        "main_noweapon_results.md": main_results_text(data),
        "offensive_and_unseen_eval.md": offensive_unseen_text(data),
        "potential_based_shaping_ablation.md": potential_text(data),
        "dodgemissile_preliminary_validation.md": dodge_text(data),
        "limitations.md": limitations_text(),
    }
    for name, text in texts.items():
        (text_dir / name).write_text(text.strip() + "\n", encoding="utf-8")


def experimental_setup_text():
    return """
The experiments are conducted in the LAG-master 1v1 UAV air-combat simulator using the 1v1/NoWeapon/Selfplay scenario as the main benchmark. The learning algorithm is PPO with the original policy network kept unchanged. The proposed reward-modeling pipeline uses Qwen2.5-VL-7B-Instruct only as an offline semantic teacher. Standardized two-dimensional situation images are generated from the simulator state and annotated with eight semantic tactical labels. These labels are then checked against physics-derived scores, fused in a label-wise manner, and used to train a lightweight surrogate reward network. During PPO training, the surrogate provides semantic scores from state features without invoking the VLM online.

The main method, denoted PPO-PVVL-SR, combines the surrogate semantic reward, label-wise physical verification, potential-based shaping, and a mixed situation curriculum. The curriculum includes offensive, neutral, defensive, and random initial situations to address the reachability issue observed in the default initial distribution. The final Stage13 confirmation compares PPO-PVVL-SR against PPO-Phys and PPO-Cur using three random seeds and four evaluation protocols: default, offensive, mixed, and unseen. Stage13 continues from Stage12 300k actor/critic weights for 200k additional steps; optimizer state is not restored because the original runner does not save PPO optimizer checkpoints.
"""


def semantic_label_quality_text(data):
    f = data["fusion_summary"]
    return f"""
The offline Qwen2.5-VL teacher provides useful spatial and geometric semantic information, but the raw VLM scores are not reliable enough to be used directly as rewards. This is especially visible for energy-related and threat-related labels, where the VLM must infer numeric or dynamical quantities from a rendered diagram. On the formal100 semantic dataset, the raw VLM overall MAE is {f['overall_vlm_mae']:.4f}. After physical verification and label-wise fusion, the fused overall MAE decreases to {f['overall_fused_mae']:.4f}. The number of high-difference cases is reduced from {f['high_diff_cases_before_fusion']}/{f['num_samples']} to {f['high_diff_cases_after_fusion']}/{f['num_samples']}.

This result supports the central design choice of using the VLM as a semantic teacher rather than as an unconstrained reward oracle. Geometry-related labels are allowed to preserve VLM contributions, while energy labels are dominated by physical computation and threat labels use physical override rules. The resulting fused labels are more suitable as training targets for a surrogate semantic reward model. The interpretation is therefore not that VLM scores replace physical rules, but that VLM semantic recognition can be made useful when filtered through physics-based consistency checks.
"""


def surrogate_reward_text(data):
    c = data["cache_coverage"]
    v1 = data["surrogate_v1_summary"]
    v2 = data["surrogate_v2_summary"]
    b = data["surrogate_benchmark"]
    return f"""
Exact cached-VLM lookup is not sufficient for PPO training because the simulator state space is continuous. In the Stage7 coverage diagnosis, the exact cache hit rate during random rollout is {c['exact_cache_hit_rate']:.4f}, and the fallback-to-physical rate is {c['fallback_physical_rate']:.4f}. This means that a PPO policy would rarely reuse exactly the same rendered image hash encountered during offline labeling. The surrogate semantic reward network is therefore necessary to generalize fused semantic labels from a finite offline dataset to online simulator states.

The surrogate is a lightweight MLP that maps compact air-combat state features to the eight fused semantic label scores. The first surrogate version reaches an overall MAE of {v1['overall_mae']:.4f}, but still has a high-difference ratio of {100*v1['high_diff_ratio']:.1f}%. After targeted geometry and boundary augmentation, surrogate v2 reduces the overall MAE to {v2['overall_mae']:.4f} and the high-difference ratio to {100*v2['high_diff_ratio']:.1f}%. It is also fast enough for online reward shaping: CPU inference averages {b['avg_inference_time_sec']:.2e} s per query, several orders of magnitude faster than the offline Qwen reference time. Thus, the deployed PPO loop depends only on the surrogate and physical verification modules, not on the VLM.
"""


def main_results_text(data):
    s = data["stage13"]
    p = find_group(s, "mixed_eval", "PPO_surrogate_v2_then_fusion_with_situation_curriculum")
    phys = find_group(s, "mixed_eval", "PPO_physical")
    cur = find_group(s, "mixed_eval", "PPO_original_with_situation_curriculum")
    return f"""
In the Stage13 500k mixed evaluation, PPO-PVVL-SR achieves the highest win rate among the three core methods. Its mixed win rate is {fmt_mean_std(metric_mean(p, 'win_rate'), metric_std(p, 'win_rate'))}, compared with {fmt_mean_std(metric_mean(cur, 'win_rate'), metric_std(cur, 'win_rate'))} for PPO-Cur and {fmt_mean_std(metric_mean(phys, 'win_rate'), metric_std(phys, 'win_rate'))} for PPO-Phys. PPO-PVVL-SR also shows the lowest draw rate among the three methods and the highest attack-window ratio, suggesting that the semantic reward model helps reduce conservative stalemate behavior and maintain more offensive tactical opportunities under the mixed curriculum distribution.

The result should be interpreted carefully. PPO-Phys remains a strong baseline, particularly for threat control. In mixed evaluation, PPO-PVVL-SR has enemy-threat exposure of {fmt_mean_std(metric_mean(p, 'enemy_threat_exposure'), metric_std(p, 'enemy_threat_exposure'))}, while PPO-Phys reaches a lower value of {fmt_mean_std(metric_mean(phys, 'enemy_threat_exposure'), metric_std(phys, 'enemy_threat_exposure'))}. Therefore, the proposed method is not uniformly superior across all metrics. Its strongest evidence is the combination of higher mixed win rate, lower draw tendency, and stronger attack-window occupancy, while physical shaping remains highly competitive for safety-related threat reduction.
"""


def offensive_unseen_text(data):
    s = data["stage13"]
    p_off = find_group(s, "offensive_eval", "PPO_surrogate_v2_then_fusion_with_situation_curriculum")
    phys_off = find_group(s, "offensive_eval", "PPO_physical")
    p_un = find_group(s, "unseen_eval", "PPO_surrogate_v2_then_fusion_with_situation_curriculum")
    phys_un = find_group(s, "unseen_eval", "PPO_physical")
    return f"""
The offensive evaluation highlights the benefit of combining situation curriculum with semantic reward shaping. PPO-PVVL-SR obtains an offensive-evaluation win rate of {fmt_mean_std(metric_mean(p_off, 'win_rate'), metric_std(p_off, 'win_rate'))} and an attack-window ratio of {fmt_mean_std(metric_mean(p_off, 'effective_attack_window_time_ratio_mean'), metric_std(p_off, 'effective_attack_window_time_ratio_mean'))}. These values are higher than the corresponding PPO-Phys offensive values of {fmt_mean_std(metric_mean(phys_off, 'win_rate'), metric_std(phys_off, 'win_rate'))} and {fmt_mean_std(metric_mean(phys_off, 'effective_attack_window_time_ratio_mean'), metric_std(phys_off, 'effective_attack_window_time_ratio_mean'))}. This supports the claim that semantic reward shaping can help retain offensive geometry when the initial condition makes such geometry reachable.

The unseen evaluation is more mixed. PPO-Phys achieves a higher unseen win rate, {fmt_mean_std(metric_mean(phys_un, 'win_rate'), metric_std(phys_un, 'win_rate'))}, than PPO-PVVL-SR, which reaches {fmt_mean_std(metric_mean(p_un, 'win_rate'), metric_std(p_un, 'win_rate'))}. However, PPO-PVVL-SR has a higher unseen attack-window ratio and a lower neutral-stalemate ratio than PPO-Phys. These results suggest that PVVL-SR improves some tactical geometry indicators but does not dominate the physical baseline under distribution shift. The paper should therefore present unseen results as evidence of competitiveness and complementary strengths rather than as a clean win over physical shaping.
"""


def potential_text(data):
    st12 = data["stage12"]
    then_row = find_group(st12, "mixed_eval", "PPO_surrogate_v2_then_fusion_with_situation_curriculum")
    nop_row = find_group(st12, "mixed_eval", "PPO_surrogate_v2_no_potential_with_situation_curriculum")
    return f"""
The potential-based shaping ablation should be interpreted with nuance. The no-potential variant does not simply collapse on every tactical indicator. In Stage12 mixed evaluation, the no-potential variant has attack-window and tail-advantage ratios that can be comparable to or higher than the potential-based variant. However, its win rate is lower, {fmt_mean_std(metric_mean(nop_row, 'win_rate'), metric_std(nop_row, 'win_rate'))}, compared with {fmt_mean_std(metric_mean(then_row, 'win_rate'), metric_std(then_row, 'win_rate'))} for the potential-based PPO-PVVL-SR variant. Its enemy-threat exposure is also higher, {fmt_mean_std(metric_mean(nop_row, 'enemy_threat_exposure'), metric_std(nop_row, 'enemy_threat_exposure'))}, compared with {fmt_mean_std(metric_mean(then_row, 'enemy_threat_exposure'), metric_std(then_row, 'enemy_threat_exposure'))}.

This supports a constrained conclusion: potential-based shaping is useful for outcome quality, threat control, and numerical stability, but it should not be claimed to improve every tactical statistic uniformly. The no-potential variant can produce more aggressive geometric behavior in some cases, but that behavior is not necessarily safer or more successful. For TAES framing, the ablation is best used to argue that potential-based semantic shaping provides a more stable way to inject semantic rewards without simply adding dense rewards that may change policy incentives in undesirable ways.
"""


def dodge_text(data):
    rows = data["dodge"].get("preliminary_runs", [])
    pv = next((r for r in rows if r["group"] == "PPO_surrogate_v2_then_fusion"), None)
    s = load_json(pv.get("eval_summary", "")) if pv else {}
    return f"""
The DodgeMissile experiment is included only as a preliminary weapon-enabled validation. The goal is to check whether the reward-shaping wrapper, state extractor, physical metrics, and surrogate semantic reward can run in a DodgeMissile scenario without modifying the PPO policy architecture. The selected scenario is 1v1/DodgeMissile/Selfplay, which supports reset and step calls. ShootMissile scenarios are not used because they still have an action-shape compatibility issue that is outside the scope of the present NoWeapon study.

All three DodgeMissile 50k single-seed runs complete without NaN or reward-spike failures. PPO-PVVL-SR obtains return {fmt_num(s.get('return_ego_mean'))}, episode length {fmt_num(s.get('episode_length_mean'))}, survival time {fmt_num(s.get('survival_time_mean'))}, and defensive-escape ratio {fmt_num(s.get('defensive_escape_time_ratio_mean'))}. However, all methods still lose in the reported 50k evaluation, and missile warning, lock, and hit fields are not exposed in the available info dictionary. These results should therefore be described as interface compatibility and weak transfer signals only. They do not validate full missile-combat decision-making performance.
"""


def limitations_text():
    return """
Several limitations should be stated explicitly. First, the main benchmark is the 1v1 NoWeapon setting, which is useful for geometry and maneuvering analysis but is not equivalent to full missile combat. Second, PPO-Phys remains a strong baseline and has lower enemy-threat exposure than PPO-PVVL-SR in the Stage13 mixed and unseen evaluations. Third, the default initial distribution still rarely reaches clear attack-window conditions, so the situation curriculum is an important part of the method rather than a minor training detail. Fourth, the DodgeMissile validation is small-scale, single-seed, and preliminary; all methods still lose under the reported 50k training setting.

The Stage13 500k confirmation also has a reproducibility caveat: it resumes actor and critic weights from Stage12 300k checkpoints and trains for an additional 200k steps, but the PPO optimizer state is reset because the original runner does not save optimizer checkpoints. Finally, the VLM teacher is used only offline. This is a design advantage for deployability, but it also means that the deployed system depends on the quality and coverage of the surrogate semantic reward network rather than on direct VLM reasoning at decision time.
"""


def write_appendix(out, paths):
    lines = [
        "# Appendix Material Index",
        "",
        "- Stage11 metric sanity check: `{}`".format(rel(paths["stage11_metric_sanity"])),
        "- Stage11 threshold analysis: `{}`".format(rel(paths["stage11_threshold_report"])),
        "- Stage11 rollout geometry distribution: `{}`".format(rel(paths["stage11_geometry_report"])),
        "- Stage12 ablation table: `{}`".format(rel(paths["stage12_ablation"])),
        "- Weapon scenario availability: `{}`".format(rel(paths["stage12_weapon"])),
        "",
        "These materials support metric validity, reachability diagnosis, curriculum motivation, and weapon-scenario boundary conditions.",
    ]
    (out / "appendix_material" / "appendix_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for key in ["stage11_metric_sanity", "stage11_threshold_report", "stage11_geometry_report", "stage12_ablation", "stage12_weapon"]:
        src = paths[key]
        if Path(src).exists():
            shutil.copyfile(src, out / "appendix_material" / Path(src).name)


def write_reproducibility(out, paths):
    lines = [
        "# Reproducibility Checklist",
        "",
        "- Code commit/date: placeholder; package generated on 2026-05-30.",
        "- Environment: Windows, Python environment used by LAG-master experiments.",
        "- VLM model path: `models/Qwen2.5-VL-7B-Instruct`.",
        "- Offline VLM teacher: Qwen2.5-VL-7B-Instruct.",
        "- Surrogate model: `scripts/results/surrogate_models/stage9_surrogate_v2`.",
        "- Main result directory: `scripts/results/stage13_500k_confirmation`.",
        "- DodgeMissile result directory: `scripts/results/stage13_dodgemissile_preliminary`.",
        "- Random seeds: Stage13 main results use seeds 0, 1, and 2.",
        "- Main config: `lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_mixed_curriculum_1v1.yaml`.",
        "- Evaluation configs: default, offensive, mixed, and unseen curriculum evaluation YAML files under `lag_extensions/reward_shaping/configs/`.",
        "- Training command source: Stage13 `command.txt` files under each seed directory.",
        "- Evaluation outputs: `summary.json` and `episodes.jsonl` under each Stage13 seed/eval directory.",
        "- Figure generation command: `python tools/generate_taes_paper_package.py`.",
        "- Consistency check command: `python tools/check_taes_result_consistency.py`.",
        "- Known caveat: Stage13 500k continuation restores actor/critic weights only; PPO optimizer state is reset.",
        "",
        "## Files Required to Reproduce Tables",
        "",
        f"- `{rel(paths['stage13_summary'])}`",
        f"- `{rel(paths['stage12_summary'])}`",
        f"- `{rel(paths['stage7_fusion_summary'])}`",
        f"- `{rel(paths['stage7_fusion_per_label'])}`",
        f"- `{rel(paths['stage7_cache_coverage'])}`",
        f"- `{rel(paths['dodge_summary'])}`",
        "",
        "## Files Required to Reproduce Figures",
        "",
        "- Source CSV files under `scripts/results/taes_paper_package/figure_sources/data/`.",
        "- Figure script snapshot under `scripts/results/taes_paper_package/figure_sources/generate_taes_figures.py`.",
    ]
    (out / "reproducibility_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(out, data, table_paths, figure_paths):
    lines = [
        "# TAES Experiment Package Report",
        "",
        "## Overview",
        "",
        "This package reorganizes the completed Stage6-Stage13 results into tables, figures, captions, analysis text, and reproducibility notes suitable for a TAES manuscript draft.",
        "",
        "## Recommended Title",
        "",
        FINAL_TITLE,
        "",
        "## Proposed Method Name",
        "",
        "PVVL-SR: Physics-Verified Vision-Language Semantic Reward Shaping.",
        "",
        "## Key Claims Supported by Experiments",
        "",
        "- Physics-verified label-wise fusion dramatically improves raw VLM label reliability.",
        "- Surrogate v2 is accurate enough and fast enough for online reward shaping without online VLM calls.",
        "- PPO-PVVL-SR improves Stage13 mixed-evaluation win rate and attack-window occupancy over PPO-Cur and PPO-Phys.",
        "- PPO-Phys remains stronger in enemy-threat exposure, so the result is not universal dominance.",
        "- DodgeMissile results are preliminary compatibility and weak-transfer evidence only.",
        "",
        "## Main Numerical Evidence",
        "",
        "- Stage13 mixed table: `{}`".format(rel(table_paths["table1"]["md"])),
        "- Stage13 offensive table: `{}`".format(rel(table_paths["table2"]["md"])),
        "- Stage13 unseen table: `{}`".format(rel(table_paths["table3"]["md"])),
        "- Ablation table: `{}`".format(rel(table_paths["table4"]["md"])),
        "- DodgeMissile table: `{}`".format(rel(table_paths["table5"]["md"])),
        "",
        "## Figures Generated",
    ]
    for key, files in figure_paths.items():
        lines.append(f"- {key}: `{rel(files[0])}`")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- NoWeapon results do not validate full missile-combat performance.",
        "- Physical shaping remains a strong baseline, especially for threat exposure.",
        "- Stage13 continuation resets PPO optimizer state.",
        "- DodgeMissile validation is 50k, single-seed, and preliminary.",
        "",
        "## Recommended Paper Narrative",
        "",
        "Frame the paper as a physically grounded semantic reward modeling framework. Lead with label reliability, surrogate deployability, curriculum reachability, and mixed/offensive tactical gains. Treat PPO-Phys as a strong baseline and explicitly discuss threat-exposure tradeoffs.",
        "",
        "## Remaining TODO Before Manuscript Writing",
        "",
        "- Decide whether to include Stage13 500k as the main table or include Stage12 300k as an additional trend table.",
        "- Manually inspect figure sizing in the target IEEE template.",
        "- Add exact hardware/software versions if required by the final reproducibility section.",
        "- Decide whether DodgeMissile preliminary belongs in the main paper or appendix.",
    ])
    (out / "taes_experiment_package_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(out, paths, table_paths, figure_paths):
    manifest = {
        "title": FINAL_TITLE,
        "generated_outputs": {
            "tables": {k: {kk: str(vv) for kk, vv in bundle.items()} for k, bundle in table_paths.items()},
            "figures": {k: [str(p) for p in files] for k, files in figure_paths.items()},
        },
        "source_paths": {k: str(v) for k, v in paths.items()},
    }
    (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def copy_figure_script(out):
    src = Path(__file__)
    dst = out / "figure_sources" / "generate_taes_figures.py"
    shutil.copyfile(src, dst)


def find_group(summary, eval_type, group):
    for row in summary.get("aggregate_by_eval", {}).get(eval_type, []):
        if row.get("group") == group:
            return row
    raise KeyError(f"Missing group {group} in {eval_type}")


def metric_mean(row, metric):
    if row is None:
        return None
    if metric in row:
        return row.get(metric)
    return row.get(metric + "_mean")


def metric_std(row, metric):
    if row is None:
        return None
    return row.get(metric + "_std", 0.0)


def fmt_mean_std(mean, std):
    if mean is None:
        return "n/a"
    return f"{float(mean):.4f} +/- {float(std or 0.0):.4f}"


def fmt_num(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def fnum(value):
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def load_json(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_md_table(path, caption, rows, headers):
    lines = [f"# {caption}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path, caption, rows, headers):
    cols = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{" + latex_escape(caption) + "}",
        "\\begin{tabular}{" + cols + "}",
        "\\hline",
        " & ".join(latex_escape(h) for h in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(str(row.get(h, ""))) for h in headers) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text):
    return (text.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#"))


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    main()
