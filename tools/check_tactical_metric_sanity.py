#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import load_reward_shaping_config
from tools.stage11_common import ensure_dir, evaluate_synthetic_state, make_pair_state, write_json


def main():
    parser = argparse.ArgumentParser(description="Synthetic sanity check for tactical air-combat metrics and labels.")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml")
    parser.add_argument("--output-dir", default="scripts/results/stage11_metric_sanity")
    args = parser.parse_args()

    config = load_reward_shaping_config(args.config)
    thresholds = config.get("thresholds", {})
    out_dir = ensure_dir(args.output_dir)

    cases = build_cases()
    results = []
    all_passed = True
    for case in cases:
        result = evaluate_synthetic_state(case["state"], thresholds)
        passed, checks = evaluate_expectations(result, case["expectations"])
        all_passed = all_passed and passed
        results.append({
            "name": case["name"],
            "description": case["description"],
            "expected": case["expectations"],
            "passed": passed,
            "checks": checks,
            "metrics": result["metrics"],
            "hard_labels": result["hard"],
            "soft_scores": result["scores"],
        })

    summary = {
        "all_passed": all_passed,
        "case_count": len(results),
        "passed_count": sum(1 for row in results if row["passed"]),
        "config": os.path.abspath(args.config),
    }
    write_json(os.path.join(out_dir, "sanity_cases.json"), {"summary": summary, "cases": results})
    write_report(os.path.join(out_dir, "sanity_report.md"), summary, results)
    print("PASS" if all_passed else "FAIL")
    print(os.path.abspath(os.path.join(out_dir, "sanity_report.md")))


def build_cases():
    z = 6000.0
    return [
        {
            "name": "ego_tail_advantage_attack_window",
            "description": "Ego is clearly behind the enemy, both heading north, ego is faster and aiming at enemy.",
            "state": make_pair_state([0, 0, z], [3000, 0, z], 0, 0, ego_speed=310, enemy_speed=240, altitude_m=z),
            "expectations": {
                "ego_tail_advantage": ("high", 0.15),
                "effective_attack_window": ("high", 0.25),
                "enemy_tail_threat": ("low", 0.05),
            },
        },
        {
            "name": "enemy_tail_threat",
            "description": "Enemy is clearly behind ego and aiming at ego.",
            "state": make_pair_state([3000, 0, z], [0, 0, z], 0, 0, ego_speed=240, enemy_speed=310, altitude_m=z),
            "expectations": {
                "enemy_tail_threat": ("high", 0.15),
                "enemy_missile_threat_zone": ("high", 0.25),
                "ego_tail_advantage": ("low", 0.05),
            },
        },
        {
            "name": "head_on",
            "description": "Head-on merge: both aircraft aim at each other but neither is behind the other.",
            "state": make_pair_state([0, 0, z], [4000, 0, z], 0, 180, ego_speed=260, enemy_speed=260, altitude_m=z),
            "expectations": {
                "ego_tail_advantage": ("low", 0.05),
                "enemy_tail_threat": ("low", 0.05),
                "enemy_missile_threat_zone": ("high", 0.25),
            },
        },
        {
            "name": "beam_crossing",
            "description": "Ego points at enemy, enemy crosses laterally; tail geometry should stay weak.",
            "state": make_pair_state([0, 0, z], [4000, 0, z], 0, 90, ego_speed=260, enemy_speed=260, altitude_m=z),
            "expectations": {
                "ego_tail_advantage": ("low", 0.05),
                "effective_attack_window": ("low", 0.05),
                "enemy_tail_threat": ("low", 0.05),
            },
        },
        {
            "name": "too_close_overshoot",
            "description": "Ego has tail geometry but distance is below d_attack_min, so attack window should be low.",
            "state": make_pair_state([0, 0, z], [200, 0, z], 0, 0, ego_speed=310, enemy_speed=240, altitude_m=z),
            "expectations": {
                "ego_tail_advantage": ("high", 0.5),
                "effective_attack_window": ("low", 0.05),
            },
        },
        {
            "name": "too_far",
            "description": "Ego has roughly correct geometry but distance exceeds the configured tail/attack ranges.",
            "state": make_pair_state([0, 0, z], [12000, 0, z], 0, 0, ego_speed=310, enemy_speed=240, altitude_m=z),
            "expectations": {
                "ego_tail_advantage": ("low", 0.05),
                "effective_attack_window": ("low", 0.05),
                "enemy_missile_threat_zone": ("low", 0.10),
            },
        },
    ]


def evaluate_expectations(result, expectations):
    checks = {}
    passed = True
    scores = result["scores"]
    hard = result["hard"]
    for name, (kind, threshold) in expectations.items():
        score = float(scores[name])
        if kind == "high":
            ok = bool(hard[name] or score >= threshold)
            message = "expected high; hard={} score={:.4f} threshold={:.4f}".format(hard[name], score, threshold)
        elif kind == "low":
            ok = bool((not hard[name]) and score <= threshold)
            message = "expected low; hard={} score={:.4f} threshold={:.4f}".format(hard[name], score, threshold)
        else:
            raise ValueError("Unknown expectation kind: {}".format(kind))
        checks[name] = {"passed": ok, "message": message}
        passed = passed and ok
    return passed, checks


def write_report(path, summary, results):
    lines = [
        "# Stage11 Tactical Metric Sanity Check",
        "",
        "- all_passed: `{}`".format(summary["all_passed"]),
        "- passed: `{}/{}`".format(summary["passed_count"], summary["case_count"]),
        "",
    ]
    for row in results:
        lines.extend([
            "## {}".format(row["name"]),
            "",
            row["description"],
            "",
            "- passed: `{}`".format(row["passed"]),
            "- distance_m: `{:.2f}`".format(row["metrics"]["distance"]),
            "- ego_aim_deg: `{:.2f}`".format(row["metrics"]["ego_aim_angle_deg"]),
            "- enemy_aim_deg: `{:.2f}`".format(row["metrics"]["enemy_aim_angle_deg"]),
            "- ego_tail_deg: `{:.2f}`".format(row["metrics"]["ego_tail_angle_deg"]),
            "- enemy_tail_deg: `{:.2f}`".format(row["metrics"]["enemy_tail_angle_deg"]),
            "- distance_rate_mps: `{:.2f}`".format(row["metrics"]["distance_rate"]),
            "",
            "| label | hard | score | check |",
            "|---|---:|---:|---|",
        ])
        for name, check in row["checks"].items():
            lines.append("| {} | {} | {:.4f} | {} |".format(
                name,
                row["hard_labels"][name],
                row["soft_scores"][name],
                "PASS" if check["passed"] else "FAIL",
            ))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
