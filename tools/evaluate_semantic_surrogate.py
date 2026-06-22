#!/usr/bin/env python
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import LABEL_NAMES
from lag_extensions.reward_shaping.logger import to_jsonable
from lag_extensions.reward_shaping.semantic_surrogate import SemanticSurrogatePredictor, load_surrogate_jsonl


def main():
    parser = argparse.ArgumentParser(description="Evaluate a semantic surrogate model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="scripts/results/surrogate_eval/stage8_formal100_smoke")
    parser.add_argument("--target-type", default=None, choices=[None, "raw_vlm_scores", "verified_scores", "fused_scores"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--high-diff-threshold", type=float, default=0.3)
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def evaluate(args):
    predictor = SemanticSurrogatePredictor(args.model_dir, device=args.device)
    target_type = args.target_type or predictor.config.get("target_type", "fused_scores")
    rows = load_surrogate_jsonl(args.dataset, target_type)
    os.makedirs(args.output_dir, exist_ok=True)
    predictions = []
    targets = []
    case_rows = []
    high_diff_cases = []
    worst = []

    for row in rows:
        pred = predictor.predict_features(row["state_features"])
        target = {name: float(row[target_type].get(name, 0.0)) for name in LABEL_NAMES}
        diffs = {name: abs(float(pred[name]) - float(target[name])) for name in LABEL_NAMES}
        max_label = max(diffs, key=diffs.get)
        case = {
            "sample_id": row.get("sample_id"),
            "metadata": row.get("metadata", {}),
            "prediction": pred,
            "target": target,
            "abs_diff": diffs,
            "max_error": float(diffs[max_label]),
            "max_error_label": max_label,
        }
        case_rows.append(case)
        if case["max_error"] > args.high_diff_threshold:
            high_diff_cases.append(case)
        worst.append(case)
        predictions.append([pred[name] for name in LABEL_NAMES])
        targets.append([target[name] for name in LABEL_NAMES])

    pred_arr = np.asarray(predictions, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    err = pred_arr - target_arr
    per_label = []
    for i, name in enumerate(LABEL_NAMES):
        per_label.append({
            "label": name,
            "mae": float(np.mean(np.abs(err[:, i]))) if len(rows) else 0.0,
            "mse": float(np.mean(err[:, i] ** 2)) if len(rows) else 0.0,
            "target_mean": float(np.mean(target_arr[:, i])) if len(rows) else 0.0,
            "prediction_mean": float(np.mean(pred_arr[:, i])) if len(rows) else 0.0,
        })
    summary = {
        "dataset": os.path.abspath(args.dataset),
        "model_dir": os.path.abspath(args.model_dir),
        "target_type": target_type,
        "num_rows": len(rows),
        "overall_mae": float(np.mean(np.abs(err))) if len(rows) else 0.0,
        "overall_mse": float(np.mean(err ** 2)) if len(rows) else 0.0,
        "max_error": float(np.max(np.abs(err))) if len(rows) else 0.0,
        "high_diff_threshold": args.high_diff_threshold,
        "high_diff_cases": len(high_diff_cases),
        "high_diff_ratio": len(high_diff_cases) / len(rows) if rows else 0.0,
        "prediction_min": float(np.min(pred_arr)) if len(rows) else 0.0,
        "prediction_max": float(np.max(pred_arr)) if len(rows) else 0.0,
        "target_min": float(np.min(target_arr)) if len(rows) else 0.0,
        "target_max": float(np.max(target_arr)) if len(rows) else 0.0,
    }
    _write_json(os.path.join(args.output_dir, "summary.json"), summary)
    _write_csv(os.path.join(args.output_dir, "per_label_metrics.csv"), per_label)
    _write_jsonl(os.path.join(args.output_dir, "high_diff_cases.jsonl"), high_diff_cases)
    worst = sorted(worst, key=lambda item: item["max_error"], reverse=True)[:50]
    _write_jsonl(os.path.join(args.output_dir, "worst_cases.jsonl"), worst)
    _write_csv(os.path.join(args.output_dir, "predictions.csv"), _prediction_csv_rows(case_rows))
    return summary


def _prediction_csv_rows(cases):
    rows = []
    for case in cases:
        row = {
            "sample_id": case.get("sample_id"),
            "situation_type": (case.get("metadata") or {}).get("situation_type"),
            "max_error": case["max_error"],
            "max_error_label": case["max_error_label"],
        }
        for name in LABEL_NAMES:
            row[name + "_pred"] = case["prediction"][name]
            row[name + "_target"] = case["target"][name]
            row[name + "_abs_diff"] = case["abs_diff"][name]
        rows.append(row)
    return rows


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
