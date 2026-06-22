import csv
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

from .config import LABEL_NAMES, load_reward_shaping_config
from .semantic_reward import LabelWiseSemanticFusion, PhysicalVerifier
from .vlm_cache import VLMScoreCache


def evaluate_vlm_labels(
    dataset_dir: str,
    cache_path: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    hard_threshold: float = 0.5,
    consistency_threshold: float = 0.3,
    high_diff_threshold: float = 0.4,
) -> Dict:
    output_dir = output_dir or os.path.join(dataset_dir, "vlm_eval")
    os.makedirs(output_dir, exist_ok=True)
    config = load_reward_shaping_config(config_path) if config_path else load_reward_shaping_config()
    cache = VLMScoreCache(cache_path, enabled=True)
    reward_config = config.get("reward_shaping", {})
    verifier = PhysicalVerifier(
        mode=reward_config.get("verification_mode", "soft_gate"),
        consistency_threshold=float(reward_config.get("consistency_threshold", 0.3)),
        tau=float(reward_config.get("tau", 0.2)),
        low_confidence_value=float(reward_config.get("low_confidence_value", 0.1)),
    )
    fusion_model = LabelWiseSemanticFusion(config.get("label_fusion", {}))
    rows = []
    failed_cases = []
    low_consistency = []
    raw_high_diff_cases = []
    fused_high_diff_cases = []

    for sample_dir in _sample_dirs(dataset_dir):
        metadata = _read_json(os.path.join(sample_dir, "metadata.json"))
        physical_labels = _read_json(os.path.join(sample_dir, "physical_labels.json"))
        entry = cache.get(metadata.get("state_hash")) or cache.get(metadata.get("image_hash")) or cache.get(metadata.get("image_id"))
        if entry is None:
            failed_cases.append({"sample_dir": sample_dir, "error": "cache_miss"})
            continue
        if not entry.get("valid", False):
            failed_cases.append({"sample_dir": sample_dir, "error": entry.get("error"), "raw_output": entry.get("raw_output")})
        physical_scores = {name: float(physical_labels[name]["score"]) for name in LABEL_NAMES}
        physical_hard = {name: bool(physical_labels[name]["hard"]) for name in LABEL_NAMES}
        semantic_scores = {name: float(entry.get("scores", {}).get(name, np.nan)) for name in LABEL_NAMES}
        verification = verifier.verify(physical_scores, semantic_scores)
        verification = fusion_model.fuse(verification)
        max_err = max(abs(semantic_scores[name] - physical_scores[name]) for name in LABEL_NAMES if not math.isnan(semantic_scores[name]))
        if max_err > consistency_threshold:
            low_consistency.append({
                "sample_dir": sample_dir,
                "image_path": metadata.get("image_path"),
                "max_abs_error": max_err,
                "scores": semantic_scores,
                "physical_scores": physical_scores,
            })
        raw_high = _high_diff_labels(semantic_scores, physical_scores, high_diff_threshold)
        fused_high = _high_diff_labels(verification["fused_scores"], physical_scores, high_diff_threshold)
        if raw_high:
            raw_high_diff_cases.append({
                "sample_dir": sample_dir,
                "image_path": metadata.get("image_path"),
                "high_diff_labels": raw_high,
                "scores": semantic_scores,
                "physical_scores": physical_scores,
            })
        if fused_high:
            fused_high_diff_cases.append({
                "sample_dir": sample_dir,
                "image_path": metadata.get("image_path"),
                "high_diff_labels": fused_high,
                "fused_scores": verification["fused_scores"],
                "physical_scores": physical_scores,
            })
        rows.append({
            "sample_dir": sample_dir,
            "metadata": metadata,
            "entry": entry,
            "physical_scores": physical_scores,
            "physical_hard": physical_hard,
            "semantic_scores": semantic_scores,
            "verification": verification,
        })

    summary, per_label = _compute_metrics(
        rows,
        failed_cases,
        low_consistency,
        raw_high_diff_cases,
        fused_high_diff_cases,
        cache,
        hard_threshold,
    )
    _write_json(os.path.join(output_dir, "summary.json"), summary)
    _write_csv(os.path.join(output_dir, "per_label_metrics.csv"), per_label)
    _write_jsonl(os.path.join(output_dir, "failed_cases.jsonl"), failed_cases)
    _write_jsonl(os.path.join(output_dir, "low_consistency_cases.jsonl"), low_consistency)
    _write_jsonl(os.path.join(output_dir, "raw_vlm_high_diff_cases.jsonl"), raw_high_diff_cases)
    _write_jsonl(os.path.join(output_dir, "fused_high_diff_cases.jsonl"), fused_high_diff_cases)
    return summary


def _compute_metrics(
    rows: List[Dict],
    failed_cases: List[Dict],
    low_consistency: List[Dict],
    raw_high_diff_cases: List[Dict],
    fused_high_diff_cases: List[Dict],
    cache,
    hard_threshold: float,
):
    per_label = []
    valid_count = sum(1 for row in rows if row["entry"].get("valid", False))
    fallback_count = sum(1 for row in rows if row["entry"].get("fallback_reason"))
    inference_times = [float(row["entry"].get("inference_time_sec", 0.0)) for row in rows if row["entry"].get("inference_time_sec") is not None]
    markdown_wrapped_count = sum(1 for row in rows if "```" in str(row["entry"].get("raw_output", "")))
    output_truncated_count = sum(
        1
        for row in rows
        if (row["entry"].get("generation_info") or {}).get("output_truncated")
    )
    generation_times = [
        float(((row["entry"].get("generation_info") or {}).get("timings") or {}).get("generate_time"))
        for row in rows
        if ((row["entry"].get("generation_info") or {}).get("timings") or {}).get("generate_time") is not None
    ]
    confidence_values = [
        np.mean(list(row["verification"]["confidence_scores"].values()))
        for row in rows
    ]

    for name in LABEL_NAMES:
        y_true = np.array([row["physical_scores"][name] for row in rows], dtype=np.float64)
        y_pred = np.array([row["semantic_scores"][name] for row in rows], dtype=np.float64)
        y_fused = np.array([row["verification"]["fused_scores"][name] for row in rows], dtype=np.float64)
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_fused)
        if mask.sum() == 0:
            metrics = _empty_label_metrics(name)
        else:
            y_t = y_true[mask]
            y_p = y_pred[mask]
            y_f = y_fused[mask]
            hard_true = np.array([row["physical_hard"][name] for row in rows], dtype=bool)[mask]
            hard_pred = y_p >= hard_threshold
            hard_fused = y_f >= hard_threshold
            override_values = [
                bool(row["verification"]["override_info"][name]["override_triggered"])
                for row in rows
            ]
            metrics = {
                "label": name,
                "mae": float(np.mean(np.abs(y_p - y_t))),
                "vlm_mae": float(np.mean(np.abs(y_p - y_t))),
                "fused_mae": float(np.mean(np.abs(y_f - y_t))),
                "vlm_rmse": float(np.sqrt(np.mean((y_p - y_t) ** 2))),
                "fused_rmse": float(np.sqrt(np.mean((y_f - y_t) ** 2))),
                "vlm_pearson": float(_pearson(y_t, y_p)),
                "fused_pearson": float(_pearson(y_t, y_f)),
                "vlm_spearman": float(_spearman(y_t, y_p)),
                "fused_spearman": float(_spearman(y_t, y_f)),
                "vlm_hard_accuracy": float(np.mean(hard_pred == hard_true)),
                "fused_hard_accuracy": float(np.mean(hard_fused == hard_true)),
                "vlm_f1": float(_f1(hard_true, hard_pred)),
                "fused_f1": float(_f1(hard_true, hard_fused)),
                "mean_confidence": float(np.mean([row["verification"]["confidence_scores"][name] for row in rows])),
                "mean_fused_score": float(np.mean(y_f)),
                "override_rate": float(np.mean(override_values)) if override_values else 0.0,
                "num_samples": int(mask.sum()),
            }
        per_label.append(metrics)

    vlm_maes = [row["vlm_mae"] for row in per_label if np.isfinite(row.get("vlm_mae", np.nan))]
    fused_maes = [row["fused_mae"] for row in per_label if np.isfinite(row.get("fused_mae", np.nan))]
    summary = {
        "num_samples": len(rows),
        "json_valid_rate": valid_count / len(rows) if rows else 0.0,
        "invalid_output_count": len(failed_cases),
        "fallback_count": fallback_count,
        "overall_consistency_rate": 1.0 - len(low_consistency) / len(rows) if rows else 0.0,
        "average_confidence_after_verification": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "average_fused_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "average_inference_time_per_image": float(np.mean(inference_times)) if inference_times else 0.0,
        "p50_inference_time_per_image": _percentile(inference_times, 50),
        "p90_inference_time_per_image": _percentile(inference_times, 90),
        "p95_inference_time_per_image": _percentile(inference_times, 95),
        "average_generate_time_per_image": float(np.mean(generation_times)) if generation_times else 0.0,
        "markdown_wrapped_count": markdown_wrapped_count,
        "output_truncated_count": output_truncated_count,
        "cache_hit_rate": cache.stats()["hit_rate"],
        "per_label_mae_mean": float(np.mean(vlm_maes)) if vlm_maes else 0.0,
        "overall_vlm_mae": float(np.mean(vlm_maes)) if vlm_maes else 0.0,
        "overall_fused_mae": float(np.mean(fused_maes)) if fused_maes else 0.0,
        "high_diff_cases_before_fusion": len(raw_high_diff_cases),
        "high_diff_cases_after_fusion": len(fused_high_diff_cases),
        "high_diff_ratio_before_fusion": len(raw_high_diff_cases) / len(rows) if rows else 0.0,
        "high_diff_ratio_after_fusion": len(fused_high_diff_cases) / len(rows) if rows else 0.0,
    }
    return summary, per_label


def _empty_label_metrics(name):
    return {
        "label": name,
        "mae": float("nan"),
        "vlm_mae": float("nan"),
        "fused_mae": float("nan"),
        "vlm_rmse": float("nan"),
        "fused_rmse": float("nan"),
        "vlm_pearson": float("nan"),
        "fused_pearson": float("nan"),
        "vlm_spearman": float("nan"),
        "fused_spearman": float("nan"),
        "vlm_hard_accuracy": float("nan"),
        "fused_hard_accuracy": float("nan"),
        "vlm_f1": float("nan"),
        "fused_f1": float("nan"),
        "mean_confidence": float("nan"),
        "mean_fused_score": float("nan"),
        "override_rate": float("nan"),
        "num_samples": 0,
    }


def _high_diff_labels(scores: Dict[str, float], physical_scores: Dict[str, float], threshold: float) -> Dict[str, float]:
    labels = {}
    for name in LABEL_NAMES:
        value = scores.get(name)
        if value is None or not np.isfinite(value):
            continue
        diff = abs(float(value) - float(physical_scores.get(name, 0.0)))
        if diff > threshold:
            labels[name] = float(diff)
    return labels


def _pearson(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def _spearman(a, b):
    return _pearson(_rank(a), _rank(b))


def _rank(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _f1(true, pred):
    tp = np.sum(true & pred)
    fp = np.sum(~true & pred)
    fn = np.sum(true & ~pred)
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def _percentile(values, percentile):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * percentile / 100.0
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def _sample_dirs(dataset_dir):
    return [
        os.path.join(dataset_dir, name)
        for name in sorted(os.listdir(dataset_dir))
        if name.startswith("sample_") and os.path.isdir(os.path.join(dataset_dir, name))
    ]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
