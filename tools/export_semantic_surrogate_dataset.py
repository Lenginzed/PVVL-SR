#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import LABEL_NAMES, load_reward_shaping_config
from lag_extensions.reward_shaping.logger import to_jsonable
from lag_extensions.reward_shaping.semantic_cache_retrieval import FEATURE_NAMES, feature_vector_from_metrics, sample_dirs, write_json
from lag_extensions.reward_shaping.semantic_reward import LabelWiseSemanticFusion, PhysicalVerifier
from lag_extensions.reward_shaping.vlm_cache import VLMScoreCache


def main():
    parser = argparse.ArgumentParser(description="Export state-feature -> fused-score data for a future semantic surrogate.")
    parser.add_argument("--dataset-dir", default="scripts/results/aircombat_semantic_dataset_stage7_formal100")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml")
    parser.add_argument("--output", default="scripts/results/semantic_surrogate_dataset/stage7_formal100_surrogate.jsonl")
    parser.add_argument("--allow-missing-cache", action="store_true", default=False)
    args = parser.parse_args()
    summary = export_surrogate_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def export_surrogate_dataset(args):
    config = load_reward_shaping_config(args.config)
    verifier = PhysicalVerifier(
        mode=config.get("reward_shaping", {}).get("verification_mode", "soft_gate"),
        consistency_threshold=float(config.get("reward_shaping", {}).get("consistency_threshold", 0.3)),
        tau=float(config.get("reward_shaping", {}).get("tau", 0.2)),
        low_confidence_value=float(config.get("reward_shaping", {}).get("low_confidence_value", 0.1)),
    )
    fusion = LabelWiseSemanticFusion(config.get("label_fusion", {}))
    cache = VLMScoreCache(args.cache, enabled=True)
    output_path = _resolve_output_path(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    missing_cache = 0
    fallback_physical = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for sample_dir in sample_dirs(args.dataset_dir):
            metadata = _read_json(os.path.join(sample_dir, "metadata.json"))
            metrics = _read_json(os.path.join(sample_dir, "metrics.json"))
            state = _read_optional_json(os.path.join(sample_dir, "env_state_snapshot.json")) or {}
            physical_labels = _read_json(os.path.join(sample_dir, "physical_labels.json"))
            physical_scores = _labels_to_scores(physical_labels)

            entry = _find_cache_entry(cache, metadata)
            if entry is None:
                missing_cache += 1
                if not args.allow_missing_cache:
                    continue
                semantic_scores = dict(physical_scores)
                cache_source = "missing_cache_fallback_physical"
                fallback_physical += 1
            else:
                semantic_scores = _sanitize_scores(entry.get("scores", {}))
                cache_source = entry.get("source", "cached_vlm")
                if cache_source == "cached_vlm_fallback_physical":
                    fallback_physical += 1

            verification = verifier.verify(physical_scores, semantic_scores)
            verification = fusion.fuse(verification)
            row = {
                "sample_id": os.path.basename(sample_dir),
                "state_feature_names": FEATURE_NAMES,
                "state_features": feature_vector_from_metrics(metrics, state),
                "physical_scores": physical_scores,
                "raw_vlm_scores": semantic_scores,
                "verified_scores": verification["verified_scores"],
                "fused_scores": verification["fused_scores"],
                "confidence_scores": verification["confidence_scores"],
                "fusion_policy": verification["fusion_policy"],
                "override_info": verification["override_info"],
                "cache_source": cache_source,
                "metadata": {
                    "image_id": metadata.get("image_id"),
                    "state_hash": metadata.get("state_hash"),
                    "image_hash": metadata.get("image_hash"),
                    "situation_type": metadata.get("situation_type"),
                    "selected_by_label": metadata.get("selected_by_label"),
                    "synthetic": bool(metadata.get("synthetic", False)),
                    "synthetic_generation": metadata.get("synthetic_generation"),
                    "image_path": metadata.get("image_path"),
                    "sample_dir": os.path.abspath(sample_dir),
                },
            }
            out.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
            count += 1

    summary = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "cache_path": os.path.abspath(args.cache),
        "output": output_path,
        "num_rows": count,
        "missing_cache_count": missing_cache,
        "fallback_physical_count": fallback_physical,
        "feature_names": FEATURE_NAMES,
    }
    write_json(os.path.splitext(output_path)[0] + "_summary.json", summary)
    return summary


def _find_cache_entry(cache, metadata):
    for key in (metadata.get("state_hash"), metadata.get("image_hash"), metadata.get("image_id")):
        if not key:
            continue
        entry = cache.get(key)
        if entry is not None:
            return entry
    return None


def _labels_to_scores(labels):
    scores = {}
    for name in LABEL_NAMES:
        value = labels.get(name, 0.0)
        if isinstance(value, dict):
            value = value.get("score", 0.0)
        scores[name] = float(max(0.0, min(1.0, value)))
    return scores


def _sanitize_scores(scores):
    return {name: float(max(0.0, min(1.0, scores.get(name, 0.0)))) for name in LABEL_NAMES}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_optional_json(path):
    if not os.path.exists(path):
        return None
    return _read_json(path)


def _resolve_output_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


if __name__ == "__main__":
    main()
