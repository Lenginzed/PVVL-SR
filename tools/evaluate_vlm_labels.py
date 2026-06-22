#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.vlm_label_eval import evaluate_vlm_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="scripts/results/aircombat_semantic_dataset")
    parser.add_argument("--cache", type=str, default="scripts/results/vlm_cache/qwen2_5_vl_1v1_cache.jsonl")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--hard-threshold", type=float, default=0.5)
    parser.add_argument("--consistency-threshold", type=float, default=0.3)
    parser.add_argument("--high-diff-threshold", type=float, default=0.4)
    args = parser.parse_args()

    summary = evaluate_vlm_labels(
        dataset_dir=args.dataset_dir,
        cache_path=args.cache,
        output_dir=args.output_dir,
        config_path=args.config,
        hard_threshold=args.hard_threshold,
        consistency_threshold=args.consistency_threshold,
        high_diff_threshold=args.high_diff_threshold,
    )
    print(summary)


if __name__ == "__main__":
    main()
