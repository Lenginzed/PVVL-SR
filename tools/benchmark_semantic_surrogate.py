#!/usr/bin/env python
import argparse
import json
import os
import statistics
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.semantic_surrogate import SemanticSurrogatePredictor, load_surrogate_jsonl


def main():
    parser = argparse.ArgumentParser(description="Benchmark semantic surrogate inference latency.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="scripts/results/surrogate_benchmark/stage8_benchmark.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeat", type=int, default=1000)
    args = parser.parse_args()
    summary = benchmark(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def benchmark(args):
    rows = load_surrogate_jsonl(args.dataset, "fused_scores")
    if not rows:
        raise ValueError("No rows found in {}".format(args.dataset))
    predictor = SemanticSurrogatePredictor(args.model_dir, device=args.device)
    features = [row["state_features"] for row in rows]
    times = []
    start_all = time.time()
    for i in range(int(args.repeat)):
        feature = features[i % len(features)]
        start = time.perf_counter()
        predictor.predict_features(feature)
        times.append(time.perf_counter() - start)
    total = time.time() - start_all
    summary = {
        "dataset": os.path.abspath(args.dataset),
        "model_dir": os.path.abspath(args.model_dir),
        "device": args.device,
        "repeat": int(args.repeat),
        "total_time_sec": total,
        "avg_inference_time_sec": statistics.mean(times),
        "p50_inference_time_sec": _percentile(times, 50),
        "p90_inference_time_sec": _percentile(times, 90),
        "p95_inference_time_sec": _percentile(times, 95),
        "qwen_reference_time_sec": 27.75,
        "speedup_vs_qwen_reference": 27.75 / statistics.mean(times) if times else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def _percentile(values, percentile):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100.0
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


if __name__ == "__main__":
    main()
