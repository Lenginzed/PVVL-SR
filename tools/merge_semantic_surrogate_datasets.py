#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Merge semantic surrogate JSONL datasets while preserving provenance.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--source-names", nargs="+", default=None)
    parser.add_argument("--output", default="scripts/results/semantic_surrogate_dataset/stage8_surrogate_v1_formal100_geometry80.jsonl")
    args = parser.parse_args()
    summary = merge(args.inputs, args.output, args.source_names)
    print(json.dumps(summary, indent=2, sort_keys=True))


def merge(inputs, output, source_names=None):
    if source_names is not None and len(source_names) != len(inputs):
        raise ValueError("--source-names length must match --inputs length")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    counts = {}
    total = 0
    with open(output, "w", encoding="utf-8") as out:
        for i, path in enumerate(inputs):
            source = source_names[i] if source_names else os.path.splitext(os.path.basename(path))[0]
            counts[source] = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    metadata = dict(item.get("metadata", {}))
                    metadata["source_dataset"] = source
                    metadata.setdefault("synthetic", bool(metadata.get("synthetic", False)))
                    item["metadata"] = metadata
                    out.write(json.dumps(item, sort_keys=True) + "\n")
                    counts[source] += 1
                    total += 1
    summary = {
        "output": os.path.abspath(output),
        "total_rows": total,
        "source_counts": counts,
        "inputs": [os.path.abspath(path) for path in inputs],
    }
    with open(os.path.splitext(output)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


if __name__ == "__main__":
    main()
