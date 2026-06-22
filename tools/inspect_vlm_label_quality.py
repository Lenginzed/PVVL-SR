#!/usr/bin/env python
import argparse
import json
import os
import random
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import LABEL_NAMES
from lag_extensions.reward_shaping.config import load_reward_shaping_config
from lag_extensions.reward_shaping.semantic_reward import LabelWiseSemanticFusion, PhysicalVerifier
from lag_extensions.reward_shaping.vlm_cache import VLMScoreCache


def main():
    parser = argparse.ArgumentParser(description="Build an inspectable packet for VLM label quality.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", default="scripts/results/vlm_quality_inspection/stage5_panel")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--diff-threshold", type=float, default=0.4)
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/qwen2_5_vl_1v1.yaml")
    args = parser.parse_args()

    config = load_reward_shaping_config(args.config)
    verifier_cfg = config.get("reward_shaping", {})
    verifier = PhysicalVerifier(
        mode=verifier_cfg.get("verification_mode", "soft_gate"),
        consistency_threshold=float(verifier_cfg.get("consistency_threshold", 0.3)),
        tau=float(verifier_cfg.get("tau", 0.2)),
        low_confidence_value=float(verifier_cfg.get("low_confidence_value", 0.1)),
    )
    fusion_model = LabelWiseSemanticFusion(config.get("label_fusion", {}))
    cache = VLMScoreCache(args.cache, enabled=True)
    os.makedirs(args.output_dir, exist_ok=True)
    raw_high_dir = os.path.join(args.output_dir, "raw_vlm_high_diff_cases")
    fused_high_dir = os.path.join(args.output_dir, "fused_high_diff_cases")
    os.makedirs(raw_high_dir, exist_ok=True)
    os.makedirs(fused_high_dir, exist_ok=True)

    sample_dirs = _sample_dirs(args.dataset_dir)
    rng = random.Random(args.seed)
    selected = sample_dirs[:]
    rng.shuffle(selected)
    selected = selected[: min(args.num_samples, len(selected))]

    rows = []
    raw_high_count = 0
    fused_high_count = 0
    for sample_dir in selected:
        packet = _build_packet(sample_dir, cache, verifier, fusion_model, args.output_dir, args.diff_threshold)
        rows.append(packet)
        if packet["raw_vlm_high_diff"]:
            raw_high_count += 1
            target = os.path.join(raw_high_dir, packet["sample_id"])
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(packet["packet_dir"], target)
        if packet["fused_high_diff"]:
            fused_high_count += 1
            target = os.path.join(fused_high_dir, packet["sample_id"])
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(packet["packet_dir"], target)

    html_path = os.path.abspath(os.path.join(args.output_dir, "index.html"))
    md_path = os.path.abspath(os.path.join(args.output_dir, "index.md"))
    summary = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "cache_path": os.path.abspath(args.cache),
        "output_dir": os.path.abspath(args.output_dir),
        "num_selected": len(rows),
        "raw_vlm_high_diff_count": raw_high_count,
        "raw_vlm_high_diff_ratio": raw_high_count / len(rows) if rows else 0.0,
        "fused_high_diff_count": fused_high_count,
        "fused_high_diff_ratio": fused_high_count / len(rows) if rows else 0.0,
        "diff_threshold": args.diff_threshold,
        "html_path": html_path,
        "markdown_path": md_path,
    }
    _write_json(os.path.join(args.output_dir, "summary.json"), summary)
    _write_html(html_path, rows, summary)
    _write_markdown(md_path, rows, summary)
    print(summary)


def _build_packet(sample_dir, cache, verifier, fusion_model, output_dir, diff_threshold):
    sample_id = os.path.basename(sample_dir)
    metadata = _read_json(os.path.join(sample_dir, "metadata.json"))
    physical_labels = _read_json(os.path.join(sample_dir, "physical_labels.json"))
    metrics = _read_json(os.path.join(sample_dir, "metrics.json"))
    physical_scores = {name: float(physical_labels[name]["score"]) for name in LABEL_NAMES}
    entry = _lookup_cache(cache, metadata)
    vlm_scores = {name: float((entry or {}).get("scores", {}).get(name, 0.0)) for name in LABEL_NAMES}
    verification = verifier.verify(physical_scores, vlm_scores)
    verification = fusion_model.fuse(verification)
    diff = {name: abs(vlm_scores[name] - physical_scores[name]) for name in LABEL_NAMES}
    fused_diff = {name: abs(verification["fused_scores"][name] - physical_scores[name]) for name in LABEL_NAMES}
    high_labels = [name for name, value in diff.items() if value > diff_threshold]
    fused_high_labels = [name for name, value in fused_diff.items() if value > diff_threshold]
    packet_dir = os.path.abspath(os.path.join(output_dir, sample_id))
    os.makedirs(packet_dir, exist_ok=True)
    image_src = os.path.join(sample_dir, "image.png")
    image_dst = os.path.join(packet_dir, "image.png")
    if os.path.exists(image_src):
        shutil.copy2(image_src, image_dst)
    _write_json(os.path.join(packet_dir, "physical_labels.json"), physical_labels)
    _write_json(os.path.join(packet_dir, "vlm_scores.json"), vlm_scores)
    _write_json(os.path.join(packet_dir, "verified_scores.json"), verification["verified_scores"])
    _write_json(os.path.join(packet_dir, "fused_scores.json"), verification["fused_scores"])
    _write_json(os.path.join(packet_dir, "confidence_scores.json"), verification["confidence_scores"])
    _write_json(os.path.join(packet_dir, "fusion_policy.json"), verification["fusion_policy"])
    _write_json(os.path.join(packet_dir, "override_info.json"), verification["override_info"])
    _write_json(os.path.join(packet_dir, "metrics.json"), metrics)
    _write_json(os.path.join(packet_dir, "diff.json"), diff)
    _write_json(os.path.join(packet_dir, "fused_diff.json"), fused_diff)
    _write_json(os.path.join(packet_dir, "cache_entry.json"), entry or {})
    return {
        "sample_id": sample_id,
        "packet_dir": packet_dir,
        "image_rel": os.path.join(sample_id, "image.png").replace("\\", "/"),
        "valid": bool((entry or {}).get("valid", False)),
        "markdown_wrapped": "```" in ((entry or {}).get("raw_output") or ""),
        "raw_output": (entry or {}).get("raw_output", ""),
        "source": (entry or {}).get("source"),
        "physical_scores": physical_scores,
        "vlm_scores": vlm_scores,
        "confidence_scores": verification["confidence_scores"],
        "verified_scores": verification["verified_scores"],
        "fused_scores": verification["fused_scores"],
        "fusion_policy": verification["fusion_policy"],
        "override_info": verification["override_info"],
        "diff": diff,
        "fused_diff": fused_diff,
        "high_labels": high_labels,
        "fused_high_labels": fused_high_labels,
        "raw_vlm_high_diff": bool(high_labels),
        "fused_high_diff": bool(fused_high_labels),
    }


def _lookup_cache(cache, metadata):
    keys = [
        metadata.get("cache_key"),
        metadata.get("state_hash"),
        metadata.get("image_hash"),
        metadata.get("image_id"),
    ]
    for key in keys:
        if not key:
            continue
        entry = cache.get(key)
        if entry is not None:
            return entry
    return None


def _write_html(path, rows, summary):
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>VLM Label Quality</title>",
        "<style>body{font-family:Arial,sans-serif;margin:20px} table{border-collapse:collapse;width:100%;margin-bottom:28px} td,th{border:1px solid #ddd;padding:4px;font-size:12px} .high{background:#ffd7d7;font-weight:bold} img{max-width:320px;border:1px solid #ccc} pre{white-space:pre-wrap;max-height:180px;overflow:auto;background:#f5f5f5;padding:8px}</style>",
        "</head><body>",
        "<h1>VLM Label Quality Inspection</h1>",
        "<pre>{}</pre>".format(_escape(json.dumps(summary, indent=2, sort_keys=True))),
    ]
    for row in rows:
        parts.append("<h2>{} raw_high={} fused_high={}</h2>".format(row["sample_id"], row["raw_vlm_high_diff"], row["fused_high_diff"]))
        parts.append("<img src='{}'>".format(row["image_rel"]))
        parts.append("<table><tr><th>label</th><th>physical</th><th>vlm</th><th>raw diff</th><th>confidence</th><th>verified</th><th>fused</th><th>fused diff</th><th>policy</th><th>override</th><th>reason</th></tr>")
        for name in LABEL_NAMES:
            cls = " class='high'" if row["diff"][name] > 0.4 or row["fused_diff"][name] > 0.4 else ""
            override = row["override_info"][name]
            parts.append(
                "<tr{}><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    cls,
                    name,
                    row["physical_scores"][name],
                    row["vlm_scores"][name],
                    row["diff"][name],
                    row["confidence_scores"][name],
                    row["verified_scores"][name],
                    row["fused_scores"][name],
                    row["fused_diff"][name],
                    row["fusion_policy"][name]["policy"],
                    override["override_triggered"],
                    override["override_reason"],
                )
            )
        parts.append("</table>")
        parts.append("<p>valid={} markdown_wrapped={} source={}</p>".format(row["valid"], row["markdown_wrapped"], row["source"]))
        parts.append("<pre>{}</pre>".format(_escape(row["raw_output"])))
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def _write_markdown(path, rows, summary):
    lines = ["# VLM Label Quality Inspection", "", "```json", json.dumps(summary, indent=2, sort_keys=True), "```", ""]
    for row in rows:
        lines.extend(["## {} raw_high={} fused_high={}".format(row["sample_id"], row["raw_vlm_high_diff"], row["fused_high_diff"]), ""])
        lines.append("![{}]({})".format(row["sample_id"], row["image_rel"]))
        lines.extend(["", "|label|physical|vlm|raw diff|confidence|verified|fused|fused diff|policy|override|reason|", "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"])
        for name in LABEL_NAMES:
            marker = "**" if row["diff"][name] > 0.4 or row["fused_diff"][name] > 0.4 else ""
            override = row["override_info"][name]
            lines.append("|{}{}{}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{}|{}|{}|".format(
                marker, name, marker,
                row["physical_scores"][name],
                row["vlm_scores"][name],
                row["diff"][name],
                row["confidence_scores"][name],
                row["verified_scores"][name],
                row["fused_scores"][name],
                row["fused_diff"][name],
                row["fusion_policy"][name]["policy"],
                override["override_triggered"],
                override["override_reason"],
            ))
        lines.extend(["", "valid={} markdown_wrapped={} source={}".format(row["valid"], row["markdown_wrapped"], row["source"]), "", "```text", row["raw_output"], "```", ""])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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


def _escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


if __name__ == "__main__":
    main()
