#!/usr/bin/env python
import argparse
import json
import os
import statistics
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import load_reward_shaping_config
from lag_extensions.reward_shaping.qwen_vl_scorer import QwenVLMScorer
from lag_extensions.reward_shaping.vlm_cache import VLMScoreCache
from lag_extensions.reward_shaping.vlm_prompt import build_qwen_vl_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="scripts/results/aircombat_semantic_dataset")
    parser.add_argument("--config", type=str, default="lag_extensions/reward_shaping/configs/qwen2_5_vl_1v1.yaml")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--mock-vlm", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--pixel-preset", type=str, default=None, choices=["tiny", "fast", "balanced", "accurate"])
    parser.add_argument("--prompt-version", type=str, default=None, choices=["ultra_compact_v1", "compact_v1", "metric_aware_v1", "verbose_v1", "v1"])
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--load-mode", type=str, default=None, choices=["auto", "cuda", "explicit_single_gpu", "balanced"])
    parser.add_argument("--dtype", type=str, default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    config = load_reward_shaping_config(args.config)
    vlm_config = config.get("vlm", {})
    cache_config = config.get("cache", {})
    cache_path = args.cache or cache_config.get("path")
    prompt_version = args.prompt_version or vlm_config.get("prompt_version", "compact_v1")
    prompt = build_qwen_vl_prompt(prompt_version)
    min_pixels, max_pixels = _resolve_pixels(vlm_config, args.pixel_preset, args.min_pixels, args.max_pixels)
    max_new_tokens = int(args.max_new_tokens or vlm_config.get("max_new_tokens", 128))
    attn_implementation = _resolve_attention(vlm_config)

    print(json.dumps({
        "event": "qwen_vl_label_start",
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "cache_path": os.path.abspath(cache_path),
        "prompt_version": prompt_version,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_new_tokens": max_new_tokens,
        "device": vlm_config.get("device", "cuda"),
        "load_mode": args.load_mode or vlm_config.get("load_mode", "auto"),
        "device_map": args.device_map or vlm_config.get("device_map"),
        "dtype": args.dtype or vlm_config.get("dtype", "float16"),
        "dry_run": bool(args.dry_run or args.mock_vlm),
        "resume": bool(args.resume),
    }, sort_keys=True))

    scorer = QwenVLMScorer(
        model_name_or_path=vlm_config.get("model_name_or_path", "Qwen/Qwen2.5-VL-7B-Instruct"),
        device=vlm_config.get("device", "cuda"),
        dtype=args.dtype or vlm_config.get("dtype", "auto"),
        temperature=float(vlm_config.get("temperature", 0.0)),
        top_p=float(vlm_config.get("top_p", 1.0)),
        max_new_tokens=max_new_tokens,
        load_mode=args.load_mode or vlm_config.get("load_mode", "auto"),
        device_map=args.device_map or vlm_config.get("device_map"),
        local_files_only=bool(vlm_config.get("local_files_only", True)),
        attn_implementation=attn_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        mock=bool(args.dry_run or args.mock_vlm),
        load_model=not bool(args.dry_run or args.mock_vlm),
    )
    cache = VLMScoreCache(cache_path, enabled=bool(cache_config.get("enabled", True)))

    count = 0
    valid_count = 0
    fallback_count = 0
    cache_hit_count = 0
    markdown_count = 0
    errors = 0
    times = []
    output_tokens = []
    preprocess_times = []
    generate_times = []
    decode_parse_times = []
    truncated_count = 0
    invalid_json_count = 0
    wall_start = time.time()
    sample_dirs = _sample_dirs(args.dataset_dir)
    for sample_dir in _sample_dirs(args.dataset_dir):
        if args.limit is not None and count >= args.limit:
            break
        metadata = _read_json(os.path.join(sample_dir, "metadata.json"))
        metadata["prompt_version"] = prompt_version
        physical_labels = _read_json(os.path.join(sample_dir, "physical_labels.json"))
        physical_scores = {name: float(value["score"]) for name, value in physical_labels.items()}
        result = cache.score_with_cache(
            scorer=scorer,
            image_path=os.path.join(sample_dir, "image.png"),
            prompt=prompt,
            metadata=metadata,
            physical_scores=physical_scores,
            model_config={
                "temperature": vlm_config.get("temperature", 0.0),
                "top_p": vlm_config.get("top_p", 1.0),
                "max_new_tokens": max_new_tokens,
                "prompt_version": prompt_version,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "device_map": args.device_map or vlm_config.get("device_map"),
                "dtype": vlm_config.get("dtype", "float16"),
                "attn_implementation": attn_implementation,
                "load_mode": args.load_mode or vlm_config.get("load_mode", "auto"),
            },
            fallback_to_physical=bool(vlm_config.get("fallback_to_physical", True)),
        )
        count += 1
        if result.get("valid"):
            valid_count += 1
        if result.get("cache_hit"):
            cache_hit_count += 1
        if result.get("source") == "cached_vlm_fallback_physical":
            fallback_count += 1
        if result.get("error"):
            errors += 1
        raw_output = result.get("raw_output") or ""
        if "```" in raw_output:
            markdown_count += 1
        inference_time = result.get("inference_time_sec")
        if inference_time is not None and not result.get("cache_hit"):
            times.append(float(inference_time))
        gen_info = result.get("generation_info") or {}
        if gen_info.get("output_token_count") is not None and not result.get("cache_hit"):
            output_tokens.append(float(gen_info["output_token_count"]))
        timings = gen_info.get("timings") or {}
        if not result.get("cache_hit"):
            if timings.get("preprocess_time") is not None:
                preprocess_times.append(float(timings["preprocess_time"]))
            if timings.get("generate_time") is not None:
                generate_times.append(float(timings["generate_time"]))
            decode_parse = float(timings.get("decode_time", 0.0)) + float(timings.get("json_parse_time", 0.0))
            decode_parse_times.append(decode_parse)
        if gen_info.get("output_truncated"):
            truncated_count += 1
        if gen_info.get("invalid_json"):
            invalid_json_count += 1

        if count == 1 or count % max(1, args.progress_every) == 0:
            print(json.dumps({
                "event": "progress",
                "count": count,
                "total_available": len(sample_dirs),
                "valid_rate": valid_count / count if count else 0.0,
                "cache_hit_rate": cache_hit_count / count if count else 0.0,
                "fallback_count": fallback_count,
                "last_image_id": metadata.get("image_id"),
                "last_valid": result.get("valid"),
                "last_cache_hit": result.get("cache_hit"),
                "last_source": result.get("source"),
                "last_inference_time_sec": inference_time,
                "last_generate_time_sec": timings.get("generate_time"),
                "last_output_truncated": gen_info.get("output_truncated"),
            }, sort_keys=True))

    summary = {
        "event": "qwen_vl_label_done",
        "labeled": count,
        "wall_time_sec": time.time() - wall_start,
        "avg_wall_time_per_sample_sec": (time.time() - wall_start) / count if count else 0.0,
        "valid_rate": valid_count / count if count else 0.0,
        "fallback_count": fallback_count,
        "cache_hit_count": cache_hit_count,
        "cache_hit_rate": cache_hit_count / count if count else 0.0,
        "markdown_wrapped_count": markdown_count,
        "error_count": errors,
        "avg_inference_time_sec": _mean(times),
        "p50_inference_time_sec": _percentile(times, 50),
        "p90_inference_time_sec": _percentile(times, 90),
        "p95_inference_time_sec": _percentile(times, 95),
        "avg_output_tokens": _mean(output_tokens),
        "avg_preprocess_time_sec": _mean(preprocess_times),
        "avg_generate_time_sec": _mean(generate_times),
        "avg_decode_parse_time_sec": _mean(decode_parse_times),
        "output_truncated_count": truncated_count,
        "invalid_json_count": invalid_json_count,
        "runtime_diagnostics": getattr(scorer, "last_runtime_diagnostics", {}),
        "model_load_count": getattr(scorer, "load_count", None),
        "cache": cache.stats(),
        "cache_path": os.path.abspath(cache_path),
    }
    print(json.dumps(summary, sort_keys=True))


def _sample_dirs(dataset_dir):
    return [
        os.path.join(dataset_dir, name)
        for name in sorted(os.listdir(dataset_dir))
        if name.startswith("sample_") and os.path.isdir(os.path.join(dataset_dir, name))
    ]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_pixels(vlm_config, preset_name, min_pixels, max_pixels):
    if preset_name:
        preset = (vlm_config.get("pixel_presets") or {}).get(preset_name, {})
        min_pixels = min_pixels if min_pixels is not None else preset.get("min_pixels")
        max_pixels = max_pixels if max_pixels is not None else preset.get("max_pixels")
    min_pixels = min_pixels if min_pixels is not None else vlm_config.get("min_pixels")
    max_pixels = max_pixels if max_pixels is not None else vlm_config.get("max_pixels")
    return _maybe_int(min_pixels), _maybe_int(max_pixels)


def _resolve_attention(vlm_config):
    if bool(vlm_config.get("use_flash_attention_2", False)):
        return "flash_attention_2"
    return vlm_config.get("attn_implementation")


def _maybe_int(value):
    if value is None:
        return None
    return int(value)


def _mean(values):
    return statistics.mean(values) if values else None


def _percentile(values, percentile):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100.0
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


if __name__ == "__main__":
    main()
