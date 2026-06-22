#!/usr/bin/env python
import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import load_reward_shaping_config
from lag_extensions.reward_shaping.qwen_vl_scorer import QwenVLMScorer
from lag_extensions.reward_shaping.situation_renderer import SituationRenderer
from lag_extensions.reward_shaping.vlm_cache import VLMScoreCache
from lag_extensions.reward_shaping.vlm_prompt import build_qwen_vl_prompt


PIXEL_PRESETS = {
    "tiny": {"min_pixels": 50176, "max_pixels": 100352},
    "fast": {"min_pixels": 100352, "max_pixels": 200704},
    "balanced": {"min_pixels": 200704, "max_pixels": 401408},
    "accurate": {"min_pixels": 200704, "max_pixels": 802816},
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark offline Qwen2.5-VL semantic labelling settings.")
    parser.add_argument("--dataset-dir", default="scripts/results/aircombat_semantic_dataset")
    parser.add_argument("--config", default="lag_extensions/reward_shaping/configs/qwen2_5_vl_1v1.yaml")
    parser.add_argument("--output-dir", default="scripts/results/vlm_benchmark")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--sweep", choices=["quick", "full", "custom"], default="quick")
    parser.add_argument("--image-sizes", default="384,512")
    parser.add_argument("--pixel-presets", default="tiny,fast,balanced")
    parser.add_argument("--prompt-versions", default="compact_v1,ultra_compact_v1")
    parser.add_argument("--max-new-tokens-list", default="64,96,128")
    parser.add_argument("--load-modes", default="auto,explicit_single_gpu,cuda")
    parser.add_argument("--dtypes", default="float16,auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--mock-vlm", action="store_true", default=False)
    parser.add_argument("--use-cache", action="store_true", default=False)
    args = parser.parse_args()

    config = load_reward_shaping_config(args.config)
    vlm_config = config.get("vlm", {})
    os.makedirs(args.output_dir, exist_ok=True)
    samples = _load_samples(args.dataset_dir, args.limit)
    if not samples:
        raise RuntimeError("No sample_* directories found in {}".format(args.dataset_dir))

    combos = _build_combos(args)
    cache = None
    if args.use_cache:
        cache = VLMScoreCache(os.path.join(args.output_dir, "benchmark_cache.jsonl"), enabled=True)

    rows = []
    start_all = time.time()
    for combo in combos:
        row = _run_combo(combo, samples, cache, config, args, args.output_dir)
        rows.append(row)
        print(json.dumps({"event": "benchmark_combo_done", **row}, sort_keys=True))

    csv_path = os.path.abspath(os.path.join(args.output_dir, "qwen2_5_vl_runtime_benchmark.csv"))
    summary_path = os.path.abspath(os.path.join(args.output_dir, "qwen2_5_vl_runtime_summary.json"))
    _write_csv(csv_path, rows)
    summary = {
        "model": "Qwen2.5-VL-7B-Instruct",
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "num_samples": len(samples),
        "num_configs": len(rows),
        "total_wall_time_sec": time.time() - start_all,
        "best_by_avg_time": min(rows, key=lambda item: item["avg_inference_time_sec"] or float("inf")),
        "csv_path": csv_path,
        "summary_path": summary_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps({"event": "benchmark_done", **summary}, sort_keys=True))


def _run_combo(combo, samples, cache, config, args, output_dir):
    preset = PIXEL_PRESETS[combo["pixel_preset"]]
    prompt = build_qwen_vl_prompt(combo["prompt_version"])
    renderer_config = dict(config.get("renderer", {}))
    renderer_config.update({
        "image_size": int(combo["image_size"]),
        "prompt_version": combo["prompt_version"],
    })
    renderer = SituationRenderer(renderer_config, config.get("thresholds", {}))

    times = []
    output_tokens = []
    valid_count = 0
    fallback_count = 0
    cache_hit_count = 0
    markdown_count = 0
    image_sizes = []
    preprocess_times = []
    generate_times = []
    decode_parse_times = []
    truncated_count = 0
    invalid_json_count = 0
    load_time_sec = None
    load_error = None
    load_failure_category = None
    runtime_diag = {}
    rendered_dir = os.path.join(output_dir, "rendered", combo["config_name"])
    os.makedirs(rendered_dir, exist_ok=True)
    scorer = None

    try:
        load_start = time.time()
        vlm_config = config.get("vlm", {})
        scorer = QwenVLMScorer(
            model_name_or_path=vlm_config.get("model_name_or_path", "models/Qwen2.5-VL-7B-Instruct"),
            device=vlm_config.get("device", "cuda:0"),
            dtype=combo["dtype"],
            temperature=float(vlm_config.get("temperature", 0.0)),
            top_p=float(vlm_config.get("top_p", 1.0)),
            max_new_tokens=int(combo["max_new_tokens"]),
            load_mode=combo["load_mode"],
            device_map=args.device_map or vlm_config.get("device_map", "auto"),
            local_files_only=bool(vlm_config.get("local_files_only", True)),
            attn_implementation=_resolve_attention(vlm_config),
            min_pixels=preset["min_pixels"],
            max_pixels=preset["max_pixels"],
            mock=bool(args.dry_run or args.mock_vlm),
            load_model=not bool(args.dry_run or args.mock_vlm),
        )
        load_time_sec = time.time() - load_start
        runtime_diag = getattr(scorer, "last_runtime_diagnostics", {}) or scorer.runtime_diagnostics()
    except Exception as exc:
        load_time_sec = time.time() - load_start
        load_error = "{}: {}".format(type(exc).__name__, exc)
        load_failure_category = _classify_exception(exc)
        runtime_diag = {"cpu_offload_detected": None, "load_error_traceback": traceback.format_exc()}
        return _row(
            combo,
            len(samples),
            load_time_sec=load_time_sec,
            load_error=load_error,
            load_failure_category=load_failure_category,
            runtime_diag=runtime_diag,
        )

    for sample in samples:
        sample_render_dir = os.path.join(rendered_dir, sample["sample_id"])
        metadata = renderer.render_sample(
            sample["state"],
            sample["metrics"],
            sample["labels"],
            sample_render_dir,
            sample_id=sample["sample_id"],
            image_filename="image.png",
            metadata_filename="metadata.json",
        )
        image_sizes.append(os.path.getsize(metadata["image_path"]))
        physical_scores = {name: float(value["score"]) for name, value in sample["labels"].items()}
        model_config = {
            "prompt_version": combo["prompt_version"],
            "max_new_tokens": int(combo["max_new_tokens"]),
            "min_pixels": preset["min_pixels"],
            "max_pixels": preset["max_pixels"],
            "image_size": int(combo["image_size"]),
        }
        if cache is not None:
            result = cache.score_with_cache(
                scorer=scorer,
                image_path=metadata["image_path"],
                prompt=prompt,
                metadata=metadata,
                physical_scores=physical_scores,
                model_config=model_config,
                fallback_to_physical=True,
            )
        else:
            result = scorer.score_image(
                image_path=metadata["image_path"],
                prompt=prompt,
                image_id=metadata.get("image_id"),
                physical_scores=physical_scores,
            )
        if result.get("valid"):
            valid_count += 1
        if result.get("cache_hit"):
            cache_hit_count += 1
        if result.get("source") == "cached_vlm_fallback_physical":
            fallback_count += 1
        if "```" in (result.get("raw_output") or ""):
            markdown_count += 1
        if not result.get("cache_hit") and result.get("inference_time_sec") is not None:
            times.append(float(result["inference_time_sec"]))
        gen_info = result.get("generation_info") or {}
        if not result.get("cache_hit") and gen_info.get("output_token_count") is not None:
            output_tokens.append(float(gen_info["output_token_count"]))
        timings = gen_info.get("timings") or {}
        if not result.get("cache_hit"):
            if timings.get("preprocess_time") is not None:
                preprocess_times.append(float(timings["preprocess_time"]))
            if timings.get("generate_time") is not None:
                generate_times.append(float(timings["generate_time"]))
            decode_parse_times.append(float(timings.get("decode_time", 0.0)) + float(timings.get("json_parse_time", 0.0)))
        if gen_info.get("output_truncated"):
            truncated_count += 1
        if gen_info.get("invalid_json"):
            invalid_json_count += 1

    final_diag = scorer.runtime_diagnostics() if scorer is not None and scorer.model is not None else runtime_diag
    _release_scorer(scorer)

    count = len(samples)
    return _row(
        combo,
        count,
        load_time_sec=load_time_sec,
        runtime_diag=final_diag,
        valid_rate=valid_count / count if count else 0.0,
        fallback_count=fallback_count,
        markdown_wrapped_count=markdown_count,
        avg_inference_time_sec=_mean(times),
        p50_time_sec=_percentile(times, 50),
        p90_time_sec=_percentile(times, 90),
        p95_time_sec=_percentile(times, 95),
        avg_output_tokens=_mean(output_tokens),
        avg_input_image_size_bytes=_mean(image_sizes),
        cache_hit_rate=cache_hit_count / count if count else 0.0,
        avg_preprocess_time_sec=_mean(preprocess_times),
        avg_generate_time_sec=_mean(generate_times),
        avg_decode_parse_time_sec=_mean(decode_parse_times),
        output_truncated_count=truncated_count,
        invalid_json_count=invalid_json_count,
    )


def _build_combos(args):
    if args.sweep == "quick":
        seeds = [
            (512, "balanced", "compact_v1", 128, "auto", "float16"),
            (512, "balanced", "ultra_compact_v1", 96, "auto", "float16"),
            (384, "tiny", "ultra_compact_v1", 96, "auto", "float16"),
            (512, "balanced", "ultra_compact_v1", 96, "explicit_single_gpu", "float16"),
        ]
    else:
        image_sizes = _parse_int_list(args.image_sizes)
        pixel_presets = _parse_str_list(args.pixel_presets)
        prompt_versions = _parse_str_list(args.prompt_versions)
        token_values = _parse_int_list(args.max_new_tokens_list)
        load_modes = _parse_str_list(args.load_modes)
        dtypes = _parse_str_list(args.dtypes)
        seeds = [
            (image_size, preset, prompt_version, tokens, load_mode, dtype)
            for image_size in image_sizes
            for preset in pixel_presets
            for prompt_version in prompt_versions
            for tokens in token_values
            for load_mode in load_modes
            for dtype in dtypes
        ]
    combos = []
    for image_size, preset, prompt_version, tokens, load_mode, dtype in seeds:
        if preset not in PIXEL_PRESETS:
            raise ValueError("Unknown pixel preset: {}".format(preset))
        combos.append({
            "config_name": "img{}_{}_{}_tok{}_{}_{}".format(image_size, preset, prompt_version, tokens, load_mode, dtype),
            "image_size": int(image_size),
            "pixel_preset": preset,
            "prompt_version": prompt_version,
            "max_new_tokens": int(tokens),
            "load_mode": load_mode,
            "dtype": dtype,
        })
    return combos


def _row(combo, count, **kwargs):
    preset = PIXEL_PRESETS[combo["pixel_preset"]]
    runtime_diag = kwargs.pop("runtime_diag", {}) or {}
    memory = runtime_diag.get("cuda_memory", {}) if isinstance(runtime_diag, dict) else {}
    avg_inference_time = kwargs.pop("avg_inference_time_sec", None)
    row = {
        "config_name": combo["config_name"],
        "load_mode": combo["load_mode"],
        "dtype": combo["dtype"],
        "image_size": int(combo["image_size"]),
        "min_pixels": preset["min_pixels"],
        "max_pixels": preset["max_pixels"],
        "pixel_preset": combo["pixel_preset"],
        "prompt_version": combo["prompt_version"],
        "max_new_tokens": int(combo["max_new_tokens"]),
        "num_samples": count,
        "load_time_sec": kwargs.pop("load_time_sec", None),
        "load_error": kwargs.pop("load_error", None),
        "load_failure_category": kwargs.pop("load_failure_category", None),
        "cpu_param_count": runtime_diag.get("cpu_param_count"),
        "cuda_param_count": runtime_diag.get("cuda_param_count"),
        "meta_param_count": runtime_diag.get("meta_param_count"),
        "cpu_offload_detected": runtime_diag.get("cpu_offload_detected"),
        "gpu_peak_memory_gib": memory.get("max_reserved_gib"),
        "valid_rate": kwargs.pop("valid_rate", 0.0),
        "fallback_count": kwargs.pop("fallback_count", None),
        "markdown_wrapped_count": kwargs.pop("markdown_wrapped_count", None),
        "avg_inference_time_sec": avg_inference_time,
        "p50_time_sec": kwargs.pop("p50_time_sec", None),
        "p90_time_sec": kwargs.pop("p90_time_sec", None),
        "p95_time_sec": kwargs.pop("p95_time_sec", None),
        "avg_preprocess_time_sec": kwargs.pop("avg_preprocess_time_sec", None),
        "avg_generate_time_sec": kwargs.pop("avg_generate_time_sec", None),
        "avg_decode_parse_time_sec": kwargs.pop("avg_decode_parse_time_sec", None),
        "total_time_sec": avg_inference_time,
        "avg_output_tokens": kwargs.pop("avg_output_tokens", None),
        "avg_input_image_size_bytes": kwargs.pop("avg_input_image_size_bytes", None),
        "cache_hit_rate": kwargs.pop("cache_hit_rate", None),
        "output_truncated_count": kwargs.pop("output_truncated_count", None),
        "invalid_json_count": kwargs.pop("invalid_json_count", None),
    }
    return row


def _load_samples(dataset_dir, limit):
    samples = []
    for sample_dir in _sample_dirs(dataset_dir):
        if limit is not None and len(samples) >= limit:
            break
        samples.append({
            "sample_id": os.path.basename(sample_dir),
            "state": _read_json(os.path.join(sample_dir, "env_state_snapshot.json")),
            "metrics": _read_json(os.path.join(sample_dir, "metrics.json")),
            "labels": _read_json(os.path.join(sample_dir, "physical_labels.json")),
        })
    return samples


def _sample_dirs(dataset_dir):
    return [
        os.path.join(dataset_dir, name)
        for name in sorted(os.listdir(dataset_dir))
        if name.startswith("sample_") and os.path.isdir(os.path.join(dataset_dir, name))
    ]


def _resolve_attention(vlm_config):
    if bool(vlm_config.get("use_flash_attention_2", False)):
        return "flash_attention_2"
    return vlm_config.get("attn_implementation")


def _release_scorer(scorer):
    if scorer is None:
        return
    try:
        scorer.model = None
        scorer.processor = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _classify_exception(exc):
    text = "{}: {}".format(type(exc).__name__, exc).lower()
    if "out of memory" in text or "outofmemory" in text:
        return "CUDA / VRAM insufficient"
    if "cuda" in text and "not available" in text:
        return "CUDA unavailable"
    if "flash_attn" in text or "flash attention" in text:
        return "FlashAttention dependency missing or unsupported"
    if "qwen2_5_vl" in text or "transformers" in text:
        return "Transformers/model support failure"
    return "Unknown failure"


def _parse_int_list(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_str_list(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
