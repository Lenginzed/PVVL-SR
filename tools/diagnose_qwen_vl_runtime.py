#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.qwen_vl_scorer import QwenVLMScorer
from lag_extensions.reward_shaping.vlm_prompt import build_qwen_vl_prompt


def main():
    parser = argparse.ArgumentParser(description="Diagnose Qwen2.5-VL runtime device placement and memory.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--load-mode", default="auto", choices=["auto", "cuda", "explicit_single_gpu", "balanced"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--pixel-preset", default="balanced", choices=["tiny", "fast", "balanced", "accurate"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-version", default="compact_v1", choices=["ultra_compact_v1", "compact_v1", "verbose_v1", "v1"])
    parser.add_argument("--smoke-image", default=None)
    parser.add_argument("--no-smoke", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    args = parser.parse_args()

    min_pixels, max_pixels = _pixel_preset(args.pixel_preset)
    report = {
        "model": "Qwen2.5-VL-7B-Instruct",
        "model_path": os.path.abspath(args.model_path),
        "load_mode": args.load_mode,
        "device": args.device,
        "dtype": args.dtype,
        "device_map_arg": args.device_map,
        "pixel_preset": args.pixel_preset,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "prompt_version": args.prompt_version,
        "runtime": _runtime_info(),
    }

    load_start = time.time()
    scorer = None
    try:
        scorer = QwenVLMScorer(
            model_name_or_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            load_mode=args.load_mode,
            device_map=args.device_map,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            mock=False,
            load_model=True,
        )
        report["load"] = {
            "ok": True,
            "load_time_sec": time.time() - load_start,
            "diagnostics": scorer.runtime_diagnostics(),
        }
    except Exception as exc:
        report["load"] = {
            "ok": False,
            "load_time_sec": time.time() - load_start,
            "failure_category": _classify_exception(exc),
            "error": "{}: {}".format(type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }
        _print(report, args.json)
        return 1

    image_path = args.smoke_image or _find_default_smoke_image()
    if not args.no_smoke and image_path:
        prompt = build_qwen_vl_prompt(args.prompt_version)
        result = scorer.score_image(image_path=image_path, prompt=prompt, image_id="runtime_diagnose")
        report["smoke"] = {
            "image_path": os.path.abspath(image_path),
            "valid": result.get("valid"),
            "error": result.get("error"),
            "warnings": result.get("warnings", []),
            "raw_output": result.get("raw_output"),
            "scores": result.get("scores", {}),
            "generation_info": result.get("generation_info", {}),
            "diagnostics_after_inference": scorer.runtime_diagnostics(),
        }
    else:
        report["smoke"] = {"skipped": True, "reason": "no smoke image found or --no-smoke set"}

    _print(report, args.json)
    load_diag = report.get("load", {}).get("diagnostics", {})
    if load_diag.get("cpu_offload_detected"):
        return 2
    return 0


def _print(report, as_json):
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return
    runtime = report["runtime"]
    print("\n=== Runtime ===")
    _kv("torch_version", runtime.get("torch_version"))
    _kv("transformers_version", runtime.get("transformers_version"))
    _kv("cuda_available", runtime.get("cuda_available"))
    _kv("gpu_name", runtime.get("gpu_name"))
    _kv("gpu_total_memory_gib", runtime.get("gpu_total_memory_gib"))
    print("\n=== Load ===")
    _kv("load_mode", report.get("load_mode"))
    _kv("torch_dtype", report.get("dtype"))
    _kv("device", report.get("device"))
    _kv("device_map_arg", report.get("device_map_arg"))
    load = report.get("load", {})
    _kv("load_ok", load.get("ok"))
    _kv("load_time_sec", load.get("load_time_sec"))
    if not load.get("ok"):
        _kv("failure_category", load.get("failure_category"))
        _kv("error", load.get("error"))
        return
    diag = load.get("diagnostics", {})
    _kv("cpu_param_count", diag.get("cpu_param_count"))
    _kv("cuda_param_count", diag.get("cuda_param_count"))
    _kv("meta_param_count", diag.get("meta_param_count"))
    _kv("cpu_offload_detected", diag.get("cpu_offload_detected"))
    if diag.get("cpu_offload_detected"):
        print("WARNING: CPU offload detected. Inference may be very slow.")
    memory = diag.get("cuda_memory", {})
    _kv("gpu_allocated_gib_after_load", memory.get("allocated_gib"))
    _kv("gpu_reserved_gib_after_load", memory.get("reserved_gib"))
    print("\n=== Major Modules ===")
    for name, info in (diag.get("module_devices") or {}).items():
        print("{}: dominant={} cpu={} cuda={} meta={}".format(
            name, info.get("dominant_device"), info.get("cpu"), info.get("cuda"), info.get("meta")
        ))
    print("\n=== Smoke ===")
    smoke = report.get("smoke", {})
    if smoke.get("skipped"):
        _kv("skipped", smoke.get("reason"))
    else:
        _kv("image_path", smoke.get("image_path"))
        _kv("valid", smoke.get("valid"))
        timings = (smoke.get("generation_info") or {}).get("timings") or {}
        _kv("total_time", timings.get("total_time"))
        _kv("preprocess_time", timings.get("preprocess_time"))
        _kv("generate_time", timings.get("generate_time"))
        _kv("decode_time", timings.get("decode_time"))
        _kv("json_parse_time", timings.get("json_parse_time"))
        gpu = (smoke.get("generation_info") or {}).get("gpu_memory") or {}
        _kv("gpu_peak_allocated_gib", gpu.get("max_allocated_gib"))
        _kv("gpu_peak_reserved_gib", gpu.get("max_reserved_gib"))


def _runtime_info():
    info = {}
    try:
        import torch

        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            info["gpu_name"] = props.name
            info["gpu_total_memory_gib"] = props.total_memory / (1024 ** 3)
    except Exception as exc:
        info["torch_error"] = "{}: {}".format(type(exc).__name__, exc)
    try:
        import transformers

        info["transformers_version"] = getattr(transformers, "__version__", None)
    except Exception as exc:
        info["transformers_error"] = "{}: {}".format(type(exc).__name__, exc)
    return info


def _find_default_smoke_image():
    candidates = [
        "scripts/results/aircombat_semantic_dataset_qwen_stage5_panel/sample_000000/image.png",
        "scripts/results/aircombat_semantic_dataset_qwen_test/sample_000000/image.png",
        "scripts/results/aircombat_semantic_dataset_smoke_check/sample_000000/image.png",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _pixel_preset(name):
    presets = {
        "tiny": (50176, 100352),
        "fast": (100352, 200704),
        "balanced": (200704, 401408),
        "accurate": (200704, 802816),
    }
    return presets[name]


def _classify_exception(exc):
    text = "{}: {}".format(type(exc).__name__, exc).lower()
    if "out of memory" in text or "outofmemory" in text:
        return "CUDA / VRAM insufficient"
    if "cuda" in text and "not available" in text:
        return "CUDA unavailable"
    if "flash_attn" in text or "flash attention" in text:
        return "FlashAttention dependency missing or unsupported"
    if "transformers" in text or "qwen2_5_vl" in text:
        return "Transformers/model support failure"
    return "Unknown failure"


def _kv(key, value):
    print("{:<36} {}".format(key + ":", value))


if __name__ == "__main__":
    raise SystemExit(main())
