#!/usr/bin/env python
"""Check a local Qwen2.5-VL-7B-Instruct snapshot without downloading files.

Default behavior is intentionally lightweight:
  1. inspect local files and safetensors index
  2. run AutoConfig/AutoProcessor with local_files_only=True

Use --load-model to load the 7B weights, and --smoke-image to run one
minimal image-to-JSON inference. Both optional steps can require recent
transformers, accelerate, qwen-vl-utils, CUDA, and enough memory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LABEL_NAMES = [
    "ego_tail_advantage",
    "enemy_tail_threat",
    "effective_attack_window",
    "enemy_missile_threat_zone",
    "energy_advantage",
    "energy_disadvantage",
    "defensive_escape",
    "neutral_stalemate",
]


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    return "{} B".format(num_bytes)


def load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # pragma: no cover - exact exception is platform-dependent
        return None, "{}: {}".format(type(exc).__name__, exc)


def classify_exception(exc: BaseException) -> str:
    text = "{}: {}".format(type(exc).__name__, exc)
    lower = text.lower()
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "Python dependency missing"
    if "numpy.dtype size changed" in lower or "binary incompatibility" in lower:
        return "Python binary dependency incompatibility"
    if "keyerror" in lower and "qwen2_5_vl" in lower:
        return "Transformers version does not support qwen2_5_vl"
    if "qwen2_5_vl" in lower and ("not recognized" in lower or "model type" in lower):
        return "Transformers version does not support qwen2_5_vl"
    if "cuda out of memory" in lower or "outofmemoryerror" in lower:
        return "CUDA / VRAM insufficient"
    if "cuda" in lower and ("not available" in lower or "no cuda" in lower):
        return "CUDA unavailable"
    if "safetensor" in lower or "checkpoint" in lower or "state_dict" in lower:
        return "Model weight loading failure"
    if "json" in lower:
        return "Inference output JSON parsing failure"
    return "Unknown failure"


def format_error(exc: BaseException, verbose: bool = False) -> Dict[str, Any]:
    data = {
        "ok": False,
        "category": classify_exception(exc),
        "error": "{}: {}".format(type(exc).__name__, exc),
    }
    if verbose:
        data["traceback"] = traceback.format_exc()
    return data


def check_local_files(model_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "model_path": str(model_path),
        "exists": model_path.exists(),
        "is_dir": model_path.is_dir(),
        "file_count": 0,
        "safetensors_count": 0,
        "safetensors_total_size_bytes": 0,
        "safetensors_total_size_human": "0 B",
        "required_files": {},
        "tokenizer_files": {},
        "processor_files": {},
        "index": {},
        "config": {},
        "warnings": [],
        "errors": [],
        "pass": False,
    }
    if not model_path.exists():
        result["errors"].append("model directory does not exist")
        return result
    if not model_path.is_dir():
        result["errors"].append("model path is not a directory")
        return result

    files = [path for path in model_path.iterdir() if path.is_file()]
    result["file_count"] = len(files)
    safetensors = sorted(model_path.glob("*.safetensors"))
    result["safetensors_count"] = len(safetensors)
    total_size = sum(path.stat().st_size for path in safetensors)
    result["safetensors_total_size_bytes"] = int(total_size)
    result["safetensors_total_size_human"] = human_bytes(total_size)

    required = [
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    optional = ["generation_config.json"]
    for name in required + optional:
        path = model_path / name
        result["required_files"][name] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
        if name in required and not path.exists():
            result["errors"].append("missing required file: {}".format(name))
        if name in optional and not path.exists():
            result["warnings"].append("optional file missing: {}".format(name))

    tokenizer_candidates = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model"]
    for name in tokenizer_candidates:
        path = model_path / name
        result["tokenizer_files"][name] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    if not any((model_path / name).exists() for name in tokenizer_candidates):
        result["errors"].append("missing tokenizer payload file: tokenizer.json/vocab/merges/tokenizer.model")

    processor_candidates = [
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor_config.json",
    ]
    for name in processor_candidates:
        path = model_path / name
        result["processor_files"][name] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    if not any((model_path / name).exists() for name in processor_candidates):
        result["errors"].append("missing processor/image preprocessor config")

    config_path = model_path / "config.json"
    if config_path.exists():
        config, error = load_json(config_path)
        if error:
            result["errors"].append("cannot read config.json: {}".format(error))
        elif config is not None:
            result["config"] = {
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "torch_dtype": config.get("torch_dtype"),
                "transformers_version": config.get("transformers_version"),
            }

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index, error = load_json(index_path)
        if error:
            result["errors"].append("cannot read model.safetensors.index.json: {}".format(error))
        elif index is not None:
            weight_map = index.get("weight_map", {})
            referenced_shards = sorted(set(weight_map.values()))
            missing = []
            zero_size = []
            suspicious_small = []
            shard_sizes = {}
            for shard in referenced_shards:
                shard_path = model_path / shard
                if not shard_path.exists():
                    missing.append(shard)
                    continue
                size = shard_path.stat().st_size
                shard_sizes[shard] = size
                if size == 0:
                    zero_size.append(shard)
                if size < 1024 * 1024:
                    suspicious_small.append(shard)
            extra_safetensors = sorted(path.name for path in safetensors if path.name not in referenced_shards)
            result["index"] = {
                "metadata": index.get("metadata", {}),
                "weight_map_entries": len(weight_map),
                "referenced_shard_count": len(referenced_shards),
                "referenced_shards": referenced_shards,
                "missing_referenced_shards": missing,
                "zero_size_referenced_shards": zero_size,
                "suspicious_small_referenced_shards": suspicious_small,
                "extra_safetensors_not_in_index": extra_safetensors,
                "referenced_shard_sizes": shard_sizes,
            }
            if missing:
                result["errors"].append("index references missing safetensors: {}".format(", ".join(missing)))
            if zero_size:
                result["errors"].append("index references zero-byte safetensors: {}".format(", ".join(zero_size)))
            if suspicious_small:
                result["warnings"].append(
                    "index references unusually small safetensors: {}".format(", ".join(suspicious_small))
                )
    else:
        result["errors"].append("missing model.safetensors.index.json")

    result["pass"] = len(result["errors"]) == 0
    return result


def check_transformers_metadata(model_path: Path, verbose: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "auto_config": {"ok": False},
        "auto_processor": {"ok": False},
        "versions": {},
    }
    try:
        import transformers
        from transformers import AutoConfig, AutoProcessor

        result["versions"]["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception as exc:
        err = format_error(exc, verbose=verbose)
        result["auto_config"] = err
        result["auto_processor"] = err
        result["upgrade_hint"] = transformers_upgrade_hint(err["category"])
        return result

    try:
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        result["auto_config"] = {
            "ok": True,
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
        }
    except Exception as exc:
        result["auto_config"] = format_error(exc, verbose=verbose)

    try:
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        result["auto_processor"] = {
            "ok": True,
            "processor_class": processor.__class__.__name__,
        }
    except Exception as exc:
        result["auto_processor"] = format_error(exc, verbose=verbose)

    categories = {
        item.get("category")
        for item in [result["auto_config"], result["auto_processor"]]
        if not item.get("ok")
    }
    hints = [transformers_upgrade_hint(category) for category in categories if category]
    result["upgrade_hint"] = "\n".join(sorted(set(hint for hint in hints if hint))) or None
    return result


def transformers_upgrade_hint(category: str) -> Optional[str]:
    if category == "Transformers version does not support qwen2_5_vl":
        return (
            "Upgrade local dependencies, for example: "
            "pip install -U transformers accelerate safetensors pillow qwen-vl-utils. "
            "If qwen2_5_vl is still unsupported, use: "
            "pip install -U git+https://github.com/huggingface/transformers accelerate"
        )
    if category == "Python dependency missing":
        return "Install missing inference dependencies: pip install -U transformers accelerate safetensors pillow qwen-vl-utils"
    if category == "Python binary dependency incompatibility":
        return (
            "A compiled Python package is incompatible with the installed NumPy ABI. "
            "Reinstall/align NumPy-dependent wheels in this environment, commonly: "
            "pip install --force-reinstall --no-cache-dir numpy pandas scikit-learn scipy h5py"
        )
    return None


def get_runtime_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch

        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            info["cuda_current_device"] = int(current)
            info["gpu_name"] = torch.cuda.get_device_name(current)
            info["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(current))
            info["gpu_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(current))
    except Exception as exc:
        info["torch_error"] = "{}: {}".format(type(exc).__name__, exc)

    try:
        import transformers

        info["transformers_version"] = getattr(transformers, "__version__", "unknown")
    except Exception as exc:
        info["transformers_error"] = "{}: {}".format(type(exc).__name__, exc)
    return info


def load_qwen_model(
    model_path: Path,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    attn_implementation: str = "auto",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Any, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "runtime": get_runtime_info(),
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    model = None
    processor = None
    start = time.time()
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "local_files_only": True,
        }
        if attn_implementation and attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path), **model_kwargs)
        processor_kwargs = {"local_files_only": True}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        processor = AutoProcessor.from_pretrained(str(model_path), **processor_kwargs)
        result["ok"] = True
        result["load_time_sec"] = time.time() - start
        result["model_class"] = model.__class__.__name__
        result["processor_class"] = processor.__class__.__name__
        result["model_device"] = str(getattr(model, "device", "unknown"))
        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            result["gpu_memory_allocated_bytes_after_load"] = int(torch.cuda.memory_allocated(current))
            result["gpu_memory_reserved_bytes_after_load"] = int(torch.cuda.memory_reserved(current))
            result["gpu_memory_allocated_after_load"] = human_bytes(int(torch.cuda.memory_allocated(current)))
            result["gpu_memory_reserved_after_load"] = human_bytes(int(torch.cuda.memory_reserved(current)))
    except Exception as exc:
        result.update(format_error(exc, verbose=verbose))
        result["load_time_sec"] = time.time() - start
        result["hint"] = transformers_upgrade_hint(result.get("category", "")) or None
    return result, model, processor


def extract_json(raw_output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        if not match:
            return None, "json_object_not_found"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return None, "json_parse_failed: {}".format(exc)
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    return parsed, None


def parse_label_scores(raw_output: str) -> Dict[str, Any]:
    parsed, error = extract_json(raw_output)
    if parsed is None:
        return {"valid": False, "scores": {}, "error": error}
    scores: Dict[str, float] = {}
    missing: List[str] = []
    warnings: List[str] = []
    for name in LABEL_NAMES:
        if name not in parsed:
            missing.append(name)
            continue
        try:
            value = float(parsed[name])
        except (TypeError, ValueError):
            missing.append(name)
            continue
        clipped = min(max(value, 0.0), 1.0)
        if not math.isclose(clipped, value):
            warnings.append("clipped {} from {}".format(name, value))
        scores[name] = clipped
    if missing:
        return {
            "valid": False,
            "scores": scores,
            "error": "missing_or_invalid_labels: {}".format(",".join(missing)),
            "warnings": warnings,
        }
    return {"valid": True, "scores": scores, "error": None, "warnings": warnings}


def build_smoke_prompt() -> str:
    try:
        from lag_extensions.reward_shaping.vlm_prompt import build_qwen_vl_prompt

        return build_qwen_vl_prompt()
    except Exception:
        labels = "\n".join("- {}".format(name) for name in LABEL_NAMES)
        return (
            "Output strict JSON only. Include exactly these 8 keys with float values from 0 to 1:\n"
            "{}\nDo not output Markdown, explanation, or action advice.".format(labels)
        )


def run_smoke_image(
    model: Any,
    processor: Any,
    image_path: Path,
    max_new_tokens: int = 256,
    verbose: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "image_path": str(image_path),
        "max_new_tokens": max_new_tokens,
    }
    if model is None or processor is None:
        result.update(
            {
                "category": "Model not loaded",
                "error": "--smoke-image requires a successfully loaded model; see the load_model section above",
            }
        )
        return result
    if not image_path.exists():
        result.update({"category": "File missing", "error": "smoke image does not exist"})
        return result

    start = time.time()
    try:
        import torch
        from qwen_vl_utils import process_vision_info

        prompt = build_smoke_prompt()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = parse_label_scores(raw_output)
        result.update(
            {
                "ok": bool(parsed.get("valid")),
                "raw_output": raw_output,
                "parsed_scores": parsed.get("scores", {}),
                "json_valid": bool(parsed.get("valid")),
                "parse_error": parsed.get("error"),
                "warnings": parsed.get("warnings", []),
                "inference_time_sec": time.time() - start,
            }
        )
        if not parsed.get("valid"):
            result["category"] = "Inference output JSON parsing failure"
    except Exception as exc:
        result.update(format_error(exc, verbose=verbose))
        result["inference_time_sec"] = time.time() - start
        result["hint"] = transformers_upgrade_hint(result.get("category", "")) or None
    return result


def print_section(title: str) -> None:
    print("\n=== {} ===".format(title))


def print_kv(key: str, value: Any) -> None:
    print("{:<38} {}".format(key + ":", value))


def print_report(report: Dict[str, Any]) -> None:
    files = report["files"]
    print_section("Local files")
    print_kv("model_path", files["model_path"])
    print_kv("exists", files["exists"])
    print_kv("file_count", files["file_count"])
    print_kv("safetensors_count", files["safetensors_count"])
    print_kv("safetensors_total_size", files["safetensors_total_size_human"])
    print_kv("model_type", files.get("config", {}).get("model_type"))
    print_kv("architectures", files.get("config", {}).get("architectures"))
    index = files.get("index", {})
    print_kv("index_weight_map_entries", index.get("weight_map_entries"))
    print_kv("index_referenced_shard_count", index.get("referenced_shard_count"))
    print_kv("missing_referenced_shards", index.get("missing_referenced_shards"))
    print_kv("zero_size_referenced_shards", index.get("zero_size_referenced_shards"))
    print_kv("suspicious_small_shards", index.get("suspicious_small_referenced_shards"))
    print_kv("file_check", "PASS" if files["pass"] else "FAIL")
    for warning in files.get("warnings", []):
        print("WARNING: {}".format(warning))
    for error in files.get("errors", []):
        print("ERROR: {}".format(error))

    tfm = report.get("transformers", {})
    print_section("Transformers offline metadata")
    print_kv("transformers_version", tfm.get("versions", {}).get("transformers"))
    print_kv("AutoConfig", "PASS" if tfm.get("auto_config", {}).get("ok") else "FAIL")
    if not tfm.get("auto_config", {}).get("ok"):
        print_kv("AutoConfig_category", tfm.get("auto_config", {}).get("category"))
        print_kv("AutoConfig_error", tfm.get("auto_config", {}).get("error"))
    print_kv("AutoProcessor", "PASS" if tfm.get("auto_processor", {}).get("ok") else "FAIL")
    if not tfm.get("auto_processor", {}).get("ok"):
        print_kv("AutoProcessor_category", tfm.get("auto_processor", {}).get("category"))
        print_kv("AutoProcessor_error", tfm.get("auto_processor", {}).get("error"))
    if tfm.get("upgrade_hint"):
        print("HINT: {}".format(tfm["upgrade_hint"]))

    if "load_model" in report:
        loaded = report["load_model"]
        print_section("Real model load")
        print_kv("load_model", "PASS" if loaded.get("ok") else "FAIL")
        runtime = loaded.get("runtime", {})
        print_kv("torch_version", runtime.get("torch_version"))
        print_kv("cuda_available", runtime.get("cuda_available"))
        print_kv("gpu_name", runtime.get("gpu_name"))
        print_kv("transformers_version", runtime.get("transformers_version"))
        print_kv("load_time_sec", loaded.get("load_time_sec"))
        print_kv("device_map", loaded.get("device_map"))
        print_kv("min_pixels", loaded.get("min_pixels"))
        print_kv("max_pixels", loaded.get("max_pixels"))
        print_kv("gpu_memory_reserved_after_load", loaded.get("gpu_memory_reserved_after_load"))
        if not loaded.get("ok"):
            print_kv("failure_category", loaded.get("category"))
            print_kv("error", loaded.get("error"))
            if loaded.get("hint"):
                print("HINT: {}".format(loaded["hint"]))

    if "smoke_image" in report:
        smoke = report["smoke_image"]
        print_section("Smoke image inference")
        print_kv("smoke_image", "PASS" if smoke.get("ok") else "FAIL")
        print_kv("json_valid", smoke.get("json_valid"))
        print_kv("inference_time_sec", smoke.get("inference_time_sec"))
        if smoke.get("raw_output") is not None:
            print_kv("raw_output", smoke.get("raw_output"))
        if smoke.get("parsed_scores") is not None:
            print_kv("parsed_scores", smoke.get("parsed_scores"))
        if not smoke.get("ok"):
            print_kv("failure_category", smoke.get("category"))
            print_kv("error", smoke.get("error") or smoke.get("parse_error"))
            if smoke.get("hint"):
                print("HINT: {}".format(smoke["hint"]))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a local Qwen2.5-VL-7B-Instruct model snapshot.")
    parser.add_argument("--model-path", required=True, help="Local model directory.")
    parser.add_argument("--load-model", action="store_true", help="Load the 7B model weights.")
    parser.add_argument("--smoke-image", help="Optional image path for one minimal JSON inference.")
    parser.add_argument("--device-map", default="auto", help="Device map for from_pretrained when --load-model is used.")
    parser.add_argument("--torch-dtype", default="auto", help="torch_dtype for from_pretrained when --load-model is used.")
    parser.add_argument("--attn-implementation", default="auto", help="Optional attention backend, e.g. auto or sdpa.")
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional Qwen-VL processor min_pixels.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Optional Qwen-VL processor max_pixels.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for --smoke-image.")
    parser.add_argument("--json", action="store_true", help="Print full machine-readable JSON report.")
    parser.add_argument("--verbose", action="store_true", help="Include tracebacks in JSON report.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    model_path = Path(args.model_path).expanduser().resolve()
    report: Dict[str, Any] = {
        "model": "Qwen2.5-VL-7B-Instruct",
        "files": check_local_files(model_path),
        "transformers": check_transformers_metadata(model_path, verbose=args.verbose),
    }

    model = None
    processor = None
    if args.load_model or args.smoke_image:
        load_result, model, processor = load_qwen_model(
            model_path,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            verbose=args.verbose,
        )
        report["load_model"] = load_result

    if args.smoke_image:
        report["smoke_image"] = run_smoke_image(
            model,
            processor,
            Path(args.smoke_image).expanduser().resolve(),
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print_report(report)

    hard_fail = not report["files"].get("pass", False)
    if not report["transformers"].get("auto_config", {}).get("ok"):
        hard_fail = True
    if not report["transformers"].get("auto_processor", {}).get("ok"):
        hard_fail = True
    if args.load_model and not report.get("load_model", {}).get("ok"):
        hard_fail = True
    if args.smoke_image and not report.get("smoke_image", {}).get("ok"):
        hard_fail = True
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
