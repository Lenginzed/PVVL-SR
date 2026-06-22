import json
import os
import re
import time
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .config import LABEL_NAMES


class QwenVLMScorer:
    """Offline scorer for Qwen/Qwen2.5-VL-7B-Instruct.

    Dependencies for real inference are intentionally lazy:
    pip install transformers accelerate qwen-vl-utils
    A CUDA device with sufficient memory is recommended for the 7B model.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "auto",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 256,
        load_mode: str = "auto",
        device_map: Optional[str] = None,
        local_files_only: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        mock: bool = False,
        load_model: bool = True,
    ):
        self.model_name_or_path = self._resolve_model_path(model_name_or_path)
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.load_mode = load_mode
        self.device_map = device_map
        self.local_files_only = local_files_only
        self.attn_implementation = attn_implementation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.mock = mock
        self.model = None
        self.processor = None
        self.load_count = 0
        self.last_generation_info = {}
        self.last_runtime_diagnostics = {}
        if load_model and not mock:
            self.load()

    @property
    def model_id(self) -> str:
        return "Qwen2.5-VL-7B-Instruct"

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL inference requires transformers, accelerate, and qwen-vl-utils. "
                "Use --dry-run or --mock-vlm to test the offline pipeline without these dependencies."
            ) from exc

        torch_dtype = self._torch_dtype(torch)
        device_map, move_to_device = self._resolve_load_strategy()
        attn_implementation = self.attn_implementation
        if attn_implementation is None:
            attn_implementation = None
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "local_files_only": self.local_files_only,
        }
        if attn_implementation and attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        if move_to_device:
            self.model.to(self.device)
        self.load_processor(self.min_pixels, self.max_pixels)
        self.load_count += 1
        self.last_runtime_diagnostics = self.runtime_diagnostics()

    def load_processor(self, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> None:
        from transformers import AutoProcessor

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        processor_kwargs = {"local_files_only": self.local_files_only}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = int(self.max_pixels)
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, **processor_kwargs)

    def set_vision_pixels(self, min_pixels: Optional[int], max_pixels: Optional[int]) -> None:
        if self.mock:
            self.min_pixels = min_pixels
            self.max_pixels = max_pixels
            return
        if self.processor is None or min_pixels != self.min_pixels or max_pixels != self.max_pixels:
            self.load_processor(min_pixels, max_pixels)

    def score_image(
        self,
        image_path: str,
        prompt: str,
        image_id: Optional[str] = None,
        physical_scores: Optional[Dict[str, float]] = None,
    ) -> Dict:
        start = time.time()
        if self.mock:
            scores = self._mock_scores(image_id or image_path, physical_scores)
            raw_output = json.dumps(scores, sort_keys=True)
            return {
                "image_id": image_id,
                "model": self.model_id,
                "valid": True,
                "scores": scores,
                "raw_output": raw_output,
                "error": None,
                "warnings": [],
                "inference_time_sec": time.time() - start,
                "generation_info": {
                    "timings": {"total_time": time.time() - start},
                    "output_truncated": False,
                    "missing_keys": [],
                    "invalid_json": False,
                },
            }

        if self.model is None or self.processor is None:
            self.load()

        raw_output = ""
        try:
            raw_output = self._generate(image_path, prompt)
            parse_start = time.time()
            parsed = self.parse_scores(raw_output)
            parse_time = time.time() - parse_start
            missing_keys = list(parsed.get("missing_keys", []))
            invalid_json = bool(parsed.get("invalid_json", not parsed.get("valid", False)))
            output_truncated = self._looks_truncated(raw_output, parsed)
            self.last_generation_info.setdefault("timings", {})["json_parse_time"] = parse_time
            self.last_generation_info["timings"]["total_time"] = time.time() - start
            self.last_generation_info["missing_keys"] = missing_keys
            self.last_generation_info["invalid_json"] = invalid_json
            self.last_generation_info["output_truncated"] = output_truncated
            if output_truncated:
                parsed.setdefault("warnings", []).append("output_may_be_truncated")
            parsed.update({
                "image_id": image_id,
                "model": self.model_id,
                "raw_output": raw_output,
                "inference_time_sec": time.time() - start,
                "generation_info": dict(self.last_generation_info),
            })
            return parsed
        except Exception as exc:
            return {
                "image_id": image_id,
                "model": self.model_id,
                "valid": False,
                "scores": {},
                "raw_output": raw_output,
                "error": str(exc),
                "warnings": [],
                "inference_time_sec": time.time() - start,
                "generation_info": dict(self.last_generation_info),
            }

    def _generate(self, image_path: str, prompt: str) -> str:
        import torch

        timings = {}
        total_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        message_start = time.time()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        timings["image_load_time"] = time.time() - message_start
        prompt_start = time.time()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        timings["prompt_build_time"] = time.time() - prompt_start
        preprocess_start = time.time()
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            image_load_start = time.time()
            image = Image.open(image_path).convert("RGB")
            timings["image_load_time"] += time.time() - image_load_start
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        timings["preprocess_time"] = time.time() - preprocess_start
        transfer_start = time.time()
        inputs = inputs.to(self.model.device)
        timings["input_transfer_time"] = time.time() - transfer_start
        input_token_count = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else None
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs.update({"temperature": self.temperature, "top_p": self.top_p})
        generate_start = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        timings["generate_time"] = time.time() - generate_start
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_token_count = int(trimmed[0].shape[-1]) if trimmed else None
        decode_start = time.time()
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        timings["decode_time"] = time.time() - decode_start
        timings.setdefault("json_parse_time", 0.0)
        timings["total_time"] = time.time() - total_start
        memory = self.cuda_memory_stats()
        self.last_generation_info = {
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "timings": timings,
            "gpu_memory": memory,
        }
        return decoded

    @staticmethod
    def parse_scores(raw_output: str) -> Dict:
        error = None
        warnings = []
        payload = None
        invalid_json = False
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    error = "json_parse_failed: {}".format(exc)
                    invalid_json = True
            else:
                error = "json_object_not_found"
                invalid_json = True

        if not isinstance(payload, dict):
            return {
                "valid": False,
                "scores": {},
                "error": error or "json_not_object",
                "warnings": warnings,
                "missing_keys": LABEL_NAMES[:],
                "invalid_json": invalid_json or True,
            }

        scores = {}
        missing = []
        for name in LABEL_NAMES:
            if name not in payload:
                missing.append(name)
                continue
            try:
                value = float(payload[name])
            except (TypeError, ValueError):
                missing.append(name)
                continue
            clipped = float(np.clip(value, 0.0, 1.0))
            if clipped != value:
                warnings.append("clipped {} from {}".format(name, value))
            scores[name] = clipped

        valid = len(missing) == 0
        if missing:
            error = "missing_or_invalid_labels: {}".format(",".join(missing))
        return {
            "valid": valid,
            "scores": scores,
            "error": error,
            "warnings": warnings,
            "missing_keys": missing,
            "invalid_json": False,
        }

    def _mock_scores(self, key: str, physical_scores: Optional[Dict[str, float]]) -> Dict[str, float]:
        if physical_scores:
            return {name: float(np.clip(physical_scores.get(name, 0.0), 0.0, 1.0)) for name in LABEL_NAMES}
        seed = abs(hash(key)) % (2 ** 32)
        rng = np.random.RandomState(seed)
        return {name: float(rng.rand()) for name in LABEL_NAMES}

    def _torch_dtype(self, torch):
        if self.dtype == "auto":
            return "auto"
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        raise ValueError("Unsupported dtype: {}".format(self.dtype))

    def _resolve_load_strategy(self) -> Tuple[object, bool]:
        mode = self.load_mode or "auto"
        if mode == "auto":
            return self.device_map or "auto", False
        if mode == "balanced":
            return self.device_map or "balanced", False
        if mode == "explicit_single_gpu":
            return {"": self._single_gpu_device()}, False
        if mode == "cuda":
            return None, True
        raise ValueError("Unsupported load_mode: {}".format(mode))

    def _single_gpu_device(self):
        if isinstance(self.device, str) and self.device.startswith("cuda:"):
            try:
                return int(self.device.split(":", 1)[1])
            except ValueError:
                return self.device
        if self.device == "cuda":
            return 0
        return self.device

    def runtime_diagnostics(self) -> Dict:
        diagnostics = {
            "load_mode": self.load_mode,
            "device": self.device,
            "device_map": self.device_map,
            "model_name_or_path": self.model_name_or_path,
            "torch_dtype": self.dtype,
            "load_count": self.load_count,
            "module_devices": {},
            "hf_device_map": getattr(self.model, "hf_device_map", None) if self.model is not None else None,
            "cpu_param_count": 0,
            "cuda_param_count": 0,
            "meta_param_count": 0,
            "other_param_count": 0,
            "cpu_offload_detected": False,
            "cuda_memory": self.cuda_memory_stats(),
        }
        if self.model is None:
            return diagnostics
        for name, module in self.model.named_children():
            diagnostics["module_devices"][name] = self._module_device_summary(module)
        for param in self.model.parameters():
            count = int(param.numel())
            device_type = param.device.type
            if device_type == "cpu":
                diagnostics["cpu_param_count"] += count
            elif device_type == "cuda":
                diagnostics["cuda_param_count"] += count
            elif device_type == "meta":
                diagnostics["meta_param_count"] += count
            else:
                diagnostics["other_param_count"] += count
        hf_map = diagnostics["hf_device_map"] or {}
        map_values = [str(value) for value in hf_map.values()] if isinstance(hf_map, dict) else []
        diagnostics["cpu_offload_detected"] = (
            diagnostics["cpu_param_count"] > 0
            or diagnostics["meta_param_count"] > 0
            or any(value == "cpu" or value == "disk" for value in map_values)
        )
        return diagnostics

    @staticmethod
    def _module_device_summary(module) -> Dict:
        counts = {"cpu": 0, "cuda": 0, "meta": 0, "other": 0}
        for param in module.parameters(recurse=True):
            key = param.device.type if param.device.type in counts else "other"
            counts[key] += int(param.numel())
        dominant = max(counts, key=counts.get) if sum(counts.values()) else "none"
        counts["dominant_device"] = dominant
        return counts

    @staticmethod
    def cuda_memory_stats() -> Dict:
        try:
            import torch
        except Exception:
            return {}
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return {
            "cuda_available": True,
            "gpu_name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "total_memory_gib": float(props.total_memory / (1024 ** 3)),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "allocated_gib": float(torch.cuda.memory_allocated(device) / (1024 ** 3)),
            "reserved_gib": float(torch.cuda.memory_reserved(device) / (1024 ** 3)),
            "max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
            "max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / (1024 ** 3)),
        }

    @staticmethod
    def _looks_truncated(raw_output: str, parsed: Dict) -> bool:
        if parsed.get("valid"):
            return False
        stripped = (raw_output or "").strip()
        if not stripped:
            return False
        if parsed.get("missing_keys") and not stripped.endswith(("}", "```")):
            return True
        return stripped.count("{") > stripped.count("}")

    @staticmethod
    def _resolve_model_path(model_name_or_path: str) -> str:
        if os.path.isabs(model_name_or_path) or os.path.exists(model_name_or_path):
            return model_name_or_path
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate = os.path.join(root, model_name_or_path)
        if os.path.exists(candidate):
            return candidate
        return model_name_or_path
