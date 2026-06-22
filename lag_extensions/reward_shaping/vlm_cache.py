import hashlib
import json
import os
from typing import Dict, Optional

from .config import LABEL_NAMES
from .config import project_root


class VLMScoreCache:
    def __init__(self, path: str, enabled: bool = True):
        self.path = _resolve_output_path(path)
        self.enabled = enabled
        self.entries = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.enabled:
            self._load()

    def get(self, key: str) -> Optional[Dict]:
        if not self.enabled:
            self.misses += 1
            return None
        entry = self.entries.get(str(key))
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def set(self, key: str, entry: Dict) -> Dict:
        if not self.enabled:
            return entry
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        item = dict(entry)
        item["cache_key"] = str(key)
        item["key"] = str(key)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, sort_keys=True) + "\n")
        self._add_entry(item)
        self.writes += 1
        return item

    def score_with_cache(
        self,
        scorer,
        image_path: str,
        prompt: str,
        metadata: Dict,
        physical_scores: Dict[str, float],
        model_config: Optional[Dict] = None,
        fallback_to_physical: bool = True,
    ) -> Dict:
        key = make_vlm_cache_key(metadata, prompt, scorer.model_name_or_path, model_config or {})
        cached = self.get(key)
        if cached is not None:
            item = dict(cached)
            item["cache_hit"] = True
            return item

        result = scorer.score_image(
            image_path=image_path,
            prompt=prompt,
            image_id=metadata.get("image_id"),
            physical_scores=physical_scores,
        )
        result.update({
            "cache_hit": False,
            "metadata": {
                "image_id": metadata.get("image_id"),
                "image_hash": metadata.get("image_hash"),
                "state_hash": metadata.get("state_hash"),
                "renderer_version": metadata.get("renderer_version"),
                "prompt_version": metadata.get("prompt_version"),
            },
            "state_hash": metadata.get("state_hash"),
            "image_hash": metadata.get("image_hash"),
        })
        if not result.get("valid", False) and fallback_to_physical:
            result["scores"] = {name: float(physical_scores.get(name, 0.0)) for name in LABEL_NAMES}
            result["source"] = "cached_vlm_fallback_physical"
            result["fallback_reason"] = result.get("error") or "invalid_vlm_output"
        else:
            result["source"] = "cached_vlm"
            result["fallback_reason"] = None
        return self.set(key, result)

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": self.hits / total if total else 0.0,
            "size": len(self.entries),
        }

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    self._add_entry(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def _add_entry(self, entry: Dict) -> None:
        aliases = [
            entry.get("cache_key"),
            entry.get("key"),
            entry.get("image_id"),
            entry.get("image_hash"),
            entry.get("state_hash"),
        ]
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
        aliases.extend([
            metadata.get("image_id"),
            metadata.get("image_hash"),
            metadata.get("state_hash"),
        ])
        for alias in aliases:
            if alias:
                self.entries[str(alias)] = entry


def make_vlm_cache_key(metadata: Dict, prompt: str, model_name: str, model_config: Dict) -> str:
    payload = {
        "image_hash": metadata.get("image_hash"),
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_version": metadata.get("prompt_version"),
        "model_name": model_name,
        "model_config": model_config,
        "renderer_version": metadata.get("renderer_version"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "vlm:" + hashlib.sha256(encoded).hexdigest()


def _resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root(), path))
