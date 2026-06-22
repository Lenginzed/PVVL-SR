import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .config import LABEL_NAMES, project_root
from .logger import to_jsonable
from .semantic_cache_retrieval import FEATURE_NAMES, feature_vector_from_metrics


MODEL_FILENAME = "surrogate_model.pt"
CONFIG_FILENAME = "surrogate_config.json"
METRICS_FILENAME = "final_metrics.json"


class FeatureNormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = np.asarray(mean, dtype=np.float32) if mean is not None else None
        self.std = np.asarray(std, dtype=np.float32) if std is not None else None

    def fit(self, features: np.ndarray):
        self.mean = features.mean(axis=0).astype(np.float32)
        self.std = features.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return features.astype(np.float32)
        return ((features.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict]):
        payload = payload or {}
        return cls(payload.get("mean"), payload.get("std"))


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float):
    import torch
    import torch.nn as nn

    layers = []
    current_dim = int(input_dim)
    for _ in range(max(int(num_layers) - 1, 1)):
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.SiLU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, int(output_dim)))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class SemanticSurrogatePredictor:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = _resolve_path(model_dir)
        self.device = device
        self.model = None
        self.config = {}
        self.normalizer = FeatureNormalizer()
        self.feature_names = FEATURE_NAMES[:]
        self.label_names = LABEL_NAMES[:]
        self.last_info = {}
        self.load()

    def load(self):
        import torch

        checkpoint_path = os.path.join(self.model_dir, MODEL_FILENAME)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Surrogate checkpoint not found: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = dict(checkpoint.get("config", {}))
        self.feature_names = checkpoint.get("feature_names") or self.config.get("feature_names") or FEATURE_NAMES[:]
        self.label_names = checkpoint.get("label_names") or LABEL_NAMES[:]
        self.normalizer = FeatureNormalizer.from_dict(checkpoint.get("normalizer"))
        self.model = build_mlp(
            input_dim=int(self.config.get("input_dim", len(self.feature_names))),
            output_dim=int(self.config.get("output_dim", len(self.label_names))),
            hidden_dim=int(self.config.get("hidden_dim", 128)),
            num_layers=int(self.config.get("num_layers", 3)),
            dropout=float(self.config.get("dropout", 0.0)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.last_info = {
            "model_dir": self.model_dir,
            "device": self.device,
            "target_type": self.config.get("target_type"),
            "loaded": True,
        }
        return self

    def predict_features(self, features: List[float]) -> Dict[str, float]:
        import torch

        start = time.time()
        array = np.asarray(features, dtype=np.float32).reshape(1, -1)
        normalized = self.normalizer.transform(array)
        with torch.no_grad():
            tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            prediction = self.model(tensor).detach().cpu().numpy()[0]
        scores = {
            name: float(np.clip(prediction[i], 0.0, 1.0))
            for i, name in enumerate(self.label_names)
        }
        self.last_info = {
            "source": "surrogate",
            "model_dir": self.model_dir,
            "target_type": self.config.get("target_type"),
            "device": self.device,
            "inference_time_sec": time.time() - start,
            "feature_dim": int(array.shape[1]),
            "loaded": True,
        }
        return scores

    def predict_from_metrics(self, metrics: Dict, state: Optional[Dict] = None) -> Dict[str, float]:
        features = feature_vector_from_metrics(metrics, state)
        return self.predict_features(features)


def load_surrogate_jsonl(path: str, target_type: str = "fused_scores"):
    path = _resolve_path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            features = item.get("state_features")
            target = item.get(target_type)
            if not features or not isinstance(target, dict):
                continue
            rows.append(item)
    return rows


def rows_to_arrays(rows: List[Dict], target_type: str):
    features = np.asarray([row["state_features"] for row in rows], dtype=np.float32)
    targets = np.asarray([
        [float(row[target_type].get(name, 0.0)) for name in LABEL_NAMES]
        for row in rows
    ], dtype=np.float32)
    return features, np.clip(targets, 0.0, 1.0)


def save_surrogate_checkpoint(output_dir: str, model, config: Dict, normalizer: FeatureNormalizer, metrics: Dict):
    import torch

    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "normalizer": normalizer.to_dict(),
        "feature_names": config.get("feature_names", FEATURE_NAMES),
        "label_names": LABEL_NAMES,
    }
    torch.save(checkpoint, os.path.join(output_dir, MODEL_FILENAME))
    _write_json(os.path.join(output_dir, CONFIG_FILENAME), config)
    _write_json(os.path.join(output_dir, METRICS_FILENAME), metrics)


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root(), path))
