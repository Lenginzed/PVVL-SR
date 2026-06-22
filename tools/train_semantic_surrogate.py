#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
import sys
from typing import Dict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import LABEL_NAMES
from lag_extensions.reward_shaping.logger import to_jsonable
from lag_extensions.reward_shaping.semantic_surrogate import (
    FEATURE_NAMES,
    FeatureNormalizer,
    build_mlp,
    load_surrogate_jsonl,
    rows_to_arrays,
    save_surrogate_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight semantic surrogate reward network.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="scripts/results/surrogate_models/stage8_formal100_smoke")
    parser.add_argument("--target-type", choices=["raw_vlm_scores", "verified_scores", "fused_scores"], default="fused_scores")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--normalize-features", action="store_true", default=True)
    parser.add_argument("--no-normalize-features", action="store_false", dest="normalize_features")
    args = parser.parse_args()
    summary = train(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    rows = load_surrogate_jsonl(args.dataset, args.target_type)
    if len(rows) < 5:
        raise ValueError("Need at least 5 surrogate rows, got {}".format(len(rows)))
    features, targets = rows_to_arrays(rows, args.target_type)
    indices = np.arange(len(rows))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    train_idx, val_idx, test_idx = _split_indices(indices)

    normalizer = FeatureNormalizer()
    if args.normalize_features:
        normalizer.fit(features[train_idx])
    x_train = normalizer.transform(features[train_idx])
    x_val = normalizer.transform(features[val_idx])
    x_test = normalizer.transform(features[test_idx])
    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

    model_config = {
        "input_dim": int(features.shape[1]),
        "output_dim": len(LABEL_NAMES),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "target_type": args.target_type,
        "normalize_features": bool(args.normalize_features),
        "feature_names": rows[0].get("state_feature_names", FEATURE_NAMES),
        "dataset": os.path.abspath(args.dataset),
        "seed": int(args.seed),
    }
    model = build_mlp(
        input_dim=model_config["input_dim"],
        output_dim=model_config["output_dim"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train)),
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.csv")
    log_rows = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        train_metrics = _eval_arrays(model, x_train, y_train, device)
        val_metrics = _eval_arrays(model, x_val, y_val, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "train_mae": train_metrics["overall_mae"],
            "val_mse": val_metrics["overall_mse"],
            "val_mae": val_metrics["overall_mae"],
            "val_max_error": val_metrics["max_error"],
        }
        log_rows.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            _write_csv(log_path, log_rows)

    final = {
        "num_rows": len(rows),
        "target_type": args.target_type,
        "device": str(device),
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "train": _eval_arrays(model, x_train, y_train, device),
        "val": _eval_arrays(model, x_val, y_val, device),
        "test": _eval_arrays(model, x_test, y_test, device),
        "output_range": _prediction_range(model, normalizer.transform(features), device),
    }
    final["test"]["per_label_mae"] = {
        LABEL_NAMES[i]: final["test"]["per_label_mae"][i]
        for i in range(len(LABEL_NAMES))
    }
    final["val"]["per_label_mae"] = {
        LABEL_NAMES[i]: final["val"]["per_label_mae"][i]
        for i in range(len(LABEL_NAMES))
    }
    final["train"]["per_label_mae"] = {
        LABEL_NAMES[i]: final["train"]["per_label_mae"][i]
        for i in range(len(LABEL_NAMES))
    }
    save_surrogate_checkpoint(args.output_dir, model, model_config, normalizer, final)
    _write_csv(log_path, log_rows)
    _write_json(os.path.join(args.output_dir, "split_indices.json"), {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    })
    return final


def _eval_arrays(model, features, targets, device) -> Dict:
    import torch

    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(features, dtype=torch.float32, device=device)).detach().cpu().numpy()
    err = pred - targets
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2))
    return {
        "overall_mse": float(np.mean(err ** 2)),
        "overall_mae": float(np.mean(np.abs(err))),
        "per_label_mae": [float(x) for x in np.mean(np.abs(err), axis=0)],
        "max_error": float(np.max(np.abs(err))) if err.size else 0.0,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0,
    }


def _prediction_range(model, features, device):
    import torch

    with torch.no_grad():
        pred = model(torch.as_tensor(features, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return {
        "min": float(np.min(pred)),
        "max": float(np.max(pred)),
        "out_of_range_count": int(np.sum((pred < -1e-6) | (pred > 1.0 + 1e-6))),
    }


def _split_indices(indices):
    n = len(indices)
    train_end = max(1, int(round(0.70 * n)))
    val_end = max(train_end + 1, int(round(0.85 * n)))
    val_end = min(val_end, n - 1)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _resolve_device(name):
    import torch

    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
