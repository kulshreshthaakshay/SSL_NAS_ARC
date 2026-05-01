"""
Deterministic evaluation for the corrected SSL-NAS framework.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.nas.controller import NASController
from src.ssl.model import SSLModel
from src.utils.financial_metrics import compute_financial_metrics, load_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_NAME = "future_30d_direction"
FORECAST_HORIZON = 30


def get_project_root():
    return Path(__file__).parent


class SequenceEncoderWrapper(nn.Module):
    def __init__(self, ssl_model):
        super().__init__()
        self.ssl_model = ssl_model

    def forward(self, x):
        return self.ssl_model.encode_sequence(x)


def load_models(device="cpu"):
    project_root = get_project_root()
    ssl_checkpoint_path = project_root / "models" / "ssl_best_model.pt"
    nas_checkpoint_path = project_root / "models" / "nas_best_model.pt"
    if not ssl_checkpoint_path.exists():
        raise FileNotFoundError(f"SSL model not found at {ssl_checkpoint_path}")
    if not nas_checkpoint_path.exists():
        raise FileNotFoundError(f"NAS model not found at {nas_checkpoint_path}")

    ssl_checkpoint = torch.load(ssl_checkpoint_path, map_location=device)
    input_dim = ssl_checkpoint.get("input_dim", 5)
    seq_len = ssl_checkpoint.get("seq_len", 30)
    ssl_model = SSLModel(input_dim=input_dim).to(device)
    ssl_model.load_state_dict(ssl_checkpoint["model_state_dict"])
    ssl_model.eval()
    encoder = SequenceEncoderWrapper(ssl_model)
    for param in encoder.parameters():
        param.requires_grad = False

    with torch.no_grad():
        dummy = torch.randn(1, seq_len, input_dim).to(device)
        encoder_dim = encoder(dummy).shape[-1]

    nas_checkpoint = torch.load(nas_checkpoint_path, map_location=device)
    controller = NASController(
        input_dim=encoder_dim,
        hidden_dim=64,
        num_classes=2,
        temperature=nas_checkpoint.get("temperature", 0.1),
    ).to(device)
    controller.load_state_dict(nas_checkpoint["model_state_dict"])
    controller.eval()
    logger.info(f"Loaded NAS controller. Best architecture: {nas_checkpoint.get('best_arch', 'unknown')}")
    return encoder, controller


def load_test_data():
    processed = get_project_root() / "data" / "processed"
    paths = {
        "windows": processed / "windows_test.npy",
        "labels": processed / "labels_test.npy",
        "returns": processed / "future_returns_test.npy",
        "metadata": processed / "sample_metadata_test.csv",
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing test artifacts: {missing}")
    windows = np.load(paths["windows"])
    labels = np.load(paths["labels"]).astype(np.int64)
    returns = np.load(paths["returns"])
    metadata = load_metadata(paths["metadata"])
    if not (len(windows) == len(labels) == len(returns) == len(metadata)):
        raise ValueError("Test windows, labels, returns, and metadata are misaligned")
    return windows, labels, returns, metadata


def compute_classification_metrics(encoder, controller, windows, labels, device="cpu"):
    x = torch.FloatTensor(windows).to(device)
    all_preds = []
    all_probs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            features = encoder(x[i:i + batch_size])
            logits = controller(features, mode="softmax")
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    all_preds = np.asarray(all_preds)
    all_probs = np.asarray(all_probs)
    metrics = {
        "accuracy": float(accuracy_score(labels, all_preds)),
        "precision": float(precision_score(labels, all_preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, all_preds, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(labels, all_preds, average="weighted", zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, all_probs))
    except Exception as exc:
        logger.warning(f"Could not compute AUC: {exc}")
        metrics["auc"] = None
    return metrics, all_preds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    encoder, controller = load_models(device)
    windows, labels, actual_returns, metadata = load_test_data()
    class_metrics, predictions = compute_classification_metrics(
        encoder, controller, windows, labels, device
    )
    fin_metrics = compute_financial_metrics(
        predictions,
        labels,
        actual_returns,
        metadata,
        forecast_horizon=FORECAST_HORIZON,
    )

    unique, counts = np.unique(labels, return_counts=True)
    results = {
        "task": TASK_NAME,
        "forecast_horizon_days": FORECAST_HORIZON,
        "classification_metrics": class_metrics,
        "financial_metrics": fin_metrics,
        "test_set_size": int(len(windows)),
        "label_distribution": {str(k): int(v) for k, v in zip(unique, counts)},
        "eval_weight_mode": "softmax",
    }

    results_dir = get_project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "evaluation_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    logger.info(f"Results saved to {results_path}")
    return results


if __name__ == "__main__":
    main()
