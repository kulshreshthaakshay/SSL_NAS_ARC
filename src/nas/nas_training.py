import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.nas.controller import NASController
from src.ssl.model import SSLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NASTrainer:
    def __init__(
        self,
        ssl_checkpoint_path="models/ssl_best_model.pt",
        batch_size=32,
        w_lr=1e-3,
        a_lr=3e-4,
        entropy_reg=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.ssl_checkpoint_path = ssl_checkpoint_path
        self.device = device
        self.batch_size = batch_size
        self.entropy_reg = entropy_reg
        self.encoder, self.ssl_info = self._load_pretrained_encoder(ssl_checkpoint_path)
        encoder_dim = self._get_encoder_output_dim()
        self.controller = NASController(
            input_dim=encoder_dim,
            hidden_dim=64,
            num_classes=2,
            temperature=5.0,
        ).to(device)

        arch_params = list(self.controller.arch_parameters())
        arch_param_ids = {id(p) for p in arch_params}
        weight_params = [p for p in self.controller.parameters() if id(p) not in arch_param_ids]
        self.w_optimizer = optim.Adam(weight_params, lr=w_lr)
        self.a_optimizer = optim.Adam(arch_params, lr=a_lr)
        self.arch_weights_history = []
        logger.info(f"Initialized NAS controller with temporal encoder dim: {encoder_dim}")

    def _load_pretrained_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        input_dim = checkpoint.get("input_dim", 5)
        ssl_model = SSLModel(input_dim=input_dim).to(self.device)
        ssl_model.load_state_dict(checkpoint["model_state_dict"])
        ssl_model.eval()

        class SequenceEncoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.encode_sequence(x)

        encoder = SequenceEncoder(ssl_model)
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info(f"Loaded frozen SSL sequence encoder from {checkpoint_path}")
        return encoder, checkpoint

    def _get_encoder_output_dim(self):
        input_dim = self.ssl_info.get("input_dim", 5)
        seq_len = self.ssl_info.get("seq_len", 30)
        dummy_input = torch.randn(1, seq_len, input_dim).to(self.device)
        with torch.no_grad():
            output = self.encoder(dummy_input)
        if output.dim() != 3:
            raise ValueError(f"Expected sequence encoder output, got {tuple(output.shape)}")
        return output.shape[-1]

    def train(self, weight_data, arch_data, valid_data, epochs=50, early_stopping_patience=10):
        weight_loader = DataLoader(weight_data, batch_size=self.batch_size, shuffle=True)
        arch_loader = DataLoader(arch_data, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False)

        self.controller.max_epochs = epochs
        best_valid_acc = -float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.controller.set_temperature(epoch)
            self.controller.train()
            arch_loss = self._update_architecture_params(arch_loader)
            weight_loss, train_acc = self._update_weight_params(weight_loader)

            self.controller.eval()
            valid_loss, valid_metrics = self._evaluate_loader(valid_loader)
            valid_acc = valid_metrics["accuracy"]
            arch_info = self.controller.get_arch_info()
            self.arch_weights_history.append({
                "epoch": epoch + 1,
                "temperature": self.controller.temperature,
                **arch_info,
            })

            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(
                f"Temp: {self.controller.temperature:.4f}, Arch Loss: {arch_loss:.4f}, "
                f"Weight Loss: {weight_loss:.4f}"
            )
            logger.info(f"Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")
            logger.info(f"Architecture Weights: {dict(sorted(arch_info.items(), key=lambda x: x[1], reverse=True))}")

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                patience_counter = 0
                self._save_checkpoint(epoch, valid_loss, valid_metrics)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        self._save_arch_history()
        self._log_final_validation(valid_loader, best_valid_acc)

    def _features(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        if features.dim() != 3 or features.size(-1) != self.controller.input_dim:
            raise ValueError(
                f"Expected features (batch, seq, {self.controller.input_dim}), got {tuple(features.shape)}"
            )
        return features

    def _update_architecture_params(self, arch_loader):
        total_loss = 0.0
        steps = 0
        for x, y in arch_loader:
            x, y = x.to(self.device), y.to(self.device)
            features = self._features(x)
            self.a_optimizer.zero_grad()
            logits = self.controller(features, mode="sample")
            loss = nn.CrossEntropyLoss()(logits, y)
            alpha_weights = torch.softmax(self.controller.alpha, dim=-1)
            entropy = -torch.sum(alpha_weights * torch.log(alpha_weights + 1e-8))
            loss = loss + self.entropy_reg * entropy
            loss.backward()
            self.a_optimizer.step()
            total_loss += loss.item()
            steps += 1
        return total_loss / steps if steps else 0.0

    def _update_weight_params(self, weight_loader):
        total_loss = 0.0
        all_preds = []
        all_labels = []
        steps = 0
        for x, y in tqdm(weight_loader, desc="Training Weights"):
            x, y = x.to(self.device), y.to(self.device)
            features = self._features(x)
            self.w_optimizer.zero_grad()
            logits = self.controller(features, mode="sample")
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            weight_params = [p for name, p in self.controller.named_parameters() if name != "alpha"]
            torch.nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
            self.w_optimizer.step()

            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            total_loss += loss.item()
            steps += 1
        return total_loss / steps if steps else 0.0, accuracy_score(all_labels, all_preds)

    def _evaluate_loader(self, loader):
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        steps = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                features = self._features(x)
                logits = self.controller(features, mode="softmax")
                loss = nn.CrossEntropyLoss()(logits, y)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                total_loss += loss.item()
                steps += 1
        metrics = self._classification_metrics(all_labels, all_preds, all_probs)
        return total_loss / steps if steps else float("inf"), metrics

    @staticmethod
    def _classification_metrics(labels, preds, probs):
        metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }
        try:
            metrics["auc"] = float(roc_auc_score(labels, probs))
        except Exception:
            metrics["auc"] = None
        return metrics

    def _save_checkpoint(self, epoch, valid_loss, valid_metrics):
        models_dir = Path(__file__).parent.parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.controller.state_dict(),
            "w_optimizer_state_dict": self.w_optimizer.state_dict(),
            "a_optimizer_state_dict": self.a_optimizer.state_dict(),
            "valid_loss": valid_loss,
            "valid_metrics": valid_metrics,
            "valid_acc": valid_metrics["accuracy"],
            "arch_info": self.controller.get_arch_info(),
            "best_arch": self.controller.discretize(),
            "temperature": self.controller.temperature,
            "eval_weight_mode": "softmax",
            "ssl_checkpoint_path": self.ssl_checkpoint_path,
            "encoder_output": "sequence",
        }
        torch.save(checkpoint, models_dir / "nas_best_model.pt")
        logger.info(f"Saved NAS checkpoint - Acc: {valid_metrics['accuracy']:.4f}")

    def _save_arch_history(self):
        results_dir = Path(__file__).parent.parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "arch_weights_history.json", "w") as f:
            json.dump(self.arch_weights_history, f, indent=2)
        logger.info(f"Saved architecture weights history to {results_dir / 'arch_weights_history.json'}")

    def _log_final_validation(self, valid_loader, best_valid_acc):
        _, metrics = self._evaluate_loader(valid_loader)
        arch_info = self.controller.get_arch_info()
        max_arch_weight = max(arch_info.values())
        logger.info("\n=== Final Evaluation Results ===")
        logger.info(f"Selected architecture: {self.controller.discretize()}")
        logger.info(f"Best validation accuracy: {best_valid_acc:.4f}")
        if max_arch_weight < 0.35:
            logger.warning(
                f"Weak NAS search signal: max architecture weight is {max_arch_weight:.4f} "
                "(threshold=0.35)"
            )
        logger.info(f"Final validation metrics: {metrics}")
        logger.info("================================")

    def prepare_financial_data(self):
        project_root = Path(__file__).parent.parent.parent
        processed_dir = project_root / "data" / "processed"
        paths = {
            "weight": (processed_dir / "windows_train.npy", processed_dir / "labels_train.npy"),
            "arch": (processed_dir / "windows_arch.npy", processed_dir / "labels_arch.npy"),
            "valid": (processed_dir / "windows_val.npy", processed_dir / "labels_val.npy"),
        }
        missing = [p for pair in paths.values() for p in pair if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing NAS data artifacts: {missing}")

        datasets = []
        for split, (windows_path, labels_path) in paths.items():
            windows = np.load(windows_path)
            labels = np.load(labels_path).astype(np.int64)
            if len(windows) != len(labels):
                raise ValueError(f"{split} windows/labels mismatch: {len(windows)} vs {len(labels)}")
            logger.info(f"Loaded {split}: windows={windows.shape}, labels={labels.shape}")
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"{split} label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
            datasets.append(TensorDataset(torch.FloatTensor(windows), torch.LongTensor(labels)))
        return tuple(datasets)
