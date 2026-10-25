"""PyTorch GRU model, training loop, checkpoint resume and evaluation."""

from __future__ import annotations

import json
import hashlib
import os
import random
import copy
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .training_data import PreparedTrainingData


def cuda_architecture_is_supported(
    capability: tuple[int, int],
    supported_architectures: list[str],
) -> bool:
    expected = f"sm_{capability[0]}{capability[1]}"
    return not supported_architectures or expected in supported_architectures


def resolve_training_device(training_config: dict) -> torch.device:
    requested = str(training_config.get("device", "cuda")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "cuda":
        raise ValueError("training.device must be either 'cuda' or 'cpu'")
    if not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is not available")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    supported = torch.cuda.get_arch_list()
    expected = f"sm_{capability[0]}{capability[1]}"
    if not cuda_architecture_is_supported(capability, supported):
        raise RuntimeError(
            f"GPU {name} ({expected}) is incompatible with this PyTorch build. "
            f"Supported CUDA architectures: {supported}. "
            "In Kaggle Settings, select GPU T4 x2 and restart the session."
        )
    try:
        probe = torch.ones(1, device="cuda")
        _ = float((probe + 1).item())
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            f"CUDA preflight failed on {name} ({expected}): {exc}. "
            "Restart the Kaggle session with GPU T4 x2."
        ) from exc
    return torch.device("cuda")


def set_global_seed(seed: int, include_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _capture_rng_state(include_cuda: bool = False) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if include_cuda else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    except Exception as exc:
        raise ValueError(f"Cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    return value


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_history_save(history_rows: list[dict[str, float | int]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        pd.DataFrame(history_rows).to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scaler_fingerprint(data: PreparedTrainingData) -> str:
    digest = hashlib.sha256()
    for name in data.feature_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    digest.update(data.standardizer.means.tobytes())
    digest.update(data.standardizer.scales.tobytes())
    digest.update(data.standardizer.scaled.tobytes())
    return digest.hexdigest()


def _resume_signature(data: PreparedTrainingData, config: dict) -> dict[str, Any]:
    training_invariants = {
        key: value
        for key, value in config["training"].items()
        if key not in {"epochs", "num_workers"}
    }
    signature = {
        "feature_names": data.feature_names,
        "model": config["model"],
        "expected_window_size": int(config["data"]["expected_window_size"]),
        "allowed_labels": [int(value) for value in config["data"]["allowed_labels"]],
        "reaction_margin_frames": int(data.margin_frames),
        "window_counts": {split: len(frame) for split, frame in data.windows.items()},
        "training": training_invariants,
        "scaler_sha256": _scaler_fingerprint(data),
    }
    # Keep the original split signature byte-for-byte compatible with baseline
    # checkpoints created before participant-fold training was added.
    if data.development_fold is not None:
        signature["development_fold"] = int(data.development_fold)
    return signature


def _resolve_resume_path(output_dir: Path, resume: str) -> Path | None:
    last_path = output_dir / "last_checkpoint.pt"
    normalized = resume.strip()
    if normalized == "none":
        existing = [
            path for path in (last_path, output_dir / "best_model.pt")
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"Output already contains checkpoints: {existing}. "
                "Use --resume auto or choose a new output directory."
            )
        return None
    if normalized == "auto":
        if last_path.exists():
            return last_path
        legacy_best = output_dir / "best_model.pt"
        if legacy_best.exists():
            raise FileExistsError(
                f"Found {legacy_best} but no resumable last_checkpoint.pt. "
                "Choose a new output directory or remove the legacy output explicitly."
            )
        return None
    candidate = Path(normalized).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")
    return candidate.resolve()


def _validate_resume_checkpoint(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("Unsupported or missing checkpoint_version")
    if checkpoint.get("resume_signature") != expected:
        raise ValueError(
            "Checkpoint is incompatible with the selected features, scaler, windows, model, or training config"
        )
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "grad_scaler_state_dict", "best_model_state_dict", "best_score", "best_epoch",
        "epochs_without_improvement", "stopped_early", "history_rows", "rng_state",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing resume state: {missing}")


class IndexedWindowDataset(Dataset):
    """Return contiguous windows from one shared frame-level feature matrix."""

    def __init__(self, matrix: np.ndarray, windows: pd.DataFrame, window_size: int) -> None:
        self.matrix = matrix
        self.starts = windows["start_row"].to_numpy(dtype=np.int64, copy=True)
        self.ends = windows["end_row_exclusive"].to_numpy(dtype=np.int64, copy=True)
        self.labels = windows["label"].to_numpy(dtype=np.int64, copy=True)
        self.window_size = int(window_size)
        if not np.all(self.ends - self.starts == self.window_size):
            raise ValueError("Dataset received windows with inconsistent lengths")
        if len(self.starts) and (self.starts.min() < 0 or self.ends.max() > len(matrix)):
            raise ValueError("Window row bounds exceed the feature matrix")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = int(self.starts[index]), int(self.ends[index])
        features = torch.from_numpy(self.matrix[start:end])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return features, label


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        bidirectional: bool = False,
        pooling: str = "last",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"last", "attention"}:
            raise ValueError("model.pooling must be either 'last' or 'attention'")
        self.pooling = pooling
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.output_norm = nn.LayerNorm(output_size) if layer_norm else nn.Identity()
        self.attention = nn.Linear(output_size, 1) if pooling == "attention" else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_size, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs, hidden = self.gru(features)
        if self.pooling == "attention":
            assert self.attention is not None
            attention_weights = torch.softmax(self.attention(outputs).squeeze(-1), dim=1)
            representation = torch.sum(outputs * attention_weights.unsqueeze(-1), dim=1)
        elif self.gru.bidirectional:
            representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            representation = hidden[-1]
        representation = self.output_norm(representation)
        return self.classifier(representation)


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in labels
        },
    }


def _run_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp_enabled: bool = False,
    gradient_clip_norm: float = 1.0,
    grad_scaler: Any | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(features)
                loss = criterion(logits, labels)
            if training:
                assert optimizer is not None and grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        losses.append(float(loss.detach().cpu()))
        targets.append(labels.detach().cpu().numpy())
        predictions.append(logits.detach().argmax(dim=1).cpu().numpy())

    return (
        float(np.mean(losses)),
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def _make_loader(
    data: PreparedTrainingData,
    split: str,
    config: dict,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    training_config = config["training"]
    dataset = IndexedWindowDataset(
        data.matrix,
        data.windows[split],
        int(config["data"]["expected_window_size"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training_config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(training_config["num_workers"]) > 0,
    )


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_history_plot(history: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_macro_f1"], label="train")
    axes[1].plot(history["epoch"], history["validation_macro_f1"], label="validation")
    axes[1].set(title="Macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0.0, 1.0))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _classification_report_frame(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().rename_axis("class_or_average").reset_index()


def train_and_evaluate(
    data: PreparedTrainingData,
    config: dict,
    output_dir: str | Path,
    evaluate_test: bool,
    resume: str = "none",
    evaluate_only: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _resolve_resume_path(output_dir, resume)
    training_config = config["training"]
    model_config = config["model"]
    if data.development_fold is not None and evaluate_test:
        raise ValueError("Test evaluation is forbidden during participant-fold training")
    if evaluate_only and resume == "none":
        raise ValueError("evaluate_only requires a resumable checkpoint")
    labels = [int(value) for value in config["data"]["allowed_labels"]]
    seed = int(training_config["seed"])
    device = resolve_training_device(training_config)
    set_global_seed(seed, include_cuda=device.type == "cuda")
    amp_enabled = bool(training_config["mixed_precision"]) and device.type == "cuda"

    train_loader = _make_loader(data, "train", config, shuffle=True, device=device)
    validation_loader = _make_loader(data, "validation", config, shuffle=False, device=device)
    model = GRUClassifier(
        input_size=len(data.feature_names),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        num_classes=int(model_config["num_classes"]),
        bidirectional=bool(model_config["bidirectional"]),
        pooling=str(model_config.get("pooling", "last")),
        layer_norm=bool(model_config.get("layer_norm", False)),
    ).to(device)

    train_labels = data.windows["train"]["label"].to_numpy(dtype=np.int64)
    counts = np.bincount(train_labels, minlength=len(labels)).astype(np.float32)
    if (counts == 0).any():
        raise ValueError("Every class must have at least one training window")
    weights = counts.sum() / (len(labels) * counts)
    loss_weights = torch.tensor(weights, dtype=torch.float32, device=device) \
        if bool(training_config["class_weighted_loss"]) else None
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("training.label_smoothing must be in [0, 1)")
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_config["scheduler_factor"]),
        patience=int(training_config["scheduler_patience"]),
    )
    try:
        grad_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except (AttributeError, TypeError):
        # Compatibility with older PyTorch images that expose only cuda.amp.
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    signature = _resume_signature(data, config)
    run_config = {
        "device": str(device),
        "torch_version": torch.__version__,
        "selected_feature_count": len(data.feature_names),
        "selected_features": data.feature_names,
        "reaction_margin_frames": data.margin_frames,
        "development_fold": data.development_fold,
        "window_counts": {split: len(frame) for split, frame in data.windows.items()},
        "config": config,
    }
    _save_json(output_dir / "run_config.json", run_config)
    data.standardizer.save(output_dir / "scaler.npz")

    history_rows: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    stopped_early = False
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    resumed_from: str | None = None

    if resume_path is not None:
        checkpoint = _torch_load(resume_path, device)
        _validate_resume_checkpoint(checkpoint, signature)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history_rows = list(checkpoint["history_rows"])
        best_state_dict = checkpoint["best_model_state_dict"]
        stopped_early = bool(checkpoint.get("stopped_early", False))
        start_epoch = int(checkpoint["epoch"]) + 1
        if int(training_config["epochs"]) < int(checkpoint["epoch"]):
            raise ValueError(
                "Configured epochs is smaller than the epoch stored in the checkpoint"
            )
        _restore_rng_state(checkpoint["rng_state"])
        resumed_from = str(resume_path)
        _atomic_torch_save(checkpoint, last_checkpoint_path)
        _atomic_torch_save({
            "checkpoint_version": 1,
            "epoch": best_epoch,
            "validation_macro_f1": best_score,
            "feature_names": data.feature_names,
            "model_config": model_config,
            "model_state_dict": best_state_dict,
        }, best_checkpoint_path)
        _atomic_history_save(history_rows, output_dir / "history.csv")
        print(f"Resumed from epoch {start_epoch - 1}: {resume_path}")
    else:
        print("Starting a new training run")

    max_epochs = int(training_config["epochs"])
    if evaluate_only:
        print(f"Evaluate-only mode: using best checkpoint from epoch {best_epoch}")
    epoch_range = (
        range(start_epoch, max_epochs + 1)
        if not stopped_early and not evaluate_only
        else range(0)
    )
    for epoch in epoch_range:
        train_loss, train_true, train_pred = _run_loader(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            amp_enabled=amp_enabled,
            gradient_clip_norm=float(training_config["gradient_clip_norm"]),
            grad_scaler=grad_scaler,
        )
        validation_loss, validation_true, validation_pred = _run_loader(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            amp_enabled=amp_enabled,
        )
        train_metrics = _metric_summary(train_true, train_pred, labels)
        validation_metrics = _metric_summary(validation_true, validation_pred, labels)
        validation_score = float(validation_metrics["macro_f1"])
        scheduler.step(validation_score)
        history_rows.append({
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "validation_macro_f1": validation_score,
            "validation_accuracy": float(validation_metrics["accuracy"]),
        })
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={validation_loss:.4f} | val_macro_f1={validation_score:.4f}"
        )
        if validation_score > best_score + 1e-6:
            best_score = validation_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            _atomic_torch_save({
                "checkpoint_version": 1,
                "epoch": epoch,
                "validation_macro_f1": validation_score,
                "feature_names": data.feature_names,
                "model_config": model_config,
                "model_state_dict": best_state_dict,
            }, best_checkpoint_path)
        else:
            epochs_without_improvement += 1
        stopped_early = epochs_without_improvement >= int(training_config["early_stopping_patience"])
        if best_state_dict is None:
            raise RuntimeError("No best model state is available after an epoch")
        _atomic_history_save(history_rows, output_dir / "history.csv")
        _atomic_torch_save({
            "checkpoint_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "grad_scaler_state_dict": grad_scaler.state_dict(),
            "best_model_state_dict": best_state_dict,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "stopped_early": stopped_early,
            "history_rows": history_rows,
            "rng_state": _capture_rng_state(include_cuda=device.type == "cuda"),
            "resume_signature": signature,
        }, last_checkpoint_path)
        print(f"Saved resumable checkpoint: {last_checkpoint_path}")
        if stopped_early:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    if not history_rows or best_state_dict is None:
        raise RuntimeError("Training did not produce or restore any completed epoch")
    history = pd.DataFrame(history_rows)
    _save_history_plot(history, output_dir / "training_history.png")
    model.load_state_dict(best_state_dict)

    validation_loss, validation_true, validation_pred = _run_loader(
        model, validation_loader, criterion, device, amp_enabled=amp_enabled
    )
    validation_metrics = _metric_summary(validation_true, validation_pred, labels)
    validation_metrics.update({"loss": validation_loss, "best_epoch": best_epoch})
    _save_json(output_dir / "validation_metrics.json", validation_metrics)

    final_true, final_pred = validation_true, validation_pred
    evaluated_split = "validation"
    test_metrics: dict[str, Any] | None = None
    if evaluate_test:
        # Test is intentionally constructed only after model selection is complete.
        test_loader = _make_loader(data, "test", config, shuffle=False, device=device)
        test_loss, test_true, test_pred = _run_loader(
            model, test_loader, criterion, device, amp_enabled=amp_enabled
        )
        test_metrics = _metric_summary(test_true, test_pred, labels)
        test_metrics.update({"loss": test_loss, "selected_by_validation_epoch": best_epoch})
        _save_json(output_dir / "test_metrics.json", test_metrics)
        prediction_frame = data.windows["test"].copy()
        prediction_frame.insert(0, "window_row", np.arange(len(test_true), dtype=np.int64))
        prediction_frame["true_label"] = test_true
        prediction_frame["predicted_label"] = test_pred
        prediction_frame.to_csv(output_dir / "predictions_test.csv", index=False)
        final_true, final_pred = test_true, test_pred
        evaluated_split = "test"

    _classification_report_frame(final_true, final_pred, labels).to_csv(
        output_dir / "classification_report.csv", index=False
    )
    _save_confusion_matrix(final_true, final_pred, labels, output_dir / "confusion_matrix.png")
    result = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "resumed_from": resumed_from,
        "evaluate_only": evaluate_only,
        "development_fold": data.development_fold,
        "evaluated_split": evaluated_split,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "output_dir": str(output_dir),
    }
    _save_json(output_dir / "result.json", result)
    return result
