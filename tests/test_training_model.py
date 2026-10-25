"""GRU smoke tests."""

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from src.training import (
    GRUClassifier,
    IndexedWindowDataset,
    _atomic_torch_save,
    _capture_rng_state,
    _resolve_resume_path,
    _resume_signature,
    _restore_rng_state,
    _torch_load,
    _validate_resume_checkpoint,
    cuda_architecture_is_supported,
    train_and_evaluate,
)
from src.training_data import FeatureStandardizer, PreparedTrainingData


def test_gru_forward_backward_and_checkpoint_round_trip(tmp_path) -> None:
    model = GRUClassifier(
        input_size=10,
        hidden_size=8,
        num_layers=2,
        dropout=0.1,
        num_classes=7,
    )
    features = torch.randn(4, 48, 10)
    labels = torch.tensor([0, 1, 2, 3])
    logits = model(features)
    assert logits.shape == (4, 7)
    torch.nn.functional.cross_entropy(logits, labels).backward()

    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    clone = GRUClassifier(10, 8, 2, 0.1, 7)
    clone.load_state_dict(torch.load(checkpoint, weights_only=True))


def test_causal_attention_gru_returns_class_logits() -> None:
    model = GRUClassifier(
        input_size=10,
        hidden_size=8,
        num_layers=2,
        dropout=0.4,
        num_classes=7,
        bidirectional=False,
        pooling="attention",
        layer_norm=True,
    )
    features = torch.randn(4, 48, 10)
    logits = model(features)
    assert logits.shape == (4, 7)
    assert model.gru.bidirectional is False
    logits.sum().backward()
    assert model.attention is not None
    assert model.attention.weight.grad is not None


def test_indexed_dataset_reads_only_requested_window() -> None:
    matrix = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    windows = pd.DataFrame({
        "start_row": [2, 10],
        "end_row_exclusive": [6, 14],
        "label": [1, 5],
    })
    dataset = IndexedWindowDataset(matrix, windows, window_size=4)
    features, label = dataset[1]
    assert tuple(features.shape) == (4, 3)
    assert int(label) == 5
    assert np.array_equal(features.numpy(), matrix[10:14])


def test_atomic_checkpoint_and_resume_resolution(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    assert _resolve_resume_path(output, "none") is None
    assert _resolve_resume_path(output, "auto") is None
    checkpoint_path = output / "last_checkpoint.pt"
    _atomic_torch_save({"value": 3}, checkpoint_path)
    assert checkpoint_path.exists()
    assert not (output / "last_checkpoint.pt.tmp").exists()
    assert _torch_load(checkpoint_path, torch.device("cpu"))["value"] == 3
    assert _resolve_resume_path(output, "auto") == checkpoint_path
    with pytest.raises(FileExistsError):
        _resolve_resume_path(output, "none")


def test_corrupt_and_incompatible_checkpoints_fail_loudly(tmp_path) -> None:
    corrupt = tmp_path / "last_checkpoint.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="Cannot read checkpoint"):
        _torch_load(corrupt, torch.device("cpu"))
    with pytest.raises(ValueError, match="incompatible"):
        _validate_resume_checkpoint({"checkpoint_version": 1, "resume_signature": {"x": 1}}, {"x": 2})


def test_rng_state_round_trip() -> None:
    import random

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = _capture_rng_state(include_cuda=False)
    expected = (random.random(), np.random.random(), torch.rand(1))
    _restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    supported = ["sm_70", "sm_75", "sm_80"]
    assert not cuda_architecture_is_supported((6, 0), supported)
    assert cuda_architecture_is_supported((7, 5), supported)
    if torch.cuda.is_available():
        cuda_state = _capture_rng_state(include_cuda=True)
        cuda_state["torch"] = cuda_state["torch"].to("cuda")
        cuda_state["cuda"] = [value.to("cuda") for value in cuda_state["cuda"]]
        _restore_rng_state(cuda_state)


def _tiny_training_data() -> PreparedTrainingData:
    matrix = np.linspace(-1.0, 1.0, 36 * 2, dtype=np.float32).reshape(36, 2)
    windows = {}
    for split, offset in (("train", 0), ("validation", 12), ("test", 24)):
        starts = np.arange(offset, offset + 12, 2, dtype=np.int64)
        windows[split] = pd.DataFrame({
            "video_name": [f"{split}.mp4"] * 6,
            "participant_id": [1 if split == "train" else 2 if split == "validation" else 3] * 6,
            "split": [split] * 6,
            "fold": [0] * 6,
            "segment_id": [0] * 6,
            "start_row": starts,
            "end_row_exclusive": starts + 2,
            "start_frame": np.arange(0, 12, 2),
            "end_frame": np.arange(1, 13, 2),
            "label": [0, 1, 0, 1, 0, 1],
        })
    standardizer = FeatureStandardizer(
        feature_names=["left_present", "right_present"],
        means=np.zeros(2, dtype=np.float32),
        scales=np.ones(2, dtype=np.float32),
        scaled=np.zeros(2, dtype=bool),
    )
    return PreparedTrainingData(matrix, standardizer.feature_names, windows, standardizer, 0)


def _tiny_config(epochs: int) -> dict:
    return {
        "data": {
            "expected_window_size": 2,
            "allowed_labels": [0, 1],
        },
        "model": {
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "num_classes": 2,
            "bidirectional": False,
        },
        "training": {
            "device": "cpu",
            "seed": 42,
            "batch_size": 2,
            "epochs": epochs,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "early_stopping_patience": 10,
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "mixed_precision": False,
            "class_weighted_loss": True,
        },
    }


def test_interrupted_then_resumed_matches_uninterrupted_training(tmp_path) -> None:
    data = _tiny_training_data()
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    train_and_evaluate(data, _tiny_config(3), uninterrupted, evaluate_test=False, resume="none")
    train_and_evaluate(data, _tiny_config(1), resumed, evaluate_test=False, resume="none")
    train_and_evaluate(data, _tiny_config(3), resumed, evaluate_test=False, resume="auto")

    full = _torch_load(uninterrupted / "last_checkpoint.pt", torch.device("cpu"))
    continued = _torch_load(resumed / "last_checkpoint.pt", torch.device("cpu"))
    assert full["epoch"] == continued["epoch"] == 3
    assert full["history_rows"] == continued["history_rows"]
    for name, value in full["model_state_dict"].items():
        assert torch.equal(value, continued["model_state_dict"][name])


def test_evaluate_only_does_not_run_another_epoch(tmp_path) -> None:
    data = _tiny_training_data()
    output = tmp_path / "evaluate_only"
    config = _tiny_config(1)
    config["training"]["label_smoothing"] = 0.05
    train_and_evaluate(data, config, output, evaluate_test=False, resume="none")
    result = train_and_evaluate(
        data,
        config,
        output,
        evaluate_test=False,
        resume="auto",
        evaluate_only=True,
    )
    checkpoint = _torch_load(output / "last_checkpoint.pt", torch.device("cpu"))
    assert checkpoint["epoch"] == 1
    assert len(checkpoint["history_rows"]) == 1
    assert result["evaluate_only"] is True


def test_fold_signature_preserves_legacy_non_fold_compatibility() -> None:
    data = _tiny_training_data()
    config = _tiny_config(1)
    assert "development_fold" not in _resume_signature(data, config)
    data.development_fold = 3
    assert _resume_signature(data, config)["development_fold"] == 3
