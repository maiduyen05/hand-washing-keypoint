import json

import pytest

from summarize_cv import summarize_cv


def test_cv_summary_requires_five_unique_validation_only_folds(tmp_path) -> None:
    directories = []
    for fold in range(5):
        directory = tmp_path / f"fold_{fold}"
        directory.mkdir()
        result = {
            "best_epoch": fold + 1,
            "development_fold": fold,
            "evaluated_split": "validation",
            "test_metrics": None,
            "validation_metrics": {
                "accuracy": 0.70 + fold * 0.01,
                "balanced_accuracy": 0.69 + fold * 0.01,
                "macro_f1": 0.68 + fold * 0.01,
                "weighted_f1": 0.70 + fold * 0.01,
                "per_class": {"0": {"f1": 0.60 + fold * 0.01}},
            },
        }
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        directories.append(directory)

    summary, frame = summarize_cv(directories)
    assert summary["fold_count"] == 5
    assert summary["mean"]["macro_f1"] == pytest.approx(0.70)
    assert summary["sample_std"]["macro_f1"] > 0
    assert frame["fold"].tolist() == list(range(5))
