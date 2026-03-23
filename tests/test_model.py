"""
Post-train tests: Quality Gate.
These tests run AFTER training to verify the model artifacts
and that metrics meet minimum thresholds.
"""
import json
import os

import pytest


class TestArtifactsExist:
    """Check that training produced all required artifacts."""

    def test_metrics_json_exists(self, metrics_path):
        assert os.path.exists(metrics_path), (
            f"metrics.json not found at {metrics_path}"
        )

    def test_model_pkl_exists(self, model_path):
        assert os.path.exists(model_path), (
            f"Model file not found at {model_path}"
        )

    def test_confusion_matrix_exists(self, cm_path):
        assert os.path.exists(cm_path), (
            f"Confusion matrix not found at {cm_path}"
        )


class TestMetricsFormat:
    """Check that metrics.json has the correct format."""

    @pytest.fixture(autouse=True)
    def load_metrics(self, metrics_path):
        with open(metrics_path, "r") as f:
            self.metrics = json.load(f)

    def test_has_accuracy(self):
        assert "accuracy" in self.metrics, "Missing 'accuracy' key"

    def test_has_f1(self):
        assert "f1" in self.metrics, "Missing 'f1' key"

    def test_accuracy_in_range(self):
        acc = self.metrics["accuracy"]
        assert 0.0 <= acc <= 1.0, (
            f"Accuracy {acc} out of [0, 1] range"
        )

    def test_f1_in_range(self):
        f1 = self.metrics["f1"]
        assert 0.0 <= f1 <= 1.0, (
            f"F1 {f1} out of [0, 1] range"
        )


class TestQualityGate:
    """Quality Gate: model must meet minimum performance thresholds."""

    @pytest.fixture(autouse=True)
    def load_metrics(self, metrics_path):
        with open(metrics_path, "r") as f:
            self.metrics = json.load(f)

    def test_quality_gate_f1(self):
        threshold = float(os.getenv("F1_THRESHOLD", "0.50"))
        f1 = self.metrics["f1"]
        assert f1 >= threshold, (
            f"QUALITY GATE FAILED: F1={f1:.4f} < threshold={threshold}"
        )

    def test_quality_gate_accuracy(self):
        threshold = float(os.getenv("ACC_THRESHOLD", "0.50"))
        acc = self.metrics["accuracy"]
        assert acc >= threshold, (
            f"QUALITY GATE FAILED: Accuracy={acc:.4f} "
            f"< threshold={threshold}"
        )
