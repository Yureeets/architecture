"""Shared pytest fixtures."""
import os

import pytest


@pytest.fixture
def data_path():
    """Path to the raw dataset."""
    return os.getenv("DATA_PATH", "data/raw/hmnist_28_28_L.csv")


@pytest.fixture
def metrics_path():
    """Path to saved metrics."""
    return "metrics.json"


@pytest.fixture
def model_path():
    """Path to saved model."""
    return "models/model.pkl"


@pytest.fixture
def cm_path():
    """Path to confusion matrix image."""
    return "confusion_matrix.png"
