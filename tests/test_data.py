"""
Pre-train tests: data validation.
These tests run BEFORE training to verify the input data is valid.
"""
import os

import numpy as np
import pandas as pd
import pytest


class TestDataExists:
    """Check that the dataset file is present."""

    def test_data_file_exists(self, data_path):
        assert os.path.exists(data_path), (
            f"Dataset not found: {data_path}"
        )

    def test_data_file_not_empty(self, data_path):
        if os.path.exists(data_path):
            assert os.path.getsize(data_path) > 0, "Dataset file is empty"


class TestDataSchema:
    """Check data schema and quality."""

    @pytest.fixture(autouse=True)
    def load_data(self, data_path):
        self.df = pd.read_csv(data_path)

    def test_label_column_exists(self):
        assert 'label' in self.df.columns, (
            "CSV must contain a 'label' column"
        )

    def test_no_nan_labels(self):
        assert self.df['label'].notna().all(), (
            "Labels contain NaN values"
        )

    def test_minimum_rows(self):
        min_rows = 100
        assert len(self.df) >= min_rows, (
            f"Dataset has {len(self.df)} rows, need >= {min_rows}"
        )

    def test_feature_count(self):
        n_features = len(self.df.columns) - 1  # minus label
        assert n_features > 0, "No feature columns found"

    def test_pixel_value_range(self):
        features = self.df.drop('label', axis=1)
        assert features.min().min() >= 0, "Pixel values below 0"
        assert features.max().max() <= 255, "Pixel values above 255"

    def test_label_classes(self):
        unique_labels = self.df['label'].nunique()
        assert unique_labels >= 2, (
            f"Only {unique_labels} class(es) found, need >= 2"
        )
