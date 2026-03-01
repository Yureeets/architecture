import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """
    Preprocessing of features (pixel values):
    1. Filling missing values with the median.
    2. Normalization of pixel values to the range [0, 1].

    Parameters:
        X: DataFrame with pixel features (0–255)

    Returns:
        np.ndarray with normalized features
    """
    X_processed = X.copy()

    missing_count = X_processed.isnull().sum().sum()
    if missing_count > 0:
        print(f"  Found {missing_count} missing values. Filling with median...")
        X_processed = X_processed.fillna(X_processed.median())
    else:
        print("  No missing values found")

    X_normalized = X_processed.values / 255.0

    print(f"  Features normalized: min={X_normalized.min():.3f}, max={X_normalized.max():.3f}")
    return X_normalized


def encode_labels(y: pd.Series) -> tuple:
    """
    Encoding of the target variable using LabelEncoder.

    Parameters:
        y: Series with numerical class labels

    Returns:
        y_encoded (np.ndarray) — закодовані мітки
        label_encoder (LabelEncoder) — об'єкт для зворотного перетворення
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"  Labels encoded: {len(le.classes_)} classes — {list(le.classes_)}")
    return y_encoded, le


def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Splitting the sample into training and test sets
    using train_test_split with stratification.

    Parameters:
        X: feature matrix
        y: label vector
        test_size: test set fraction (default 0.2)
        random_state: seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Stratification to preserve class proportions
    )

    print(f"Sample split:")
    print(f"   Training: {X_train.shape[0]} samples ({(1 - test_size) * 100:.0f}%)")
    print(f"   Test:     {X_test.shape[0]} samples ({test_size * 100:.0f}%)")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Testing the module
    from data_loader import load_hmnist_data

    X, y = load_hmnist_data()

    X_processed = preprocess_features(X)
    y_encoded, le = encode_labels(y)
    X_train, X_test, y_train, y_test = split_data(X_processed, y_encoded)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
