"""
Train pipeline script for CI/CD.
Trains a model and saves artifacts: metrics.json, confusion_matrix.png, models/model.pkl
"""
import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

LABEL_NAMES = {
    0: 'akiec', 1: 'bcc', 2: 'bkl',
    3: 'df', 4: 'nv', 5: 'vasc', 6: 'mel',
}

SEED = 42


def load_data(data_path: str):
    """Load CSV data and split into features/labels."""
    df = pd.read_csv(data_path)

    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column")

    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y


def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train a RandomForest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average='weighted')),
        "precision": float(precision_score(y_test, y_pred, average='weighted')),
        "recall": float(recall_score(y_test, y_pred, average='weighted')),
        "n_classes": int(len(np.unique(y_test))),
        "test_samples": int(len(y_test)),
    }
    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred, output_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    class_labels = [LABEL_NAMES.get(int(c), str(c)) for c in unique_labels]

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {output_path}")


def main():
    data_path = os.getenv("DATA_PATH", "data/raw/hmnist_28_28_L.csv")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("TRAIN PIPELINE (CI/CD)")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1/4] Loading data from: {data_path}")
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 2. Train model
    print("\n[2/4] Training RandomForest...")
    model = train_model(X_train, y_train, n_estimators=100, max_depth=10)
    print("  Model trained!")

    # 3. Evaluate
    print("\n[3/4] Evaluating...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  Precision:{metrics['precision']:.4f}")
    print(f"  Recall:   {metrics['recall']:.4f}")

    # 4. Save artifacts
    print("\n[4/4] Saving artifacts...")

    # metrics.json
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metrics saved: metrics.json")

    # confusion_matrix.png
    save_confusion_matrix(y_test, y_pred)

    # model.pkl
    model_path = os.path.join(models_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()
