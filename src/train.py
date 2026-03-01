import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_hmnist_data, LABEL_NAMES
from preprocessing import preprocess_features, encode_labels, split_data

warnings.filterwarnings('ignore')


# Hyperparameters
PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'random_state': 42,
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPERIMENT_NAME = "HAM10000_Skin_Lesion_Classification"


def plot_confusion_matrix(y_true, y_pred, labels, save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix saved: {save_path}")


def plot_feature_importance(model, top_n: int = 30, save_path: str = 'feature_importance.png'):
    """Plot and save feature importance (top-N)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'pixel{i}' for i in indices], fontsize=8)
    ax.set_title(f'Top-{top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature Importance saved: {save_path}")


def train():
    """Main function for training with MLflow logging."""

    print("=" * 60)
    print("STEP 4: Training model with MLflow")
    print("=" * 60)

    print("\nLoading data")
    X, y = load_hmnist_data()

    print("\nPreprocessing data")
    X_processed = preprocess_features(X)
    y_encoded, label_encoder = encode_labels(y)

    print("\nSplitting data")
    X_train, X_test, y_train, y_test = split_data(
        X_processed, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nInitializing MLflow")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("\nTraining model and logging to MLflow")

    with mlflow.start_run():

        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("n_estimators", PARAMS['n_estimators'])
        mlflow.log_param("max_depth", PARAMS['max_depth'])
        mlflow.log_param("learning_rate", PARAMS['learning_rate'])
        mlflow.log_param("subsample", PARAMS['subsample'])
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(label_encoder.classes_))
        print("   Hyperparameters logged")

        print(f"\n   Training GradientBoostingClassifier...")
        print(f"      n_estimators={PARAMS['n_estimators']}, "
              f"max_depth={PARAMS['max_depth']}, "
              f"learning_rate={PARAMS['learning_rate']}")

        model = GradientBoostingClassifier(**PARAMS)
        model.fit(X_train, y_train)
        print("   Model trained!")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"\n   Metrics on test set:")
        print(f"      Accuracy:           {accuracy:.4f}")
        print(f"      F1 Score (weighted): {f1_weighted:.4f}")
        print(f"      F1 Score (macro):    {f1_macro:.4f}")
        print(f"      Precision:           {precision:.4f}")
        print(f"      Recall:              {recall:.4f}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        print("   Metrics logged")

        mlflow.sklearn.log_model(model, "gradient_boosting_model")
        print("   Model logged")

        class_labels = [LABEL_NAMES.get(c, str(c)) for c in sorted(label_encoder.classes_)]

        cm_path = "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, class_labels, cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = "feature_importance.png"
        plot_feature_importance(model, top_n=30, save_path=fi_path)
        mlflow.log_artifact(fi_path)
        report = classification_report(
            y_test, y_pred,
            target_names=class_labels,
            digits=4
        )
        report_path = "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)
        print("   Artifacts logged (confusion_matrix, feature_importance, classification_report)")

        print(f"\n   MLflow Run ID: {mlflow.active_run().info.run_id}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nTo view results in MLflow UI, run:")
    print(f"  mlflow ui")
    print(f"  Open: http://127.0.0.1:5000")
    print(f"\nExperiment: {EXPERIMENT_NAME}")

    for f in [cm_path, fi_path, report_path]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    train()
