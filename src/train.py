import sys
import os
import argparse
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

EXPERIMENT_NAME = "HAM10000_Skin_Lesion_Classification"


def parse_args():
    """Parse CLI arguments for hyperparameters."""
    parser = argparse.ArgumentParser(
        description='Train GradientBoostingClassifier on HAM10000 with MLflow logging'
    )
    parser.add_argument('--n_estimators', type=int, default=200,
                        help='Number of boosting stages (default: 200)')
    parser.add_argument('--max_depth', type=int, default=чі5,
                        help='Maximum depth of individual trees (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate / shrinkage (default: 0.1)')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Fraction of samples for fitting trees (default: 0.8)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--author', type=str, default='Yurii Polulikh',
                        help='Author name for MLflow tag')
    parser.add_argument('--dataset_version', type=str, default='v1.0',
                        help='Dataset version for MLflow tag')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional MLflow run name')
    parser.add_argument('prepared_data_dir', type=str, nargs='?', default='data/prepared',
                        help='Directory with prepared data (default: data/prepared)')
    parser.add_argument('models_dir', type=str, nargs='?', default='models',
                        help='Directory to save models (default: models)')
    return parser.parse_args()




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
    print(f"   Confusion Matrix saved: {save_path}")


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
    print(f"   Feature Importance saved: {save_path}")


def train(args=None):
    """Main training function with MLflow logging."""

    if args is None:
        args = parse_args()

    prepared_data_dir = args.prepared_data_dir
    models_dir = args.models_dir
    
    os.makedirs(models_dir, exist_ok=True)

    print("Training model with MLflow")
    print("=" * 60)

    # 1. Load prepared data
    print(f"\n[1/5] Loading prepared data from: {prepared_data_dir}")

    train_path = os.path.join(prepared_data_dir, "train.csv")
    test_path = os.path.join(prepared_data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Prepared data not found in {prepared_data_dir}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    # Determine number of classes (for logging)
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(unique_classes)

    # 4. MLflow init
    print("\n[4/5] Initializing MLflow")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"   Experiment: {EXPERIMENT_NAME}")

    # 5. Train and log
    print("\n[5/5] Training and logging")

    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'random_state': args.random_state,
    }

    run_name = args.run_name or f"GBC_depth{args.max_depth}_est{args.n_estimators}_lr{args.learning_rate}"

    with mlflow.start_run(run_name=run_name):

        # MLflow Tags (metadata)
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("model_type", "GradientBoostingClassifier")
        mlflow.set_tag("dataset", "HAM10000")
        mlflow.set_tag("task", "skin_lesion_classification")
        print("   Tags set (author, dataset_version, model_type, dataset, task)")

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("subsample", args.subsample)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", n_classes)
        print("   Hyperparameters logged")


        print(f"\n   Training GradientBoostingClassifier...")
        print(f"      n_estimators={args.n_estimators}, "
              f"max_depth={args.max_depth}, "
              f"learning_rate={args.learning_rate}")

        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train, y_train)
        print("   Model trained!")

        # Predict on BOTH train and test sets
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Train metrics 
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1_weighted", train_f1)

        # Test metrics
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
        test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_weighted", test_f1_weighted)
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        mlflow.log_metric("test_precision_weighted", test_precision)
        mlflow.log_metric("test_recall_weighted", test_recall)

        # --- Overfitting gap ---
        overfit_gap = train_accuracy - test_accuracy
        mlflow.log_metric("overfit_gap_accuracy", overfit_gap)

        print(f"\n   Train metrics:")
        print(f"      Accuracy: {train_accuracy:.4f}   F1: {train_f1:.4f}")
        print(f"   Test metrics:")
        print(f"      Accuracy: {test_accuracy:.4f}   F1: {test_f1_weighted:.4f}")
        print(f"      Precision: {test_precision:.4f}  Recall: {test_recall:.4f}")
        print(f"   Overfit gap (train-test accuracy): {overfit_gap:.4f}")

        # --- Log model ---
        mlflow.sklearn.log_model(model, "gradient_boosting_model")
        print("   Model logged")

        # --- Log artifacts ---
        class_labels = [LABEL_NAMES.get(int(c), str(c)) for c in sorted(unique_classes)]


        cm_path = "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred_test, class_labels, cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = "feature_importance.png"
        plot_feature_importance(model, top_n=30, save_path=fi_path)
        mlflow.log_artifact(fi_path)

        report = classification_report(y_test, y_pred_test, target_names=class_labels, digits=4)
        report_path = "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)
        print("   Artifacts logged (confusion_matrix, feature_importance, classification_report)")

        run_id = mlflow.active_run().info.run_id
        print(f"\n   MLflow Run ID: {run_id}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nTo view results in MLflow UI, run:")
    print(f"  mlflow ui")
    print(f"  Open: http://127.0.0.1:5000")
    print(f"\nTo filter by tags in MLflow Search:")
    print(f'  tags.model_type = "GradientBoostingClassifier"')
    print(f'  tags.author = "{args.author}"')

    for f in [cm_path, fi_path, report_path]:
        if os.path.exists(f):
            os.remove(f)

    return run_id


if __name__ == '__main__':
    train()
