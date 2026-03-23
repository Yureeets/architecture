"""
Hyperparameter Tuning Analysis: max_depth study.

Runs 5+ experiments varying max_depth from underfitting to overfitting.
Logs both train and test metrics to MLflow for overfitting analysis.

Usage:
    python src/run_experiments.py
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train


# max_depth values: from underfitting (shallow) to overfitting (deep)
MAX_DEPTH_VALUES = [1, 2, 3, 5, 7, 10, 15]


def run_experiments():
    """Run multiple experiments varying max_depth."""

    print("=" * 60)
    print("HYPERPARAMETER TUNING: max_depth study")
    print(f"Values to test: {MAX_DEPTH_VALUES}")
    print(f"Total runs: {len(MAX_DEPTH_VALUES)}")
    print("=" * 60)

    results = []

    for i, depth in enumerate(MAX_DEPTH_VALUES, 1):
        print(f"\n{'='*60}")
        print(f"RUN {i}/{len(MAX_DEPTH_VALUES)} — max_depth={depth}")
        print(f"{'='*60}")

        args = argparse.Namespace(
            n_estimators=200,
            max_depth=depth,
            learning_rate=0.1,
            subsample=0.8,
            test_size=0.2,
            random_state=42,
            author="Yurii Polulikh",
            dataset_version="v1.0",
            run_name=f"tuning_max_depth_{depth}",
        )

        run_id = train(args)
        results.append((depth, run_id))
        print(f"\nRun {i} completed — max_depth={depth}, run_id={run_id}\n")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 60)
    print(f"\nResults summary:")
    print(f"{'max_depth':>10} | {'Run ID'}")
    print(f"{'-'*10}-+-{'-'*32}")
    for depth, run_id in results:
        print(f"{depth:>10} | {run_id}")

    print(f"\nTotal runs: {len(results)}")
    print(f"\nTo compare runs in MLflow UI:")
    print(f"  1. Run: mlflow ui")
    print(f"  2. Open: http://127.0.0.1:5000")
    print(f"  3. Select all runs -> Click 'Compare'")
    print(f"  4. Plot: X-axis = max_depth, Y-axis = test_accuracy or test_f1_weighted")
    print(f"\nTo filter by tag in MLflow Search:")
    print(f'  tags.model_type = "GradientBoostingClassifier"')


if __name__ == '__main__':
    run_experiments()
