import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

import optuna
import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_processed_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    abs_path = to_absolute_path(path)
    
    if abs_path.endswith(".csv"):
        df = pd.read_csv(abs_path)
        # Use our existing logic for hmnist
        if 'label' in df.columns:
            X = df.drop('label', axis=1).values
            y = df['label'].values
        else:
            # Fallback if lab 3 uses 'target'
            if 'target' in df.columns:
                X = df.drop(columns=['target']).values
                y = df['target'].values
            else:
                raise ValueError("CSV should contain 'label' or 'target' column.")
                
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        return X_train, X_test, y_train, y_test
    
    elif abs_path.endswith((".pkl", ".pickle")):
        obj = joblib.load(abs_path)
        if isinstance(obj, dict):
            if {"X_train", "X_test", "y_train", "y_test"}.issubset(obj.keys()):
                return obj["X_train"], obj["X_test"], obj["y_train"], obj["y_test"]
        raise ValueError("Unsupported or unknown pickle format.")
    
    raise ValueError("Supporting .pickle/.pkl or .csv.")

def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_type == "logistic_regression":
        # Filter params for LR
        lr_params = {k: v for k, v in params.items() if k in ['C', 'solver', 'penalty']}
        clf = LogisticRegression(random_state=seed, max_iter=500, **lr_params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Unknown model.type='{model_type}'.")

def evaluate(model: Any, X_train: np.ndarray, y_train: np.ndarray, 
             X_test: np.ndarray, y_test: np.ndarray, metric: str) -> float:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if metric == "f1":
        return float(f1_score(y_test, y_pred, average="binary" if len(np.unique(y_test)) == 2 else "weighted"))
    
    if metric == "roc_auc":
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
            
        if len(np.unique(y_test)) > 2:
            return float(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted"))
        return float(roc_auc_score(y_test, y_score))
        
    raise ValueError(f"Unsupported metric: {metric}")

def evaluate_cv(model: Any, X: np.ndarray, y: np.ndarray, metric: str, seed: int, n_splits: int = 5) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        m = clone(model)
        scores.append(evaluate(m, X_tr, y_tr, X_te, y_te, metric))
    return float(np.mean(scores))

def make_sampler(sampler_name: str, seed: int, grid_space: Dict[str, Any] = None) -> optuna.samplers.BaseSampler:
    sampler_name = sampler_name.lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "grid":
        if not grid_space:
            raise ValueError("For sampler='grid' need to set grid_space.")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError("sampler should be: tpe, random, grid")

def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if model_type == "random_forest":
        space = cfg.hpo.random_forest
        return {
            "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
            "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low, space.min_samples_split.high),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", space.min_samples_leaf.low, space.min_samples_leaf.high),
        }
    if model_type == "logistic_regression":
        space = cfg.hpo.logistic_regression
        solver = trial.suggest_categorical("solver", list(space.solver))
        return {
            "C": trial.suggest_float("C", space.C.low, space.C.high, log=True),
            "solver": solver,
        }
    raise ValueError(f"Unknown model.type='{model_type}'.")

def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)
        
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            mlflow.log_params(params)
            
            model = build_model(cfg.model.type, params=params, seed=cfg.seed)
            
            if cfg.hpo.use_cv:
                X = np.concatenate([X_train, X_test], axis=0)
                y = np.concatenate([y_train, y_test], axis=0)
                score = evaluate_cv(model, X, y, metric=cfg.hpo.metric, seed=cfg.seed, n_splits=cfg.hpo.cv_folds)
            else:
                score = evaluate(model, X_train, y_train, X_test, y_test, metric=cfg.hpo.metric)
                
            mlflow.log_metric(cfg.hpo.metric, score)
            return score
    return objective

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    X_train, X_test, y_train, y_test = load_processed_data(cfg.data.processed_path)
    
    grid_space = None
    if cfg.hpo.sampler.lower() == "grid":
        if cfg.model.type == "random_forest":
            grid_space = {
                "n_estimators": list(cfg.hpo.grid.random_forest.n_estimators),
                "max_depth": list(cfg.hpo.grid.random_forest.max_depth),
            }
        elif cfg.model.type == "logistic_regression":
            grid_space = {
                "C": list(cfg.hpo.grid.logistic_regression.C),
            }
            
    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed, grid_space=grid_space)
    
    with mlflow.start_run(run_name="hpo_parent") as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", cfg.seed)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")
        
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)
        
        # --- Study statistics ---
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        mlflow.log_metric("trials_completed", completed)
        mlflow.log_metric("trials_pruned", pruned)
        mlflow.log_metric("trials_failed", failed)
        
        best_trial = study.best_trial
        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(best_trial.params, "best_params.json")
        
        print(f"\nStudy statistics: {completed} completed, {pruned} pruned, {failed} failed")
        print(f"Best trial: {best_trial.number}")
        print(f"  Value: {best_trial.value}")
        print(f"  Params: {best_trial.params}")
        
        # Retrain and log the best model
        best_model = build_model(cfg.model.type, params=best_trial.params, seed=cfg.seed)
        best_score = evaluate(best_model, X_train, y_train, X_test, y_test, metric=cfg.hpo.metric)
        mlflow.log_metric(f"final_{cfg.hpo.metric}", best_score)
        
        models_dir = to_absolute_path("models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "best_model.pkl")
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        
        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")
        
        # --- Model Registry ---
        if cfg.mlflow.register_model:
            model_uri = f"runs:/{parent_run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, cfg.mlflow.model_name)
            print(f"Model registered: {cfg.mlflow.model_name}, version {mv.version}")
            
        print(f"Final {cfg.hpo.metric} with best params: {best_score}")

if __name__ == "__main__":
    main()
