## Handles training XGBoost/LightGBM with optional optuna tuning, MLFlow logging, and model serialization.

from asyncio.log import logger

import numpy as np
import pandas as pd
import json
import joblib
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
import optuna
import mlflow
import mlflow.xgboost
import mlflow.lightgbm

from src.config import (TARGETS, MODELS_DIR, MLFLOW_TRACKING_URI, RANDOM_STATE, load_feature_registry)
from src.models.evaluation import compute_metrics, log_metrics
from src.utils.logging_config import setup_logging

def setup_mlflow(experiment_name: str) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(f"MLFlow experiment set to: {experiment_name} (ID: {experiment.experiment_id})")
    return experiment.experiment_id


def train_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, task_type: str = 'regression', params: Optional[Dict] = None, scale_pos_weight: float = 1.0, experiment_name: str = 'default', run_name: str = 'xgboost', feature_names: Optional[List[str]] = None, log_to_mlflow: bool = True) -> Tuple[object, Dict[str, float], Dict[str, float]]:
    """Train an XGBoost model with MLFlow logging."""
    logger.info(f"Starting XGBoost training for task: {task_type}")

    if params is None:
        if task_type == 'regression':
            params = {
               "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": RANDOM_STATE,
                "verbosity": 0, 
            }
        else:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": RANDOM_STATE,
                "verbosity": 0,
            }
    
    params['random_state'] = RANDOM_STATE
    params['verbosity'] = 0

    if task_type == 'regression':
        if 'objective' not in params:
            params['objective'] = 'reg:squarederror'
        model = xgb.XGBRegressor(**params)
    else:
        if 'objective' not in params:
            params['objective'] = 'binary:logistic'
        if 'scale_pos_weight' not in params:
            params['scale_pos_weight'] = scale_pos_weight
        model = xgb.XGBClassifier(**params)
    

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    # Prediction
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    # Probabilities for classification
    train_probs = None
    val_probs = None
    if task_type == 'classification':
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]

    # Metrics
    train_metrics = compute_metrics(y_train, train_preds, task_type, train_probs)
    val_metrics = compute_metrics(y_val, val_preds, task_type, val_probs)

    log_metrics(run_name, train_metrics, 'train')
    log_metrics(run_name, val_metrics, 'val')

    # MLFlow logging
    if log_to_mlflow:
        setup_mlflow(experiment_name)
        with mlflow.start_run(run_name=run_name):
            # Log params
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("task_type", task_type)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_features", X_train.shape[1])

            # Log metrics
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"train_{k}", v)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"val_{k}", v)

            # Log feature importance
            if feature_names:
                importance = model.feature_importances_
                fi_df = pd.DataFrame({
                    "feature": feature_names[:len(importance)],
                    "importance": importance
                }).sort_values("importance", ascending=False)
                fi_path = MODELS_DIR / "xgboost" / f"{run_name}_feature_importance.csv"
                fi_df.to_csv(fi_path, index=False)
                mlflow.log_artifact(str(fi_path))

    
    model_path = MODELS_DIR / "xgboost" / f"{run_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")

    return model, val_metrics, train_metrics


def train_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, task_type: str = 'regression', params: Optional[Dict] = None, scale_pos_weight: float = 1.0, experiment_name: str = "default", run_name: str = "lightgbm", feature_names: Optional[List[str]] = None, log_to_mlflow: bool = True) -> Tuple[object, Dict[str, float], Dict[str, float]]:
    """Train a LightGBM model with MLFlow logging."""
    logger.info(f"Training LightGBM model for task: {task_type}")

    if params is None:
        if task_type == 'regression':
            params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": RANDOM_STATE,
                "verbosity": -1
            }
        else:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": RANDOM_STATE,
                "verbosity": -1
            }
    
    params['random_state'] = RANDOM_STATE
    params['verbosity'] = -1

    if task_type == 'regression':
        if 'objective' not in params:
            params['objective'] = 'regression'
        model = lgb.LGBMRegressor(**params)
    else:
        if 'objective' not in params:
            params['objective'] = 'binary'
        if 'scale_pos_weight' not in params:
            params['scale_pos_weight'] = scale_pos_weight
        model = lgb.LGBMClassifier(**params)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_probs = None
    val_probs = None
    if task_type == 'classification':
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]

    train_metrics = compute_metrics(y_train, train_preds, task_type, train_probs)
    val_metrics = compute_metrics(y_val, val_preds, task_type, val_probs)

    if log_to_mlflow:
        setup_mlflow(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("task_type", task_type)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_features", X_train.shape[1])

            for k, v in train_metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"train_{k}", v)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"val_{k}", v)

            mlflow.lightgbm.log_model(model, 'model')

            if feature_names:
                importance = model.feature_importances_
                fi_df = pd.DataFrame({
                    "feature": feature_names[:len(importance)],
                    "importance": importance
                }).sort_values("importance", ascending=False)
                fi_path = MODELS_DIR / "lightgbm" / f"{run_name}_feature_importance.csv"
                fi_df.to_csv(fi_path, index=False)
                mlflow.log_artifact(str(fi_path))
    
    model_path = MODELS_DIR / "lightgbm" / f"{run_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")

    return model, val_metrics, train_metrics


def train_tuned_model(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, task_type: str, model_type: str, best_params: Dict, scale_pos_weight: float = 1.0, experiment_name: str = 'default', run_name: str = 'tuned', feature_names: Optional[List[str]] = None, log_to_mlflow: bool = True) -> Tuple[object, Dict[str, float], Dict[str, float]]:
    """Train model with best hyperparameters from tuning."""
    if model_type == 'xgboost':
        params = {**best_params}
        if task_type == 'regression':
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        else:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
            params['scale_pos_weight'] = scale_pos_weight
        params['tree_method'] = 'hist'
        params['random_state'] = RANDOM_STATE
        params['verbosity'] = 0

        return train_xgboost(X_train, y_train, X_val, y_val, task_type = task_type, params = params, experiment_name = experiment_name, run_name = run_name, feature_names = feature_names, log_to_mlflow = log_to_mlflow)
    elif model_type == 'lightgbm':
        params = {**best_params}
        if task_type == 'regression':
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        else:
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            params['scale_pos_weight'] = scale_pos_weight
        params['random_state'] = RANDOM_STATE
        params['verbosity'] = -1

        return train_lightgbm(X_train, y_train, X_val, y_val, task_type = task_type, params = params, experiment_name = experiment_name, run_name = run_name, feature_names = feature_names, log_to_mlflow = log_to_mlflow)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

def save_best_model(model: object, target_key: str, model_type: str, val_metrics: Dict, feature_names: List[str], params: Dict) -> Path:
    target_cfg = TARGETS[target_key]
    save_dir = MODELS_DIR / "best" / target_key / model_type
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / 'model.joblib'
    joblib.dump(model, model_path)

    metadata = {
        "target_key": target_key,
        "target_name": target_cfg.name,
        "target_column": target_cfg.target_column,
        "task_type": target_cfg.task_type,
        "model_type": model_type,
        "val_metrics": val_metrics,
        "params": {k: v for k, v in params.items()
                   if isinstance(v, (int, float, str, bool))},
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "saved_at": datetime.now().isoformat(),
    }
    meta_path = save_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    feat_path = save_dir / 'features.json'
    with open(feat_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Best model saved: {model_path}")
    return save_dir


