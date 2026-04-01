# Optuna-based hyperparameter tuning for the forecasting model using XGBoost and LightGBM.

import numpy as np
import pandas as pd
import optuna
from typing import Dict, Tuple, Optional, Callable
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
from sklearn.metrics import f1_score

from src.models.evaluation import compute_metrics
from src.config import RANDOM_STATE
from src.utils.logging_config import setup_logging

logger = setup_logging('models.tuner')

optuna.logging.set_verbosity(optuna.logging.WARNING)

def xgboost_regression_objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-0, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - preds) ** 2))
    return rmse


def xgboost_classification_objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scale_pos_weight: float = 1.0) -> float:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, zero_division=0)
    return f1


def lightgbm_regression_objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True)
    }

    model = lgb.LGBMRegressor(**params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    preds = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - preds) ** 2))
    return rmse


def lightgbm_classification_objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scale_pos_weight: float = 1.0) -> float:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
        "is_unbalance": False,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    }

    model = lgb.LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, zero_division=0)
    return f1


def run_optuna_study(objective_fn: Callable, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 50, direction: str = 'minimize', study_name: str = 'study', **kwargs) -> Tuple[optuna.Study, Dict]:
    logger.info(f"Starting optuna study: {study_name} ({n_trials} trials, direction={direction})")
    study = optuna.create_study(study_name = study_name, direction = direction, sampler = optuna.samplers.TPESampler(seed = RANDOM_STATE), pruner = optuna.pruners.MedianPruner(n_warmup_steps = 5))

    def objective(trial):
        return objective_fn(trial, X_train, y_train, X_val, y_val, **kwargs)
    
    study.optimize(objective, n_trials = n_trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"Best trial: {best.value} with params: {best.params}")
    logger.info(f"Best params: {best.params}")

    return study, best.params
