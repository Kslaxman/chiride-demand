"""Model evaluation utilities"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import ( # type: ignore
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, mean_absolute_percentage_error
)
from src.utils.logging_config import setup_logging

logger = setup_logging('models.evaluation')

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """"Compute regression metrics.
    
    Returns: dict with mae, rmse, mape, r2, median_ae
    """
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)

    # Handle Edge Cases
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = np.median(np.abs(y_true - y_pred))

    # MAPE
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = np.nan

    # Symmetric MAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    nonzero_denom = denom != 0
    if nonzero_denom.sum() > 0:
        smape = np.mean(np.abs(y_true[nonzero_denom] - y_pred[nonzero_denom]) / denom[nonzero_denom]) * 100
    else:
        smape = np.nan

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "smape": round(smape, 4),
        "r2": round(r2, 4),
        "median_ae": round(median_ae, 4)
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute classification metrics.
    
    Returns: dict with accuracy, precision, recall, f1, roc_auc"""

    y_true = np.asarray(y_true, dtype = int)
    y_pred = np.asarray(y_pred, dtype = int)

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4)
    }

    if y_prob is not None:
        try:
            metrics['roc_auc'] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics['roc_auc'] = np.nan

    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_positivies'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)

    return metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """"Compute metrics based on task type.
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        task_type: 'regression' or 'classification'
        y_prob: Predicted probabilities for classification (optional)"""
    
    if task_type == "regression":
        return regression_metrics(y_true, y_pred)
    elif task_type == "classification":
        return classification_metrics(y_true, y_pred, y_prob)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    

def format_metrics_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Format metrics into a DataFrame for better visualization."""
    df = pd.DataFrame(results).T
    df.index.name = 'Model'
    return df


def log_metrics(model_name: str, metrics: Dict[str, float], split: str = 'val') -> None:
    logger.info(f"{model_name} - {split} metrics: {metrics}")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: {v}")