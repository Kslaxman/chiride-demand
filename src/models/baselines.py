import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass

from src.models.evaluation import compute_metrics, log_metrics
from src.utils.logging_config import setup_logging

logger = setup_logging('models.baselines')

####
# Regression baselines
####

class NaiveLastValue:
    """Predict the current value as the next value"""

    def __init__(self):
        self.name = "Naive_LastValue"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """lag_1h as prediction"""
        if 'lag_1h' in X.columns:
            preds = X['lag_1h'].fillna(0).values
        elif 'trip_count' in X.columns:
            preds = X['trip_count'].values
        else:
            preds = np.zeros(len(X))
        return preds
    

class NaiveSameHourYesterday:
    """Predict the same hour from yesterday (lag_24h)"""

    def __init__(self):
        self.name = "Naive_SameHourYesterday"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if 'lag_24h' in X.columns:
            return X['lag_24h'].fillna(0).values
        return np.zeros(len(X))
    

class NaiveSameHourLastWeek:
    """Predict the same hour from last week (lag_168h)"""

    def __init__(self):
        self.name = "Naive_SameHourLastWeek"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if 'lag_168h' in X.columns:
            return X['lag_168h'].fillna(0).values
        return np.zeros(len(X))
    

class NaiveHistoricalMean:
    """Predict the training set mean for all observations"""

    def __init__(self):
       self.name = "Naive_HistoricalMean"
       self.mean_value = 0.0

    def fit(self, y_train: np.ndarray):
        self.mean_value = np.nanmean(y_train)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_value)
    

class NaiveSlotAverage:
    """Predict using time-slot average (hour x day-of-week)"""

    def __init__(self):
        self.name = "Naive_SlotAverage"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if 'slot_avg_leakfree' in X.columns:
            return X['slot_avg_leakfree'].fillna(0).values
        elif 'slot_avg' in X.columns:
            return X['slot_avg'].fillna(0).values
        return np.zeros(len(X))
    

class NaiveRollingMean:
    """Predict using rolling 3-hour mean"""

    def __init__(self):
        self.name = "Naive_Rolling3hMean"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "roll_3h_mean" in X.columns:
            return X['roll_3h_mean'].fillna(0).values
        return np.zeros(len(X))
    

####
## Classification baselines
####

class NaiveAlwaysMajority:
    """Majority class"""

    def __init__(self):
        self.name = "Naive_AlwaysMajority"
        self.majority_class = 0

    def fit(self, y_train: np.ndarray):
        values, counts = np.unique(y_train[~np.isnan(y_train)], return_counts=True)
        self.majority_class = int(values[np.argmax(counts)])
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prob for positive class"""
        if self.majority_class == 1:
            return np.ones(len(X))
        return np.zeros(len(X))
    

class NaiveClassRate:
    """Predict class 1 with the training class frequency"""

    def __init__(self):
        self.name = "Naive_ClassRate"
        self.positive_rate = 0.0

    def fit(self, y_train: np.ndarray):
        valid = y_train[~np.isnan(y_train)]
        self.positive_rate = np.mean(valid == 1)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(42)
        return (rng.random(len(X)) < self.positive_rate).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.positive_rate)
    

class NaiveThresholdBaseline:
    """Surge detection: predict surge if demand > threshold"""

    def __init__(self, feature: str = "demand_vs_24h_avg", threshold: float = 1.5):
        self.name = f"Naive_Threshold({feature} > {threshold})"
        self.feature = feature
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature in X.columns:
            return (X[self.feature].fillna(0) > self.threshold).astype(int).values
        return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature in X.columns:
            vals = X[self.feature].fillna(0).values
            return np.clip(vals / (self.threshold * 2), 0, 1)
        return np.zeros(len(X))
    

####
## Baselines
####

def run_regression_baselines(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, X_test: pd.DataFrame = None, y_test: np.ndarray = None) -> Dict[str, Dict]:
    logger.info("=" * 60)
    logger.info("Running regression baselines")
    logger.info("=" * 60)

    baselines = [
        NaiveLastValue(),
        NaiveSameHourYesterday(),
        NaiveSameHourLastWeek(),
        NaiveHistoricalMean().fit(y_train),
        NaiveSlotAverage(),
        NaiveRollingMean()
    ]

    results = {}
    for model in baselines:
        logger.info(f"\n {model.name}")

        if hasattr(model, 'fit'):
            model.fit(y_train)

        val_preds = model.predict(X_val)
        val_metrics = compute_metrics(y_val, val_preds, task_type="regression")
        log_metrics(model.name, val_metrics, split='val')

        result = {
            "val_metrics": val_metrics,
            "val_preds": val_preds
        }

        if X_test is not None and y_test is not None:
            test_preds = model.predict(X_test)
            test_metrics = compute_metrics(y_test, test_preds, "regression")
            result['test_metrics'] = test_metrics
            result['test_preds'] = test_preds

        results[model.name] = result

    return results


def run_classification_baselines(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, X_test: pd.DataFrame = None, y_test: np.ndarray = None, target_type: str = 'surge') -> Dict[str, Dict]:
    logger.info("=" * 60)
    logger.info("Running classification baselines")
    logger.info("=" * 60)

    baselines = [
        NaiveAlwaysMajority(),
        NaiveClassRate()
    ]

    if target_type == 'surge':
        baselines.append(NaiveThresholdBaseline('demand_vs_24h_avg', 1.5))

    results = {}
    for model in baselines:
        logger.info(f"\n {model.name}")

        if hasattr(model, "fit"):
            model.fit(y_train)

        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        val_metrics = compute_metrics(y_val, val_preds, "classification", val_preds)
        log_metrics(model.name, val_metrics, "val")

        result = {
            "val_metrics": val_metrics,
            "val_preds": val_preds
        }   

        if X_test is not None and y_test is not None:
            test_preds = model.predict(X_test)
            test_probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            test_metrics = compute_metrics(y_test, test_preds, "classification", test_probs)
            result['test_metrics'] = test_metrics
            result['test_preds'] = test_preds

        results[model.name] = result
    return results