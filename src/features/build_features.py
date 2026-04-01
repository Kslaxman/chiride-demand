"""
Feature selection and preparation utilities
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json

from src.config import FEATURE_REGISTRY_PATH, load_feature_registry
from src.utils.logging_config import setup_logging

logger = setup_logging('features.build')


def get_feature_groups(dataset: str = 'citywide') -> Dict[str, List[str]]:
    """"Load feature groups from the registry for a given dataset type."""
    registry = load_feature_registry()

    groups = {}
    for key, value in registry.items():
        if key.startswith('features_') and isinstance(value, list):
            group_name = key.replace('features_', '')
            groups[group_name] = value

    logger.info(f'Feature groups for {dataset}')
    for name, cols in groups.items():
        logger.info(f'  {name}: {len(cols)} features')

    return groups


def select_features(df: pd.DataFrame, feature_cols: List[str], variance_threshold: float = 0.0, correlation_threshold: float = 0.98, max_features: Optional[int] = None) -> List[str]:
    """Select features by removing zero-variance and highly correlated features."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].select_dtypes(include=[np.number])
    selected = list(X.columns)

    # remove zero/near-zero variance
    variances = X.var()
    low_var = variances[variances <= variance_threshold].index.tolist()
    if low_var:
        logger.info(f"Removing {len(low_var)} zero-variance features: {low_var[:5]}")
        selected = [c for c in selected if c not in low_var]

    # remove highly correlated features
    if correlation_threshold < 1.0 and len(selected) > 1:
        corr_matrix = X[selected].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] >= correlation_threshold].tolist()
            for hc in high_corr:
                to_drop.add(hc)

        if to_drop: 
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            selected = [c for c in selected if c not in to_drop]
    
    if max_features and len(selected) > max_features:
        logger.info(f"Limiting to top {max_features} features")
        selected = selected[:max_features]

    return selected


def validate_features(train_X: pd.DataFrame, val_X: pd.DataFrame, test_X: pd.DataFrame) -> None:
    for name, X in [('Train', train_X), ('Val', val_X), ('Test', test_X)]:
        nan_pct = (X.isna().sum().sum() / (X.shape[0] * X.shape[1])) * 100
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        