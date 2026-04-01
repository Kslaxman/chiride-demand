import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def calculate_psi(expected_series: pd.Series, actual_series: pd.Series, max_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Calculate the Population Stability Index (PSI) for two series (expected/reference vs actual/current).
    
    Args:
        expected_series: The reference data series (e.g., training data).
        actual_series: The current data series to compare (e.g., recent test data).
        max_bins: Number of quantiles to use.
        epsilon: Small constant to avoid zero division and log(0).
        
    Returns:
        PSI score. Guidelines:
        < 0.1: No significant drift.
        0.1 - 0.2: Moderate drift.
        > 0.2: Significant drift.
    """
    expected = expected_series.dropna()
    actual = actual_series.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Ensure max bins does not exceed number of unique values
    n_unique = expected.nunique()
    n_bins = min(max_bins, n_unique) if n_unique > 0 else max_bins
    n_bins = max(1, n_bins)
    
    if n_bins == 1:
        return 0.0  # Cannot perform binning effectively
        
    # Create bins using expected series
    try:
        bins = pd.qcut(expected, q=n_bins, retbins=True, duplicates='drop')[1]
    except ValueError:
        # Fallback if qcut fails due to extreme skewness or categorical-like numerics
        bins = np.linspace(expected.min(), expected.max(), n_bins + 1)
        
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calculate expected and actual probabilities per bin
    expected_cuts = pd.cut(expected, bins=bins)
    actual_cuts = pd.cut(actual, bins=bins)

    expected_pct = expected_cuts.value_counts(normalize=True).sort_index().values
    actual_pct = actual_cuts.value_counts(normalize=True).sort_index().values

    # Replace zeros with epsilon
    expected_pct = np.clip(expected_pct, epsilon, 1.0)
    actual_pct = np.clip(actual_pct, epsilon, 1.0)
    
    # Normalize again to ensure they sum to 1 after clipping
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()

    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def detect_drift_report(df_ref: pd.DataFrame, df_cur: pd.DataFrame, features: List[str], 
                        threshold: float = 0.2) -> Dict:
    """
    Generate a drift report over a list of numeric features using PSI.
    
    Args:
        df_ref: Reference DataFrame (e.g. Training Data)
        df_cur: Current DataFrame (e.g. Test / Recent Data)
        features: List of feature column names to compute drift on.
        threshold: PSI threshold determining significant drift.
        
    Returns:
        A dictionary report with overall drift status and individual feature PSI scores.
    """
    psi_results = {}
    drifted_features = []
    
    for feat in features:
        if feat in df_ref.columns and feat in df_cur.columns:
            if pd.api.types.is_numeric_dtype(df_ref[feat]):
                psi_score = calculate_psi(df_ref[feat], df_cur[feat])
                psi_results[feat] = psi_score
                
                if psi_score > threshold:
                    drifted_features.append(feat)
            else:
                # Fallback for strict categorical
                psi_results[feat] = 0.0
                
    n_features = len(psi_results)
    n_drifted = len(drifted_features)
    
    report = {
        "status": "DRIFT_DETECTED" if n_drifted > 0 else "NO_DRIFT",
        "drifted_features": drifted_features,
        "n_drifted": n_drifted,
        "n_features": n_features,
        "percent_drifted": round((n_drifted / max(n_features, 1)) * 100, 2),
        "scores": dict(sorted(psi_results.items(), key=lambda item: item[1], reverse=True))
    }
    
    return report
