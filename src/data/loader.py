"""
Data loading and train/val/test splitting utilities.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path

from src.config import (
    CITYWIDE_FEATURES_PATH, H3_FEATURES_PATH,
    TRAIN_END, VAL_END, RANDOM_STATE
)
from src.utils.logging_config import setup_logging

logger = setup_logging("data.loader")

DATETIME_CANDIDATES = ["hour_bucket", "hour_start", "timestamp", "datetime", "date"]

def _find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for col in DATETIME_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _get_time_series(df: pd.DataFrame) -> pd.Series:
    # Case 1: DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series()

    # Case 2: Named index that looks like a datetime column
    if df.index.name in DATETIME_CANDIDATES:
        return pd.to_datetime(df.index).to_series()

    # Case 3: Datetime column exists
    dt_col = _find_datetime_column(df)
    if dt_col is not None:
        return pd.to_datetime(df[dt_col])

    raise ValueError(
        f"Cannot find datetime information.\n"
        f"Index name: {df.index.name}\n"
        f"Index type: {type(df.index)}\n"
        f"Columns: {sorted(df.columns.tolist())[:20]}"
    )


def load_citywide_data() -> pd.DataFrame:
    logger.info(f"Loading citywide data from {CITYWIDE_FEATURES_PATH}")
    df = pd.read_parquet(CITYWIDE_FEATURES_PATH)

    logger.info(f"Raw columns: {list(df.columns[:10])}... ({len(df.columns)} total)")

    # Find the datetime column and set as index
    dt_col = _find_datetime_column(df)

    if dt_col is not None:
        logger.info(f"Found datetime column: '{dt_col}'")
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col).sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        logger.info(f"Index is already DatetimeIndex")
        df = df.sort_index()
    else:
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            logger.info(f"Converted existing index to DatetimeIndex")
        except Exception:
            logger.warning("Could not find or create a datetime index")

    return df


def load_h3_data(sample_frac: Optional[float] = None) -> pd.DataFrame:
    logger.info(f"Loading H3 data from {H3_FEATURES_PATH}")
    df = pd.read_parquet(H3_FEATURES_PATH)

    logger.info(f"  Raw columns: {list(df.columns[:10])}... ({len(df.columns)} total)")

    dt_col = _find_datetime_column(df)
    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col])
        logger.info(f"  Datetime column: '{dt_col}'")
    else:
        logger.warning("  No datetime column found in H3 data!")

    sort_cols = []
    if dt_col:
        sort_cols.append(dt_col)
    if "h3_index" in df.columns:
        sort_cols.append("h3_index")
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    logger.info(f"Shape: {df.shape}")

    if sample_frac is not None and "h3_index" in df.columns:
        unique_cells = df["h3_index"].unique()
        n_sample = max(1, int(len(unique_cells) * sample_frac))
        rng = np.random.RandomState(RANDOM_STATE)
        sampled_cells = rng.choice(unique_cells, size=n_sample, replace=False)
        df = df[df["h3_index"].isin(sampled_cells)].reset_index(drop=True)
        logger.info(f"  Sampled {n_sample}/{len(unique_cells)} cells -> {df.shape}")

    return df


def temporal_split_citywide(df: pd.DataFrame, train_end: str = TRAIN_END, val_end: str = VAL_END) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)
    
    time_vals = _get_time_series(df)

    train = df[time_vals <= train_end_dt].copy()
    val = df[(time_vals > train_end_dt) & (time_vals <= val_end_dt)].copy()
    test = df[time_vals > val_end_dt].copy()

    logger.info("Temporal split (citywide):")
    logger.info(f"  Train: {len(train)} rows")
    logger.info(f"  Val:   {len(val)} rows")
    logger.info(f"  Test:  {len(test)} rows")

    for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
        if len(subset) == 0:
            logger.warning(f"  WARNING: {name} set is EMPTY!")
        else:
            sub_time = _get_time_series(subset)
            logger.info(f"  {name}: {sub_time.min()} to {sub_time.max()}")

    return train, val, test


def temporal_split_h3(df: pd.DataFrame, train_end: str = TRAIN_END, val_end: str = VAL_END) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    time_vals = _get_time_series(df)

    train = df[time_vals <= train_end_dt].copy()
    val = df[(time_vals > train_end_dt) & (time_vals <= val_end_dt)].copy()
    test = df[time_vals > val_end_dt].copy()

    for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
        if len(subset) == 0:
            logger.warning(f"  WARNING: {name} set is EMPTY!")

    return train, val, test


def prepare_xy(df: pd.DataFrame, target_col: str, feature_cols: List[str], drop_na: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y from a DataFrame.
    """
    available_features = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        logger.warning(f"  Missing {len(missing)} features: {sorted(missing)[:10]}...")

    if target_col not in df.columns:
        if df.index.name == target_col:
            raise ValueError(
                f"Target '{target_col}' is the index, not a column. "
                f"Reset the index first with df.reset_index()."
            )
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # Make sure target is not also in features
    available_features = [c for c in available_features if c != target_col]

    subset = df[available_features + [target_col]].copy()

    if drop_na:
        before = len(subset)
        subset = subset.dropna(subset=[target_col])
        dropped = before - len(subset)
        if dropped > 0:
            logger.info(f"  Dropped {dropped} rows with NaN target")

    X = subset[available_features]
    y = subset[target_col]

    # Fill remaining NaN features with 0 (lag features at start of series)
    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        logger.info(f"Filling NaN in {len(cols_with_nan)} feature columns with 0")
        X = X.fillna(0)

    return X, y