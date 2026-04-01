import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_DIR = MODELS_DIR / "best"


@lru_cache(maxsize=1)
def load_h3_data():
    path = DATA_DIR / "chirde.h3_hourly_comp.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        # Standardize datetime column
        candidates = ["hour_bucket", "hour_start", "timestamp", "datetime", "date"]
        for col in candidates:
            if col in df.columns:
                df["datetime"] = pd.to_datetime(df[col])
                break
        return df
    return pd.DataFrame()


@lru_cache(maxsize=1)
def load_citywide_data():
    path = DATA_DIR / "chirde.citywide_hourly.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@lru_cache(maxsize=1)
def load_h3_features():
    path = DATA_DIR / "chirde.h3_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_all_model_metadata():
    models_info = []
    if not BEST_DIR.exists():
        return models_info
    
    for target_dir in sorted(BEST_DIR.iterdir()):
        if not target_dir.is_dir():
            continue
            
        # Load all algorithm models for this target
        for algo_dir in sorted(target_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
                
            meta_file = algo_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                meta["_path"] = str(algo_dir)
                meta["_target_dir"] = target_dir.name
                meta["_algo_dir"] = algo_dir.name
                models_info.append(meta)
    
    return models_info


def load_feature_importance(target_key, algo):
    # Try the new directory structure: best/{target_key}/{algo}/
    target_dir = BEST_DIR / target_key / algo
    if target_dir.exists():
        for csv_f in target_dir.glob("*_feature_importance.csv"):
            return pd.read_csv(csv_f)
    
    # Fallback for TFT (since TFT doesn't have standard feature importance like trees)
    if algo == "tft":
        # Return empty DF so the UI just shows the feature list
        return pd.DataFrame()
        
    return pd.DataFrame()


def load_comparison_csv():
    path = MODELS_DIR / "full_model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()