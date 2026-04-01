"""
Central configuration for the project
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for d in [MODELS_DIR, MODELS_DIR / "baselines", MODELS_DIR / "xgboost",
          MODELS_DIR / "lightgbm", MODELS_DIR / "tft", MODELS_DIR / "best",
          MLFLOW_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# TFT Configuration
@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    max_encoder_length: int = 48        # 2 days lookback (frees up training samples)
    max_prediction_length: int = 1      # 1-step ahead
    hidden_size: int = 64               # Larger for better representation
    attention_head_size: int = 4
    dropout: float = 0.15
    hidden_continuous_size: int = 32
    learning_rate: float = 0.005
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20                  # Early stopping patience
    gradient_clip_val: float = 0.1
    
TFT_CONFIG = TFTConfig()

# Data Files 
CITYWIDE_FEATURES_PATH = PROCESSED_DATA_DIR / "chirde.citywide_hourly.parquet"
H3_FEATURES_PATH = PROCESSED_DATA_DIR / "chirde.h3_hourly_comp.parquet"
FEATURE_REGISTRY_PATH = PROCESSED_DATA_DIR / "feature_registry.json"

# MLflow
MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR.resolve()}"
MLFLOW_EXPERIMENT_CITYWIDE = "T1_Citywide_Demand"
MLFLOW_EXPERIMENT_REPOSITION = "T2_Repositioning"
MLFLOW_EXPERIMENT_H3_DEMAND = "T3_H3_Demand"
MLFLOW_EXPERIMENT_SURGE = "T4_Surge_Detection"

RANDOM_STATE = 42

# Time Split Configuration
# With ~31 days of data (Dec 1, 2024 – Jan 1, 2025):
#   Train: Dec 1–22 (~22 days, ~528 hours)  — 71%
#   Validation: Dec 23–27 (~5 days, ~120 hours)  — 16%
#   Test: Dec 28–Jan 1 (~4-5 days, ~97 hours)  — 13%
#   Christmas (Dec 25) in validation and NYE (Dec 31) in test
TRAIN_END = "2024-12-22 23:00:00"
VAL_END = "2024-12-27 23:00:00"


@dataclass
class TargetConfig:
    """Configuration for a single prediction target."""
    name: str
    target_column: str
    task_type: str  # "regression" or "classification"
    mlflow_experiment: str
    primary_metric: str  # Main metric for optimization
    dataset: str  # "citywide" or "h3"
    description: str = ""
    class_weight: Optional[str] = None 


# Target Definitions
TARGETS = {
    "T1": TargetConfig(
        name="T1_Citywide_Demand",
        target_column="target_1h",
        task_type="regression",
        mlflow_experiment=MLFLOW_EXPERIMENT_CITYWIDE,
        primary_metric="rmse",
        dataset="citywide",
        description="Predict total citywide ride count for the next hour"
    ),
    "T2": TargetConfig(
        name="T2_Repositioning",
        target_column="reposition_signal",
        task_type="classification",
        mlflow_experiment=MLFLOW_EXPERIMENT_REPOSITION,
        primary_metric="f1",
        dataset="h3",
        description="Predict which H3 cells will need driver repositioning",
        class_weight="balanced"
    ),
    "T3": TargetConfig(
        name="T3_H3_Demand",
        target_column="target_1h",
        task_type="regression",
        mlflow_experiment=MLFLOW_EXPERIMENT_H3_DEMAND,
        primary_metric="rmse",
        dataset="h3",
        description="Predict demand per H3 cell for next hour"
    ),
    "T4": TargetConfig(
        name="T4_Surge_Detection",
        target_column="is_surge",
        task_type="classification",
        mlflow_experiment=MLFLOW_EXPERIMENT_SURGE,
        primary_metric="f1",
        dataset="citywide",
        description="Predict whether next hour will be a demand surge",
        class_weight="balanced"
    ),
}


def load_feature_registry() -> Dict:
    with open(FEATURE_REGISTRY_PATH, "r") as f:
        return json.load(f)


def get_feature_columns(target_key: str) -> List[str]:
    registry = load_feature_registry()
    target_cfg = TARGETS[target_key]

    if target_cfg.dataset == "citywide":
        return registry.get("all_numeric_citywide", [])
    else:
        return registry.get("all_numeric_h3", [])