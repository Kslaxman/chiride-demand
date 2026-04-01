"""
Temporal Fusion Transformer trainer for T1 Citywide Demand prediction.

Uses pytorch-forecasting's TemporalFusionTransformer implementation with
Lightning for training.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pickle
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=UserWarning)

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE, MAE, SMAPE
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.config import (
    TARGETS, MODELS_DIR, PROCESSED_DATA_DIR, MLFLOW_TRACKING_URI,
    RANDOM_STATE, TRAIN_END, VAL_END, TFT_CONFIG,
    CITYWIDE_FEATURES_PATH
)
from src.models.evaluation import compute_metrics

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Feature Classification
# Time-varying KNOWN features: calendar/time features known in advance
TIME_VARYING_KNOWN = [
    "hour", "day_of_week", "day_of_month", "week_of_year",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "dom_sin", "dom_cos",
    "is_weekend", "is_holiday", "is_nye_evening", "holiday_proximity",
    "is_party_time", "is_work_commute", "is_late_night", "is_business_day",
]

# Time-varying UNKNOWN features: observed but not known in advance
TIME_VARYING_UNKNOWN = [
    # Demand metrics (only known after observation)
    "trip_count",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
    "demand_change_1h", "demand_change_24h", "demand_pct_change_1h",
    "roll_3h_mean", "roll_3h_std", "roll_3h_min", "roll_3h_max",
    "roll_6h_mean", "roll_6h_std", "roll_6h_min", "roll_6h_max",
    "roll_24h_mean", "roll_24h_std", "roll_24h_min", "roll_24h_max",
    "roll_168h_mean", "roll_168h_std", "roll_168h_min", "roll_168h_max",
    "demand_vs_3h_avg", "demand_vs_24h_avg", "short_vs_long_trend",
    # Weather (observed in real-time)
    "temperature_f", "feels_like_f", "humidity", "precipitation_mm",
    "snowfall_cm", "wind_speed_kmh", "weather_code", "is_raining",
    "is_snowing", "weather_misery",
    # Auxiliary metrics
    "avg_fare", "avg_duration_sec", "avg_distance", "shared_pct", "avg_tip",
    "unique_areas", "unique_h3_cells",
    # Slot features
    "slot_avg", "slot_std", "vs_slot_avg", "slot_zscore", "slot_deviation",
    "slot_avg_leakfree", "vs_slot_avg_leakfree",
    "slot_avg_exact", "slot_avg_daytype", "slot_avg_hour",
    # CTA
    "cta_total_rides", "cta_avg_ratio", "cta_stations_low",
]


def prepare_citywide_data(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load and prepare citywide data for TFT.
    
    Adds required columns: time_idx, group_id
    Ensures proper sorting and no NaN in key columns.
    """
    if df is None:
        df = pd.read_parquet(CITYWIDE_FEATURES_PATH)
    
    df = df.copy()
    
    if "hour_bucket" in df.columns:
        df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])
        df = df.sort_values("hour_bucket").reset_index(drop=True)
    
    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "citywide"  
    
    all_features = TIME_VARYING_KNOWN + TIME_VARYING_UNKNOWN
    available_features = [f for f in all_features if f in df.columns]
    df[available_features] = df[available_features].fillna(0.0)
    
    df["target_1h"] = df["target_1h"].ffill().fillna(0.0)
    
    for col in available_features + ["target_1h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    return df


def create_tft_datasets(
    df: pd.DataFrame,
    cfg: Optional[object] = None,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """Create train/val/test TimeSeriesDataSets from prepared DataFrame.
    
    Returns:
        (training_dataset, validation_dataset, test_dataset)
    """
    if cfg is None:
        cfg = TFT_CONFIG
    
    train_end_dt = pd.Timestamp(TRAIN_END)
    val_end_dt = pd.Timestamp(VAL_END)
    
    train_end_idx = int(df.loc[df["hour_bucket"] <= train_end_dt, "time_idx"].max())
    val_end_idx = int(df.loc[df["hour_bucket"] <= val_end_dt, "time_idx"].max())
    
    # Train: everything up to train_end_idx
    train_df = df[df["time_idx"] <= train_end_idx].copy()
    
    # Val + Test: full dataframe so TFT can build encoder windows
    # We'll constrain which time steps are predicted using min_prediction_idx
    
    known_available = [f for f in TIME_VARYING_KNOWN if f in df.columns]
    unknown_available = [f for f in TIME_VARYING_UNKNOWN if f in df.columns]
    
    val_count = len(df[(df["hour_bucket"] > train_end_dt) & (df["hour_bucket"] <= val_end_dt)])
    test_count = len(df[df["hour_bucket"] > val_end_dt])
    
    print(f"  Known features: {len(known_available)}")
    print(f"  Unknown features: {len(unknown_available)}")
    print(f"  Train rows: {len(train_df)}, Val rows: {val_count}, Test rows: {test_count}")
    
    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target_1h",
        group_ids=["group_id"],
        max_encoder_length=cfg.max_encoder_length,
        max_prediction_length=cfg.max_prediction_length,
        time_varying_known_reals=known_available,
        time_varying_unknown_reals=unknown_available + ["target_1h"],
        target_normalizer=GroupNormalizer(
            groups=["group_id"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Validation dataset: use full data up to val_end, 
    # but only predict after train_end using min_prediction_idx
    val_df = df[df["time_idx"] <= val_end_idx].copy()
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_df,
        min_prediction_idx=train_end_idx + 1,
        stop_randomization=True,
    )
    
    # Test dataset: use full data, predict only after val_end
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df,
        min_prediction_idx=val_end_idx + 1,
        stop_randomization=True,
    )
    
    return training_dataset, validation_dataset, test_dataset


def train_tft(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    cfg: Optional[object] = None,
    log_to_mlflow: bool = True,
) -> Tuple[TemporalFusionTransformer, pl.Trainer, Dict]:
    """Train a TFT model.
    
    Returns:
        (model, trainer, best_metrics)
    """
    if cfg is None:
        cfg = TFT_CONFIG
    
    # Set seed
    pl.seed_everything(RANDOM_STATE)
    
    # Data loaders
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=cfg.batch_size,
        num_workers=0,
    )
    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=cfg.batch_size,
        num_workers=0,
    )
    
    # Create TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        hidden_continuous_size=cfg.hidden_continuous_size,
        loss=RMSE(),
        optimizer="adam",
        reduce_on_plateau_patience=cfg.patience // 2,
        log_interval=10,
        log_val_interval=1,
    )
    
    n_params = sum(p.numel() for p in tft.parameters())
    print(f"  TFT parameters: {n_params:,}")
    print(f"  Encoder length: {cfg.max_encoder_length}")
    print(f"  Hidden size: {cfg.hidden_size}")
    print(f"  Attention heads: {cfg.attention_head_size}")
    
    # Checkpoint directory
    ckpt_dir = MODELS_DIR / "tft" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        min_delta=1e-4,
        verbose=True,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    
    # Determine accelerator — CPU for small dataset
    accelerator = "cpu"
    
    print(f"  Accelerator: {accelerator}")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=[early_stop, lr_monitor, checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=5,
    )
    
    # Train
    print(f"\n  Training TFT for up to {cfg.max_epochs} epochs...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Load best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"  Best checkpoint: {best_model_path}")
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    else:
        best_tft = tft
    
    return best_tft, trainer, {"best_val_loss": float(checkpoint_callback.best_model_score or 0)}


def evaluate_tft(
    model: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    df: pd.DataFrame,
    split_name: str = "val",
) -> Dict[str, float]:
    """Evaluate TFT model and return regression metrics."""
    loader = dataset.to_dataloader(
        train=False,
        batch_size=128,
        num_workers=0,
    )
    
    # Get predictions
    predictions = model.predict(loader, return_x=False, trainer_kwargs={"accelerator": "cpu"})
    predictions = predictions.cpu().numpy().flatten()
    
    actuals = torch.cat([y[0] for x, y in iter(loader)]).cpu().numpy().flatten()
    
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]
    
    metrics = compute_metrics(actuals, predictions, "regression")
    
    print(f"\n  {split_name.upper()} Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    
    return metrics


def save_tft_model(
    model: TemporalFusionTransformer,
    training_dataset: TimeSeriesDataSet,
    val_metrics: Dict[str, float],
    target_key: str = "T1",
    feature_names: Optional[List[str]] = None,
) -> Path:
    """Save the TFT model in the standard best model format.
    
    Saves:
      - Full model checkpoint (.ckpt) for pytorch-forecasting loading
      - Dataset parameters (pickle for reconstruction at inference)
      - metadata.json (compatible with existing model loader)
      - features.json
    """
    target_cfg = TARGETS[target_key]
    save_dir = MODELS_DIR / "best" / target_key / "tft"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = save_dir / "model.ckpt"
    model.to("cpu")
    
    trainer = pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(str(ckpt_path))
    print(f"  Checkpoint saved: {ckpt_path}")
    
    # Save dataset parameters for reconstruction at inference
    dataset_params = training_dataset.get_parameters()
    params_path = save_dir / "dataset_params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(dataset_params, f)
    
    # Build feature list from TFT feature groups
    if feature_names is None:
        feature_names = (
            list(training_dataset.time_varying_known_reals) +
            list(training_dataset.time_varying_unknown_reals)
        )
    
    # Save metadata (compatible with existing model loader)
    metadata = {
        "target_key": target_key,
        "target_name": target_cfg.name,
        "target_column": target_cfg.target_column,
        "task_type": target_cfg.task_type,
        "model_type": "tft",
        "val_metrics": val_metrics,
        "params": {
            "max_encoder_length": TFT_CONFIG.max_encoder_length,
            "max_prediction_length": TFT_CONFIG.max_prediction_length,
            "hidden_size": TFT_CONFIG.hidden_size,
            "attention_head_size": TFT_CONFIG.attention_head_size,
            "dropout": TFT_CONFIG.dropout,
            "hidden_continuous_size": TFT_CONFIG.hidden_continuous_size,
            "learning_rate": TFT_CONFIG.learning_rate,
        },
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "saved_at": datetime.now().isoformat(),
    }
    meta_path = save_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    feat_path = save_dir / "features.json"
    with open(feat_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n  Model saved to: {save_dir}")
    print(f"  Features: {len(feature_names)}")
    
    return save_dir
