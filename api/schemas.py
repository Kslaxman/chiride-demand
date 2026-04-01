from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime as dt_type


class SinglePredictionRequest(BaseModel):
    h3_index: Optional[str] = Field(None, description="H3 hexagon index")
    pickup_datetime: dt_type = Field(..., description="Prediction time")
    temperature: Optional[float] = Field(None, description="Temperature F")
    humidity: Optional[float] = Field(None, description="Humidity pct")
    wind_speed: Optional[float] = Field(None, description="Wind speed mph")
    precipitation: Optional[float] = Field(None, description="Precipitation in")


class BatchPredictionRequest(BaseModel):
    predictions: List[SinglePredictionRequest] = Field(
        ..., min_length=1, max_length=1000
    )


class PredictionResponse(BaseModel):
    h3_index: Optional[str] = None
    pickup_datetime: str
    predictions: Optional[Dict[str, Any]] = None
    predicted_rides: Optional[float] = None
    predicted_rides_raw: Optional[float] = None
    model_info: Optional[dict] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    model_info: dict
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    models_loaded: List[str]
    model_count: int
