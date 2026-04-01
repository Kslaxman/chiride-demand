from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from api.schemas import (
    SinglePredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse, HealthResponse,
)
from api import model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting API...")
    model_loader.load_model()
    if model_loader.IS_LOADED:
        print(f"API ready - {len(model_loader.MODELS)} models")
    else:
        print("API started without models")
    yield
    print("Shutting down")


app = FastAPI(title="Chicago Ride Demand API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model_loader.IS_LOADED else "degraded",
        model_loaded=model_loader.IS_LOADED,
        models_loaded=list(model_loader.MODELS.keys()),
        model_count=len(model_loader.MODELS),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: SinglePredictionRequest):
    if not model_loader.IS_LOADED:
        raise HTTPException(503, "Models not loaded")
    t0 = time.time()
    results = model_loader.predict_all(
        req.pickup_datetime, req.h3_index,
        req.temperature, req.humidity, req.wind_speed, req.precipitation,
    )
    ms = (time.time() - t0) * 1000
    t3 = results.get("T3", {"value": 0, "raw": 0.0})
    return PredictionResponse(
        h3_index=req.h3_index,
        pickup_datetime=req.pickup_datetime.isoformat(),
        predictions=results,
        predicted_rides=t3["value"],
        predicted_rides_raw=t3["raw"],
        model_info={"inference_ms": round(ms, 2), "models": list(results.keys())},
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest):
    if not model_loader.IS_LOADED:
        raise HTTPException(503, "Models not loaded")
    t0 = time.time()
    req_dicts = [
        {"pickup_datetime": r.pickup_datetime, "h3_index": r.h3_index,
         "temperature": r.temperature, "humidity": r.humidity,
         "wind_speed": r.wind_speed, "precipitation": r.precipitation}
        for r in req.predictions
    ]
    batch_results = model_loader.predict_batch(req_dicts)
    ms = (time.time() - t0) * 1000
    responses = []
    for i, results in enumerate(batch_results):
        r = req.predictions[i]
        t3 = results.get("T3", {"value": 0, "raw": 0.0})
        responses.append(PredictionResponse(
            h3_index=r.h3_index,
            pickup_datetime=r.pickup_datetime.isoformat(),
            predictions=results,
            predicted_rides=t3["value"],
            predicted_rides_raw=t3["raw"],
            model_info={},
        ))
    return BatchPredictionResponse(
        predictions=responses, count=len(responses),
        model_info={"total_ms": round(ms, 2),
                    "avg_ms": round(ms / max(len(responses), 1), 2)},
    )


@app.get("/model/info")
async def model_info():
    if not model_loader.IS_LOADED:
        raise HTTPException(503, "Models not loaded")
    return {t: {"name": model_loader.METADATA[t].get("target_name"),
                "task": model_loader.METADATA[t].get("task_type"),
                "algo": model_loader.METADATA[t].get("model_type"),
                "features": len(model_loader.FEATURES[t]),
                "metrics": model_loader.METADATA[t].get("val_metrics", {})}
            for t in model_loader.MODELS}
