"""Prediction API (M5; D-018, D-019).

Endpoints
---------
GET  /health    liveness and the loaded model version
GET  /model     serving metadata: model, training window, threshold and its
                rule, champion decision and acceptance outcome (D-036, D-037)
POST /predict   1..max_flights_per_request flights -> probability of ArrDel15
                (arrival 15+ minutes late) and an alert flag per flight

Input contract (D-003, D-025): only the eight pre-departure fields the feature
code reads. Unknown fields are rejected (HTTP 422), so outcome fields such as
ArrDel15 or DepDelay can never reach the model. Value ranges are the same as
the validation rule for usable rows (validate.row_quarantine_mask), plus a
valid minute in hhmm times. Categories not seen in training are scored as
missing (D-025) and reported in unknown_categories. An implausible scheduled
duration is treated as missing (D-028) and reported in
crs_elapsed_time_used = false.

The bundle is loaded once at start-up from FLIGHTDELAY_SERVING_DIR; the
application does not start without a complete bundle.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from flightdelay.features import CATEGORICAL_SOURCES
from flightdelay.serving import (
    DEFAULT_BUNDLE_DIR,
    SERVING_DIR_ENV,
    Bundle,
    load_bundle,
    load_serving_config,
    score_frame,
)
from flightdelay.validate import implausible_duration_mask

MAX_FLIGHTS = int(load_serving_config()["api"]["max_flights_per_request"])


class Flight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flight_id: str | None = Field(default=None, max_length=64)
    flight_date: date
    reporting_airline: str = Field(pattern=r"^[A-Z0-9]{2}$")
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    dest: str = Field(pattern=r"^[A-Z]{3}$")
    crs_dep_time: int = Field(ge=0, le=2400, description="scheduled local departure, hhmm")
    crs_arr_time: int = Field(ge=0, le=2400, description="scheduled local arrival, hhmm")
    crs_elapsed_time: float | None = Field(default=None, description="scheduled minutes")
    distance: float = Field(ge=1, le=6000, description="miles")

    @field_validator("crs_dep_time", "crs_arr_time")
    @classmethod
    def _valid_minute(cls, v: int) -> int:
        if v % 100 >= 60:
            raise ValueError("minutes part of hhmm must be 00-59")
        return v


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flights: Annotated[list[Flight], Field(min_length=1, max_length=MAX_FLIGHTS)]


class Prediction(BaseModel):
    flight_id: str | None
    delay_probability: float
    delay_alert: bool
    unknown_categories: list[str]
    crs_elapsed_time_used: bool


class PredictResponse(BaseModel):
    model_version: str | None
    model_type: str
    threshold: float
    predictions: list[Prediction]


# Request field -> processed-batch column (the schema the model was trained on).
FIELD_TO_COLUMN = {
    "flight_date": "FlightDate",
    "reporting_airline": "Reporting_Airline",
    "origin": "Origin",
    "dest": "Dest",
    "crs_dep_time": "CRSDepTime",
    "crs_arr_time": "CRSArrTime",
    "crs_elapsed_time": "CRSElapsedTime",
    "distance": "Distance",
}


def to_batch(flights: list[Flight]) -> pd.DataFrame:
    """Flights -> rows in the processed-batch format with the ingest dtypes."""
    rows = [f.model_dump(include=set(FIELD_TO_COLUMN)) for f in flights]
    df = pd.DataFrame(rows, columns=list(FIELD_TO_COLUMN)).rename(columns=FIELD_TO_COLUMN)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    for col in ("Reporting_Airline", "Origin", "Dest"):
        df[col] = df[col].astype("category")
    df["CRSDepTime"] = df["CRSDepTime"].astype("Int16")
    df["CRSArrTime"] = df["CRSArrTime"].astype("Int16")
    df["CRSElapsedTime"] = pd.to_numeric(df["CRSElapsedTime"]).astype("float32")
    df["Distance"] = df["Distance"].astype("float32")
    return df


def predict(bundle: Bundle, flights: list[Flight]) -> PredictResponse:
    df = to_batch(flights)
    proba = score_frame(bundle.model, df)
    duration_used = (df["CRSElapsedTime"].notna() & ~implausible_duration_mask(df)).to_numpy()
    vocab = bundle.model.vocab
    preds = []
    for i, f in enumerate(flights):
        unknown = [feat for feat, col in CATEGORICAL_SOURCES.items()
                   if str(df.at[i, col]) not in vocab[feat]]
        p = float(proba.iloc[i])
        preds.append(Prediction(flight_id=f.flight_id, delay_probability=p,
                                delay_alert=p >= bundle.threshold, unknown_categories=unknown,
                                crs_elapsed_time_used=bool(duration_used[i])))
    return PredictResponse(model_version=bundle.meta.get("model_version"),
                           model_type=bundle.meta["model_type"], threshold=bundle.threshold,
                           predictions=preds)


def create_app(bundle_dir: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = bundle_dir or os.environ.get(SERVING_DIR_ENV) or DEFAULT_BUNDLE_DIR
        app.state.bundle = load_bundle(path)  # fails start-up if the bundle is not valid
        yield

    app = FastAPI(title="Flight arrival-delay prediction", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        b: Bundle = request.app.state.bundle
        return {"status": "ok", "model_type": b.meta["model_type"],
                "model_version": b.meta.get("model_version")}

    @app.get("/model")
    def model_info(request: Request) -> dict[str, Any]:
        meta = dict(request.app.state.bundle.meta)
        meta.pop("params", None)
        return meta

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(req: PredictRequest, request: Request) -> PredictResponse:
        return predict(request.app.state.bundle, req.flights)

    return app


app = create_app()
