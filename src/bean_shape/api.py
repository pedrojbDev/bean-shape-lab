"""HTTP interface to the exported model."""

import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import predict


class PredictionRequest(BaseModel):
    features: Annotated[
        list[Annotated[float, Field(allow_inf_nan=False)]], Field(min_length=16, max_length=16)
    ]


def create_app(model):
    app = FastAPI(title="Bean Shape Lab", version="1.0.0")

    @app.get("/health")
    def health():
        return {"model_loaded": model is not None}

    @app.post("/predict")
    def prediction(request: PredictionRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="Model artifact is unavailable")
        return predict(model, request.features)

    return app


def load_default():
    path = Path(__file__).resolve().parents[2] / "model" / "model.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


app = create_app(load_default())
