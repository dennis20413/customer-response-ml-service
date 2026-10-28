import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schema import FEATURES, PredictionRequest
from .scoring import Model

logger = logging.getLogger(__name__)


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = None
        try:
            app.state.model = Model.load(model_path or os.getenv("MODEL_PATH", "artifacts/model.json"))
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Model unavailable. Train and validate a local artifact before serving.")
        yield

    app = FastAPI(title="Customer Response ML Service", version="1.0.0", lifespan=lifespan)

    def ready():
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model unavailable; train a local model first")
        return app.state.model

    @app.get("/health")
    def health():
        model = ready()
        return {"status": "ready", "model_version": model.version}

    @app.get("/model")
    def metadata():
        model = ready()
        return {"model_version": model.version, "features": FEATURES,
                "threshold": model.threshold, "threshold_policy": "validation_f1_maximum",
                "score_remapping": {"slope": float(model.calibration[0]),
                                    "reverses_ranking": bool(model.calibration[0] < 0)},
                "known_categories": dict(zip(model.payload["categorical"], model.categories)),
                "training": model.payload["training"],
                "scope": "Historical research demo; not a lending or contact decision system"}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        model = ready()
        records = [row.model_dump() for row in request.customers]
        probabilities = model.predict(records)
        return {"model_version": model.version, "threshold": model.threshold,
                "unknown_category_counts": model.unknown_categories(records),
                "predictions": [{"response_probability": float(p),
                                 "above_validation_f1_threshold": bool(p >= model.threshold)}
                                for p in probabilities]}

    return app


app = create_app()
