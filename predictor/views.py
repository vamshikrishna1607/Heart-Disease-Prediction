import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.shortcuts import render

from .forms import HeartDiseasePredictionForm

logger = logging.getLogger(__name__)

MODEL_DIR = Path(settings.BASE_DIR) / "predictor" / "ml_model"
MODEL_PATH = MODEL_DIR / "heart_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

_model = None
_model_load_error = None
_metrics = None


def _get_model():
    """
    Lazily load the trained pipeline exactly once per process, rather than
    on every request. If loading fails (missing/corrupt artifact), the error
    is cached so the view can degrade gracefully instead of crashing.
    """
    global _model, _model_load_error
    if _model is not None or _model_load_error is not None:
        return _model
    try:
        _model = joblib.load(MODEL_PATH)
        logger.info("Loaded heart disease model from %s", MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - want to surface any load failure
        _model_load_error = str(exc)
        logger.exception("Failed to load heart disease model")
    return _model


def _get_metrics():
    global _metrics
    if _metrics is not None:
        return _metrics
    try:
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    except Exception:  # noqa: BLE001
        _metrics = {}
    return _metrics


def predict(request):
    """
    Single view handling both the empty form (GET) and a submission (POST).
    On POST: validate the form, run the model, and render the result inline
    on the same page. Any unexpected failure during prediction is caught and
    surfaced as a form-level error rather than a 500 page.
    """
    result = None
    error_message = None

    if request.method == "POST":
        form = HeartDiseasePredictionForm(request.POST)
        if form.is_valid():
            model = _get_model()
            if model is None:
                error_message = (
                    "The prediction model is not available right now "
                    f"({_model_load_error}). Please try again later."
                )
            else:
                try:
                    # The model's ColumnTransformer selects columns by name,
                    # so a one-row DataFrame with matching column names is
                    # required (a plain list/array would be silently
                    # mis-mapped).
                    features = pd.DataFrame([form.as_feature_dict()])
                    prediction = int(model.predict(features)[0])
                    probability = float(model.predict_proba(features)[0][1])
                    result = {
                        "prediction": prediction,
                        "probability": round(probability * 100, 1),
                        "risk_label": "Higher risk of heart disease"
                        if prediction == 1
                        else "Lower risk of heart disease",
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Prediction failed")
                    error_message = f"Something went wrong while scoring your input: {exc}"
        else:
            error_message = "Please correct the errors below and resubmit."
    else:
        form = HeartDiseasePredictionForm()

    metrics = _get_metrics()
    context = {
        "form": form,
        "result": result,
        "error_message": error_message,
        "best_model": metrics.get("best_model"),
        "model_results": metrics.get("results", {}),
    }
    return render(request, "predictor/predict.html", context)


def about(request):
    metrics = _get_metrics()
    context = {
        "best_model": metrics.get("best_model"),
        "model_results": metrics.get("results", {}),
    }
    return render(request, "predictor/about.html", context)
