"""
Backend for the ECG prototype.
Run:  uvicorn api:app
Then open http://127.0.0.1:8000
"""
import json
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

app = FastAPI(title="ECG Beat Classifier (prototype)")

# CORS — allow the frontend to talk to the API from any origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
N_FEATURES = model.n_features_in_


class Beat(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"status": "ok", "expected_features": int(N_FEATURES)}


@app.post("/predict")
def predict(beat: Beat):
    if len(beat.features) != N_FEATURES:
        raise HTTPException(400, f"Expected {N_FEATURES} features, got {len(beat.features)}")
    proba = model.predict_proba([beat.features])[0]
    label = "Abnormal" if proba[1] >= 0.5 else "Normal"
    return {"class": label, "confidence": round(float(max(proba)), 4)}


@app.get("/metrics")
def metrics():
    with open("metrics.json") as f:
        return json.load(f)


@app.get("/sample")
def sample():
    """Real example beats from the test records, ready to paste into /predict."""
    with open("sample_beats.json") as f:
        return json.load(f)


@app.get("/ui")
def ui_redirect():
    """Redirect to the interactive dashboard."""
    return RedirectResponse("/static/index.html")


# Serve the frontend static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
