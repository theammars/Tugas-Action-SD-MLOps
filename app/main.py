from fastapi import FastAPI
from .schema import StrokeInput
import joblib, pandas as pd, os, json

app = FastAPI(title="Stroke Prediction API")
MODEL_PATH = os.getenv("MODEL_PATH", "model/artifacts/latest/stroke_pipeline.joblib")
THRESHOLD = 0.5


manifest_path = os.path.join(os.path.dirname(MODEL_PATH), "manifest.json")
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        THRESHOLD = float(json.load(f).get("threshold", THRESHOLD))

_model = None

@app.on_event("startup")
def _load():
    global _model
    _model = joblib.load(MODEL_PATH)

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.post("/predict")
def predict(inp: StrokeInput):
    df = pd.DataFrame([inp.model_dump()])
    proba = float(_model.predict_proba(df)[0][1])
    pred = int(proba >= THRESHOLD)
    return {"prediction": pred, "probability": proba, "threshold": THRESHOLD}
