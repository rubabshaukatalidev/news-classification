"""FastAPI backend for the Urdu News Topic Classifier."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FRONTEND_DIR = ROOT / "frontend"

CATEGORIES = {
    "politics": {"ur": "سیاست", "en": "Politics"},
    "health": {"ur": "صحت", "en": "Health"},
    "education": {"ur": "تعلیم", "en": "Education"},
    "economy": {"ur": "معیشت", "en": "Economy"},
    "sports": {"ur": "کھیل", "en": "Sports"},
    "technology": {"ur": "ٹیکنالوجی", "en": "Technology"},
}

MAX_CHARS = 5000
MAX_BATCH = 100


def _ensure_model():
    """Load the model; if missing, train one from the bundled seed data."""
    if not MODEL_PATH.exists():
        subprocess.run([sys.executable, "-m", "training.train"], cwd=ROOT, check=True)
    return joblib.load(MODEL_PATH)


model = _ensure_model()

app = FastAPI(
    title="Urdu News Topic Classifier",
    version="1.0.0",
    description="Classifies Urdu news text into topic categories. Predictions are "
    "automated suggestions and should be reviewed by a human editor.",
)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH)


def _predict(texts: List[str]) -> List[dict]:
    probs = model.predict_proba(texts)
    classes = list(model.classes_)
    out = []
    for row in probs:
        scores = sorted(
            (
                {
                    "label": c,
                    "label_ur": CATEGORIES.get(c, {}).get("ur", c),
                    "label_en": CATEGORIES.get(c, {}).get("en", c.title()),
                    "score": round(float(p), 4),
                }
                for c, p in zip(classes, row)
            ),
            key=lambda d: d["score"],
            reverse=True,
        )
        top = scores[0]
        out.append(
            {
                "label": top["label"],
                "label_ur": top["label_ur"],
                "label_en": top["label_en"],
                "confidence": top["score"],
                "scores": scores,
            }
        )
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": list(model.classes_)}


@app.get("/api/categories")
def categories():
    return [{"label": c, **CATEGORIES.get(c, {"ur": c, "en": c.title()})} for c in model.classes_]


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "metrics not available")
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {k: data[k] for k in ("n_samples", "classes", "cv_folds", "cv_macro_f1", "data_source")}


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "text must not be blank")
    return _predict([text])[0]


@app.post("/api/classify/batch")
def classify_batch(req: BatchRequest):
    texts = [t.strip() for t in req.texts]
    if any(not t or len(t) > MAX_CHARS for t in texts):
        raise HTTPException(422, f"each text must be 1-{MAX_CHARS} characters")
    return {"results": [{"text": t, **r} for t, r in zip(texts, _predict(texts))]}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
