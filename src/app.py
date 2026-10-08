import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from alerts import count_violations, evaluate_rules
from database import db, init_db, list_alerts, save_prediction, window_stats

from detector import MaskDetector

app = FastAPI(title="Safety Vision API", version="0.1.0")
detector = MaskDetector()
init_db()

ALLOWED_TYPES = {"image/jpeg", "image/png"}

from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok", **detector.device_info}


@app.post("/predict")
def predict(file: UploadFile = File(...), location: str = Form("default")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG or PNG images are supported")

    data = file.file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the image")

    start = time.perf_counter()
    detections = detector.predict(image)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    num_violations = count_violations(detections)
    with db() as conn:
        prediction_id = save_prediction(conn, location, file.filename,
                                        detections, num_violations, latency_ms)
        alerts = evaluate_rules(conn, location, prediction_id, num_violations)

    return {
        "filename": file.filename,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "num_detections": len(detections),
        "counts": counts,
        "latency_ms": latency_ms,
        "model_speed_ms": detector.last_speed,
        "detections": detections,
        "prediction_id": prediction_id,
        "location": location,
        "num_violations": num_violations,
        "alerts": alerts,
    }

@app.get("/alerts")
def get_alerts(limit: int = Query(20, ge=1, le=200)):
    with db() as conn:
        return list_alerts(conn, limit)


@app.get("/stats")
def get_stats(location: str = "default", minutes: int = Query(60, ge=1, le=1440)):
    with db() as conn:
        s = window_stats(conn, location, minutes)
    s["violation_rate"] = round(s["violations"] / s["faces"], 3) if s["faces"] else 0.0
    return {"location": location, "minutes": minutes, **s}