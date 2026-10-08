import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def png_bytes():
    img = np.full((240, 320, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_expected_fields():
    r = client.post("/predict",
                    files={"file": ("blank.png", png_bytes(), "image/png")},
                    data={"location": "test"})
    assert r.status_code == 200
    body = r.json()
    for key in ["detections", "num_detections", "num_violations", "alerts", "latency_ms", "prediction_id"]:
        assert key in body
    assert body["num_detections"] == len(body["detections"])


def test_rejects_wrong_file_type():
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image():
    r = client.post("/predict", files={"file": ("bad.png", b"not really an image", "image/png")})
    assert r.status_code == 400


def test_alerts_limit_is_validated():
    assert client.get("/alerts", params={"limit": 0}).status_code == 422