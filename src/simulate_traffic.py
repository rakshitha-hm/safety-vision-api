import random
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8000"
LOCATIONS = ["ward_1", "ward_2", "icu"]

random.seed(42)
images = sorted((ROOT / "data/yolo/images/test").glob("*.png"))[:60]

for img in images:
    location = random.choice(LOCATIONS)
    with open(img, "rb") as f:
        r = requests.post(f"{URL}/predict",
                          files={"file": (img.name, f, "image/png")},
                          data={"location": location})
    r.raise_for_status()
    for alert in r.json()["alerts"]:
        print(f"ALERT [{location}] {alert['rule']}: {alert['message']}")

for location in LOCATIONS:
    print(requests.get(f"{URL}/stats", params={"location": location}).json())