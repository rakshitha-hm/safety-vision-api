import requests

URL = "http://127.0.0.1:8000/predict"
IMAGE = "data/images/maksssksksss0.png"

for i in range(20):
    with open(IMAGE, "rb") as f:
        r = requests.post(URL, files={"file": ("test.png", f, "image/png")})
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Body:", r.text[:500])
        break
    body = r.json()
    print(i, body["latency_ms"], body["model_speed_ms"])