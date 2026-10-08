import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
LOCATIONS = ["ward_1", "ward_2", "icu", "default"]
COLORS = {  # RGB, because Streamlit displays RGB images
    "with_mask": (0, 200, 0),
    "without_mask": (220, 0, 0),
    "mask_weared_incorrect": (255, 140, 0),
}

st.set_page_config(page_title="Safety Vision", layout="wide")
st.title("Safety Vision: mask compliance monitor")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: #1A1F2B;
    border: 1px solid #2A3142;
    border-radius: 12px;
    padding: 14px 18px;
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    margin: 3px;
    border-radius: 999px;
    font-size: 0.85rem;
    color: white;
}
.ok   { background: #1f8f4e; }
.bad  { background: #c0392b; }
.warn { background: #d68910; }
</style>
""", unsafe_allow_html=True)

BADGE_CLASS = {"with_mask": "ok", "without_mask": "bad", "mask_weared_incorrect": "warn"}

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.caption(f"API status: {health['status']} | device: {health.get('inference_device', 'unknown')}")
except requests.ConnectionError:
    st.error("API is not running. Start it with: python -m uvicorn app:app --app-dir src --reload")
    st.stop()


def draw_boxes(image_rgb, detections, label_mode="Violations only"):
    img = image_rgb.copy()
    for d in detections:
        b = d["box"]
        color = COLORS.get(d["class"], (255, 255, 255))
        is_violation = d["class"] != "with_mask"
        p1 = (int(b["x1"]), int(b["y1"]))
        p2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(img, p1, p2, color, 3 if is_violation else 1)
        if label_mode == "All" or (label_mode == "Violations only" and is_violation):
            cv2.putText(img, f'{d["class"]} {d["confidence"]:.2f}', (p1[0], max(p1[1] - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img

tab_detect, tab_monitor = st.tabs(["Detect", "Monitoring"])

with tab_detect:
    location = st.selectbox("Location", LOCATIONS)
    label_mode = st.radio("Labels", ["Violations only", "All", "None"], horizontal=True)
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded is not None and st.button("Run detection"):
        data = uploaded.getvalue()
        r = requests.post(f"{API_URL}/predict",
                          files={"file": (uploaded.name, data, uploaded.type)},
                          data={"location": location}, timeout=30)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text[:200]}")
        else:
            body = r.json()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([2, 1])
            col1.image(draw_boxes(image, body["detections"]))
            col2.metric("Faces detected", body["num_detections"])
            col2.metric("Violations", body["num_violations"])
            col2.metric("Latency (ms)", body["latency_ms"])
            badges = "".join(
                f'<span class="badge {BADGE_CLASS.get(c, "ok")}">{c}: {n}</span>'
                for c, n in body["counts"].items()
            )
            col2.markdown(badges, unsafe_allow_html=True)

            if body["num_violations"] == 0:
                col2.success("Compliant: no violations")
            else:
                col2.warning(f"{body['num_violations']} violation(s) detected")
            for alert in body["alerts"]:
                st.error(f"ALERT {alert['rule']}: {alert['message']}")

with tab_monitor:
    minutes = st.slider("Time window (minutes)", 5, 1440, 60)
    st.button("Refresh")  # any click reruns the script, which reloads the data

    cols = st.columns(len(LOCATIONS))
    for col, loc in zip(cols, LOCATIONS):
        s = requests.get(f"{API_URL}/stats", params={"location": loc, "minutes": minutes}, timeout=10).json()
        col.metric(loc, f"{s['violation_rate']:.0%}",
                   help=f"{s['violations']} of {s['faces']} faces in {s['images']} images")

    st.subheader("Recent alerts")
    alerts = requests.get(f"{API_URL}/alerts", params={"limit": 50}, timeout=10).json()
    if alerts:
        df = pd.DataFrame(alerts)[["created_at", "location", "rule", "message"]]
        st.dataframe(df, hide_index=True)
    else:
        st.info("No alerts yet")