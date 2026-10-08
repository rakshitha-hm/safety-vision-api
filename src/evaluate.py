from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "data/yolo/data.yaml")
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
MODELS = {
    "baseline": ROOT / "runs/baseline/weights/best.pt",
    "final_yolo11s": ROOT / "runs/exp_yolo11s/weights/best.pt",
}


def main():
    rows = []
    for name, weights in MODELS.items():
        m = YOLO(weights).val(
            data=DATA, split="test", imgsz=640,
            project=str(ROOT / "runs"), name=f"test_{name}", exist_ok=True,
        )
        for i, cls in enumerate(CLASSES):
            p, r, ap50, ap = m.box.class_result(i)
            rows.append({"model": name, "class": cls, "precision": round(p, 3),
                         "recall": round(r, 3), "mAP50": round(ap50, 3),
                         "mAP50-95": round(ap, 3)})
        rows.append({"model": name, "class": "all", "precision": round(m.box.mp, 3),
                     "recall": round(m.box.mr, 3), "mAP50": round(m.box.map50, 3),
                     "mAP50-95": round(m.box.map, 3)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "test_metrics.csv", index=False)


if __name__ == "__main__":
    main()