from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs/exp_yolo11s/weights/best.pt"
IMG_DIR = ROOT / "data/yolo/images/test"
LBL_DIR = ROOT / "data/yolo/labels/test"
OUT_DIR = ROOT / "results/errors"
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
SHORT = ["mask", "no_mask", "incorrect"]
CONF = 0.25
IOU_MATCH = 0.5


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def load_gt(lbl_path, img_w, img_h):
    boxes = []
    for line in lbl_path.read_text().splitlines():
        if not line.strip():
            continue
        c, xc, yc, w, h = line.split()
        xc, w = float(xc) * img_w, float(w) * img_w
        yc, h = float(yc) * img_h, float(h) * img_h
        boxes.append((int(c), [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]))
    return boxes

def match(gts, preds):
    rows, used = [], set()
    for gt_cls, gt_box in gts:
        best_j, best_iou = None, IOU_MATCH
        for j, (p_cls, p_box, p_conf) in enumerate(preds):
            if j in used:
                continue
            v = iou(gt_box, p_box)
            if v >= best_iou:
                best_j, best_iou = j, v
        width = round(gt_box[2] - gt_box[0], 1)
        if best_j is None:
            rows.append({"gt": CLASSES[gt_cls], "pred": None, "outcome": "missed",
                         "iou": 0.0, "conf": None, "width": width})
        else:
            used.add(best_j)
            p_cls, _, p_conf = preds[best_j]
            outcome = "correct" if p_cls == gt_cls else "wrong_class"
            rows.append({"gt": CLASSES[gt_cls], "pred": CLASSES[p_cls], "outcome": outcome,
                         "iou": round(best_iou, 2), "conf": round(p_conf, 2), "width": width})
    for j, (p_cls, p_box, p_conf) in enumerate(preds):
        if j not in used:
            rows.append({"gt": None, "pred": CLASSES[p_cls], "outcome": "false_alarm",
                         "iou": 0.0, "conf": round(p_conf, 2),
                         "width": round(p_box[2] - p_box[0], 1)})
    return rows

def draw(img, gts, preds):
    for c, b in gts:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    for c, b, s in preds:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
        cv2.putText(img, f"{SHORT[c]} {s:.2f}", (int(b[0]), int(b[1]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)
    all_rows = []

    for img_path in sorted(IMG_DIR.glob("*.png")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gts = load_gt(LBL_DIR / f"{img_path.stem}.txt", w, h)

        r = model.predict(img, conf=CONF, imgsz=640, verbose=False)[0]
        preds = [(int(c), b.tolist(), float(s)) for b, c, s in zip(
            r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]

        rows = match(gts, preds)
        for row in rows:
            row["image"] = img_path.name
        all_rows += rows

        if any(row["outcome"] != "correct" for row in rows):
            draw(img, gts, preds)
            cv2.imwrite(str(OUT_DIR / img_path.name), img)

    df = pd.DataFrame(all_rows)
    df.to_csv(ROOT / "results" / "error_analysis.csv", index=False)
    print(pd.crosstab(df["gt"].fillna("none"), df["outcome"]))
    print("\nMedian face width (px) by outcome:")
    print(df.groupby("outcome")["width"].median())


if __name__ == "__main__":
    main()