from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs"
DATA = str(ROOT / "data/yolo/data.yaml")
RUNS = ["baseline", "exp_img960", "exp_yolo11s", "exp_oversample", "exp_yolo11s_oversample"]
RARE_CLASS = 2


def main():
    rows = []
    for name in RUNS:
        weights = RUN_DIR / name / "weights" / "best.pt"
        if not weights.exists():
            print("Skipping", name, "- not found:", weights)
            continue
        args = yaml.safe_load((RUN_DIR / name / "args.yaml").read_text())
        model = YOLO(weights)
        m = model.val(data=DATA, split="val", imgsz=args["imgsz"],
                      plots=False, verbose=False)
        p, r, ap50, ap = m.box.class_result(RARE_CLASS)
        rows.append({
            "run": name,
            "imgsz": args["imgsz"],
            "mAP50": round(m.box.map50, 3),
            "mAP50-95": round(m.box.map, 3),
            "recall": round(m.box.mr, 3),
            "rare_recall": round(r, 3),
            "rare_mAP50": round(ap50, 3),
            "ms_per_img": round(m.speed["inference"], 1),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "experiments.csv", index=False)


if __name__ == "__main__":
    main()