import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--data", default="data/yolo/data.yaml")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    model = YOLO(args.model)
    model.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        seed=42,
        name=args.name,
        data=str(ROOT / args.data),
        project=str(ROOT / "runs"),
    )


if __name__ == "__main__":
    main()