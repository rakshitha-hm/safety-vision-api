from pathlib import Path

YOLO_DIR = Path("data/yolo")
RARE_CLASS = 2
FACTOR = 4


def class_ids(label_path):
    lines = label_path.read_text().splitlines()
    return [int(line.split()[0]) for line in lines if line.strip()]


def main():
    img_dir = YOLO_DIR / "images" / "train"
    lbl_dir = YOLO_DIR / "labels" / "train"

    lines = []
    counts = {0: 0, 1: 0, 2: 0}
    n_rare_images = 0

    for img in sorted(img_dir.glob("*.png")):
        ids = class_ids(lbl_dir / f"{img.stem}.txt")
        repeat = FACTOR if RARE_CLASS in ids else 1
        if repeat > 1:
            n_rare_images += 1
        lines += [img.resolve().as_posix()] * repeat
        for c in ids:
            counts[c] += repeat

    (YOLO_DIR / "train_oversampled.txt").write_text("\n".join(lines))

    base = (YOLO_DIR / "data.yaml").read_text()
    new = base.replace("train: images/train", "train: train_oversampled.txt")
    (YOLO_DIR / "data_oversampled.yaml").write_text(new)

    total = sum(counts.values())
    print(f"Images with rare class: {n_rare_images}, repeated x{FACTOR}")
    print(f"Train list length: {len(lines)}")
    for c, n in counts.items():
        print(f"class {c}: {n} boxes ({100 * n / total:.1f}%)")


if __name__ == "__main__":
    main()