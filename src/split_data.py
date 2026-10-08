import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_DIR = Path("data/images")
LBL_DIR = Path("data/yolo_labels")
OUT_DIR = Path("data/yolo")
SEED = 42


def build_image_table():
    rows = []
    for lbl_file in sorted(LBL_DIR.glob("*.txt")):
        lines = lbl_file.read_text().splitlines()
        class_ids = [int(line.split()[0]) for line in lines if line.strip()]
        rows.append({
            "stem": lbl_file.stem,
            "n_boxes": len(class_ids),
            "n_with": class_ids.count(0),
            "n_without": class_ids.count(1),
            "n_incorrect": class_ids.count(2),
            "strata": max(class_ids) if class_ids else 0,
        })
    return pd.DataFrame(rows)


def split(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["strata"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["strata"], random_state=SEED)
    return {"train": train_df, "val": val_df, "test": test_df}

def copy_files(splits):
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for name, part in splits.items():
        img_out = OUT_DIR / "images" / name
        lbl_out = OUT_DIR / "labels" / name
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        for stem in part["stem"]:
            shutil.copy(IMG_DIR / f"{stem}.png", img_out)
            shutil.copy(LBL_DIR / f"{stem}.txt", lbl_out)

def report(splits):
    rows = []
    for name, part in splits.items():
        total = part["n_boxes"].sum()
        rows.append({
            "split": name,
            "images": len(part),
            "boxes": total,
            "with_%": round(100 * part["n_with"].sum() / total, 1),
            "without_%": round(100 * part["n_without"].sum() / total, 1),
            "incorrect_%": round(100 * part["n_incorrect"].sum() / total, 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def write_yaml():
    root = OUT_DIR.resolve().as_posix()
    text = f"""path: {root}
train: images/train
val: images/val
test: images/test
names:
  0: with_mask
  1: without_mask
  2: mask_weared_incorrect
"""
    (OUT_DIR / "data.yaml").write_text(text)
    print("Wrote", OUT_DIR / "data.yaml")


if __name__ == "__main__":
    df = build_image_table()
    splits = split(df)

    # leakage check: no image may appear in two splits
    train, val, test = (set(s["stem"]) for s in splits.values())
    assert not (train & val or train & test or val & test), "Leakage!"

    copy_files(splits)
    report(splits)
    write_yaml()