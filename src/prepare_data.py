import os
import xml.etree.ElementTree as ET

ANN_DIR = "data/annotations"
IMG_DIR = "data/images"
OUT_DIR = "data/yolo_labels"

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    filename = root.find("filename").text
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bb = obj.find("bndbox")
        x1 = int(float(bb.find("xmin").text))
        y1 = int(float(bb.find("ymin").text))
        x2 = int(float(bb.find("xmax").text))
        y2 = int(float(bb.find("ymax").text))
        boxes.append((label, x1, y1, x2, y2))
    return filename, img_w, img_h, boxes

def clean_box(x1, y1, x2, y2, img_w, img_h):
    # clip coordinates so the box stays inside the image
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))
    # drop boxes that are too small or broken after clipping
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2

def voc_to_yolo(x1, y1, x2, y2, img_w, img_h):
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h

def convert_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {"images": 0, "boxes_kept": 0, "boxes_clipped": 0,
             "boxes_dropped": 0, "empty_images": 0, "missing_images": 0}

    for xml_file in sorted(os.listdir(ANN_DIR)):
        filename, img_w, img_h, boxes = parse_xml(os.path.join(ANN_DIR, xml_file))

        if not os.path.exists(os.path.join(IMG_DIR, filename)):
            stats["missing_images"] += 1
            continue

        lines = []
        for label, x1, y1, x2, y2 in boxes:
            cleaned = clean_box(x1, y1, x2, y2, img_w, img_h)
            if cleaned is None:
                stats["boxes_dropped"] += 1
                continue
            if cleaned != (x1, y1, x2, y2):
                stats["boxes_clipped"] += 1
            xc, yc, w, h = voc_to_yolo(*cleaned, img_w, img_h)
            lines.append(f"{CLASS_TO_ID[label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lines:
            stats["empty_images"] += 1

        stem = os.path.splitext(filename)[0]
        with open(os.path.join(OUT_DIR, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

        stats["images"] += 1
        stats["boxes_kept"] += len(lines)

    print(stats)


if __name__ == "__main__":
    convert_all()