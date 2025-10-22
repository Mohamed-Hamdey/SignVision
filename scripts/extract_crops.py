# scripts/extract_crops.py
import os, glob, cv2
from pathlib import Path
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--images", default="data/images/train")
parser.add_argument("--labels", default="data/labels/train")
parser.add_argument("--out", default="data_processed/crops/train")
parser.add_argument("--classes", default="classes.txt")  # optional mapping file
args = parser.parse_args()

# load class names if provided
class_names = None
if os.path.exists(args.classes):
    with open(args.classes) as f:
        class_names = [l.strip() for l in f if l.strip()]

os.makedirs(args.out, exist_ok=True)

def read_yolo_label(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, x_c, y_c, w, h = map(float, parts)
        boxes.append((int(cls), x_c, y_c, w, h))
    return boxes

img_paths = []
for ext in ("*.jpg","*.png","*.jpeg"):
    img_paths += glob.glob(os.path.join(args.images, ext))

count = 0
for p in img_paths:
    fname = Path(p).name
    lab = os.path.join(args.labels, Path(fname).with_suffix('.txt').name)
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    if not os.path.exists(lab):
        continue
    boxes = read_yolo_label(lab)
    for i,(cls, xc, yc, bw, bh) in enumerate(boxes):
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        # clamp:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w-1, x2)
        y2 = min(h-1, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cls_name = str(cls) if class_names is None else class_names[cls]
        out_dir = os.path.join(args.out, cls_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{Path(fname).stem}_{i}.jpg")
        cv2.imwrite(out_path, crop)
        count += 1

print(f"Saved {count} crops into {args.out}")
