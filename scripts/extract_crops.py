import os, glob, cv2
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--images", default="E:/Downloads/sign_language/SignVision/Sign Language Detection - data/images/train")
parser.add_argument("--labels", default="E:/Downloads/sign_language/SignVision/Sign Language Detection - data/labels/train")
parser.add_argument("--out", default="data_processed/crops/train")
parser.add_argument("--classes", default="classes.txt")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

def read_yolo_label(path):
    boxes = []
    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_c, y_c, w, h = map(float, parts)
                    boxes.append((int(cls), x_c, y_c, w, h))
    except Exception as e:
        print("Error reading:", path, e)
    return boxes

img_paths = []
for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG"):
    img_paths += glob.glob(os.path.join(args.images, ext))

print(f"Found {len(img_paths)} images in {args.images}")
count = 0

for p in img_paths:
    fname = Path(p).name
    lab = os.path.join(args.labels, Path(fname).stem + ".txt")
    if not os.path.exists(lab):
        print("No label for:", fname)
        continue

    img = cv2.imread(p)
    if img is None:
        print("Image failed to load:", fname)
        continue

    h, w = img.shape[:2]
    boxes = read_yolo_label(lab)
    print(f"{fname}: {len(boxes)} boxes")

    for i, (cls, xc, yc, bw, bh) in enumerate(boxes):
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print("Empty crop:", fname, x1, y1, x2, y2)
            continue
        out_dir = os.path.join(args.out, str(cls))
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{Path(fname).stem}_{i}.jpg"), crop)
        count += 1

print(f"✅ Saved {count} crops into {args.out}")
