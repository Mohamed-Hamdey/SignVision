# scripts/extract_keypoints.py
import os, glob, numpy as np, cv2
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--images", default="E:/Downloads/sign_language/SignVision/Sign Language Detection - data/images/train")
parser.add_argument("--labels", default="E:/Downloads/sign_language/SignVision/Sign Language Detection - data/labels/train")
parser.add_argument("--out_dir", default="data_processed/keypoints/train")
parser.add_argument("--classes", default="classes.txt")
args = parser.parse_args()

# optional map id->name
class_names = None
if os.path.exists(args.classes):
    with open(args.classes) as f:
        class_names = [l.strip() for l in f if l.strip()]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

os.makedirs(args.out_dir, exist_ok=True)
csv_path = os.path.join(args.out_dir, "labels.csv")
csv_file = open(csv_path, "w", newline="")
csvw = csv.writer(csv_file)
csvw.writerow(["npy_path","label"])

def read_yolo_label(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts)
        boxes.append((int(cls), xc, yc, bw, bh))
    return boxes

img_paths = []
for ext in ("*.jpg","*.png","*.jpeg"):
    img_paths += glob.glob(os.path.join(args.images, ext))

idx = 0
for p in tqdm(img_paths):
    fname = Path(p).name
    lab = os.path.join(args.labels, Path(fname).with_suffix('.txt').name)
    if not os.path.exists(lab):
        continue
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    boxes = read_yolo_label(lab)
    for b in boxes:
        cls, xc, yc, bw, bh = b
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # run Mediapipe on the crop (convert to RGB)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        lm = res.multi_hand_landmarks[0]
        coords = np.array([[l.x, l.y, l.z] for l in lm.landmark])  # 21x3
        # normalize relative to wrist and scale by max abs x,y
        origin = coords[0].copy()
        coords[:, :2] -= origin[:2]
        denom = max(1e-6, np.max(np.abs(coords[:, :2])))
        coords[:, :2] /= denom
        flat = coords.flatten()
        out_npy = os.path.join(args.out_dir, f"{Path(fname).stem}_{idx:05d}.npy")
        np.save(out_npy, flat)
        label_name = str(int(cls))
        if class_names is not None:
            label_name = class_names[int(cls)]
        csvw.writerow([out_npy, label_name])
        idx += 1

csv_file.close()
print("Done. saved", idx, "keypoint samples to", args.out_dir)
