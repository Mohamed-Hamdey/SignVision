# scripts/visualize_labels.py
import os, cv2
import glob
import matplotlib.pyplot as plt

IMG_DIR = "E:/Downloads/sign_language/SignVision/Sign Language Detection - data/images/train"
LBL_DIR = "E:/Downloads/sign_language/SignVision/Sign Language Detection - data/labels/train"
EXTS = ("*.jpg", "*.png", "*.jpeg")

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

sample_images = []
for ext in EXTS:
    sample_images += glob.glob(os.path.join(IMG_DIR, ext))

sample_images = sample_images[:30]  # limit to 30 images to inspect

for p in sample_images:
    fname = os.path.basename(p)
    label_path = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + ".txt")
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    boxes = []
    if os.path.exists(label_path):
        boxes = read_yolo_label(label_path)
    # draw boxes
    for cls, x_c, y_c, bw, bh in boxes:
        x1 = int((x_c - bw/2) * w)
        y1 = int((y_c - bh/2) * h)
        x2 = int((x_c + bw/2) * w)
        y2 = int((y_c + bh/2) * h)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, str(cls), (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    # show
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(fname)
    plt.show()
