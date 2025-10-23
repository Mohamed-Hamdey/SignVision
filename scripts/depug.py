import os

train_dir = "E:/Downloads/sign_language/SignVision/data_processed/crops/train"

print("📊 Training Data Distribution:")
print("=" * 40)
for class_folder in sorted(os.listdir(train_dir)):
    class_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        print(f"{class_folder}: {count} images")