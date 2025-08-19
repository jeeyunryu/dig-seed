import os
from PIL import Image
import matplotlib.pyplot as plt

def get_aspect_ratios(image_dir):
    ratios = []
    sizes = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            path = os.path.join(image_dir, file)
            try:
                img = Image.open(path)
                w, h = img.size
                ratios.append(w / h)
                sizes.append((w, h))
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return ratios, sizes

# 학습/테스트 이미지 폴더 경로
train_dir = "dataset/Reann_MPSC/jyryu/image/train"
test_dir = "dataset/Reann_MPSC/jyryu/image/test"

train_ratios, train_sizes = get_aspect_ratios(train_dir)
test_ratios, test_sizes = get_aspect_ratios(test_dir)

# 종횡비 히스토그램 시각화
plt.figure(figsize=(10, 6))
plt.hist(train_ratios, bins=30, alpha=0.6, label="Train")
plt.hist(test_ratios, bins=30, alpha=0.6, label="Test")
plt.axvline(x=1.0, color='red', linestyle='--', label="Square (1:1)")
plt.xlabel("Aspect Ratio (width / height)")
plt.ylabel("Number of Images")
plt.legend()
plt.title("Aspect Ratio Distribution (Train vs Test)")
# plt.show()
plt.savefig("aspect_ratio_distribution.png", dpi=300, bbox_inches="tight")

# 종횡비 요약 통계
import numpy as np
print("Train Aspect Ratio - mean:", np.mean(train_ratios), "median:", np.median(train_ratios))
print("Test Aspect Ratio  - mean:", np.mean(test_ratios), "median:", np.median(test_ratios))
