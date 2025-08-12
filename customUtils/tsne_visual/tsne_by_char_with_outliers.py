import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from collections import defaultdict
import re
from tqdm import tqdm
from sklearn.manifold import TSNE
import string

classes = list(string.printable[:-6])
classes.append('EOS')
classes.append('PADDING')
classes.append('UNKNOWN')

idx_to_class = dict(zip(range(len(classes)), classes))
class_to_idx = dict(zip(classes, range(len(classes))))

# 저장 디렉토리 설정
plot_save_root = "./customUtils/tsne_visual/feature_plots_inter_char"
os.makedirs(plot_save_root, exist_ok=True)

embeds_rslt = np.load('/home/jyryu/workspace/DiG/npy_files/embeddings_base_feat.npy')
labels_rslt = np.load('/home/jyryu/workspace/DiG/npy_files/labels_base_feat.npy')
imgkeys_rslt = np.load('/home/jyryu/workspace/DiG/npy_files/imgkeys_base_feat.npy')

print('finished loading embeddings')

img_root = '/home/jyryu/workspace/DiG/datasets/Reann_MPSC/jyryu/patchLevelImage/test'

def extract_sort_key(filename):
    nums = re.findall(r'\d+', filename)
    return tuple(map(int, nums)) if nums else (0, 0)

image_files = sorted([
    f for f in os.listdir(img_root)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
], key=extract_sort_key)

imgpaths =[]
for key in imgkeys_rslt:
    index = int(key.replace('image-', ''))
    if 0 <= index < len(image_files) +1:
        filename = image_files[index-1]
        img_path = os.path.join(img_root, filename)
        imgpaths.append(img_path)
        
    else:
        print(f"[경고] 인덱스 {index}가 이미지 리스트 범위를 벗어남")

reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
tsne_result = reducer.fit_transform(embeds_rslt)

# 시각화 및 이상치 저장
outlier_dict = {}

grouped_features = defaultdict(list)
grouped_paths = defaultdict(list)
# # calculated z-score within same chars.
# for embed, label, path in zip(embeds_rslt, labels_rslt, imgpaths):
#     grouped_features[label].append(embed)
#     grouped_paths[label].append(path)
for embed, label, path in zip(tsne_result, labels_rslt, imgpaths):
    grouped_features[label].append(embed)
    grouped_paths[label].append(path)

for label, feats in tqdm(grouped_features.items(), desc='Processing labels'):
    if len(feats) < 2:
        print(f"Skipping label '{label}' because it has only {len(feats)} sample(s).")
        continue 
    
    feats = np.stack(feats)  # (N, D)
    z_scores = zscore(feats, axis=0)
    anomaly_scores = np.max(np.abs(z_scores), axis=1)
    mask = anomaly_scores > 3.0

    outlier_idx = np.where(mask)[0]
    inlier_idx = np.where(~mask)[0]

    char = idx_to_class[label]

    if char == '/':
        char = 'slash'

    # 이상치 경로 저장
    outlier_dict[label] = []
    for idx in outlier_idx:
        outlier_dict[label].append(grouped_paths[label][idx])

    # --- 2D 임베딩 (PCA or t-SNE 등) ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(feats)

    # --- 시각화 ---
    plt.figure(figsize=(6, 6))
    plt.scatter(feats_2d[inlier_idx, 0], feats_2d[inlier_idx, 1], c='blue', label='Inliers', alpha=0.6)
    plt.scatter(feats_2d[outlier_idx, 0], feats_2d[outlier_idx, 1], c='red', label='Outliers', alpha=0.8)
    plt.title(f"Feature Distribution for Label: {char}")
    plt.legend()
    plt.grid(True)

    # 저장
    save_path = os.path.join(plot_save_root, f"label_{char}_features.png")
    plt.savefig(save_path)
    plt.close()
    
