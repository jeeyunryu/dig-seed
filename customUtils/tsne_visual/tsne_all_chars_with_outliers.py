import numpy as np
import matplotlib.pyplot as plt
import string
import seaborn as sns
from openai import OpenAI
from sklearn.manifold import TSNE
from collections import defaultdict
from scipy.stats import zscore
import re
import os
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.cm as cm

plot_save_root = "./customUtils/tsne_visual/feature_plots_w_outlier"
os.makedirs(plot_save_root, exist_ok=True)


classes = list(string.printable[:-6])
classes.append('EOS')
classes.append('PADDING')
classes.append('UNKNOWN')

idx_to_class = dict(zip(range(len(classes)), classes))
class_to_idx = dict(zip(classes, range(len(classes))))

# 모델에서 뽑은 임베딩 로딩
# embeds_rslt = np.load('embeddings_char.npy')
# labels_rslt = np.load('labels_char.npy')

# embeds_rslt = np.load('./npy_files/embeddings_char_embed_feat.npy')
# labels_rslt = np.load('./npy_files/labels_char_embed_feat.npy')

embeds_rslt = np.load('/home/jyryu/workspace/DiG/npy_files/embeddings_base_feat.npy')
labels_rslt = np.load('/home/jyryu/workspace/DiG/npy_files/labels_base_feat.npy')
imgkeys = np.load('/home/jyryu/workspace/DiG/npy_files/imgkeys_base_feat.npy')
char_indices = np.load('/home/jyryu/workspace/DiG/npy_files/char_idx_base_feat.npy')

imgpaths = []


print('finished loading embeddings')

img_root = '/home/jyryu/workspace/DiG/datasets/Reann_MPSC/jyryu/patchLevelImage/test'

def extract_sort_key(filename):
    nums = re.findall(r'\d+', filename)
    return tuple(map(int, nums)) if nums else (0, 0)

image_files = sorted([
    f for f in os.listdir(img_root)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
], key=extract_sort_key)


for key in imgkeys:
    index = int(key.replace('image-', ''))
    if 0 < index < len(image_files) + 1: # 1 ~ len(image_files)
        filename = image_files[index-1]
        img_path = os.path.join(img_root, filename)
        imgpaths.append(img_path)
        
    else:
        print(f"[경고] 인덱스 {index}가 이미지 리스트 범위를 벗어남")

grouped_features = defaultdict(list)
outlier_dict = defaultdict(list)
grouped_paths = defaultdict(list)
grouped_char_idx = defaultdict(list)

reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
tsne_result = reducer.fit_transform(embeds_rslt)

# for embed, label, path, index in zip(embeds_rslt, labels_rslt, imgpaths, char_indices):
#     grouped_features[label].append(embed)
#     grouped_paths[label].append(path)
#     grouped_char_idx[label].append(index)
for embed, label, path, index in zip(tsne_result, labels_rslt, imgpaths, char_indices):
    grouped_features[label].append(embed)
    grouped_paths[label].append(path)
    grouped_char_idx[label].append(index)

for label, feats in grouped_features.items():
    # feats = np.stack(feats)
    # tsne_result = np.stack(tsne_result)
    z_scores = zscore(feats, axis=0)
    anomaly_scores = np.max(np.abs(z_scores), axis=1)
    mask = anomaly_scores > 3.0

    outlier_idx = np.where(mask)[0] 
    inlier_idx = np.where(~mask)[0]
    for idx in outlier_idx:
        # outlier_dict[label].append((grouped_paths[label][idx], grouped_char_idx[label][idx]))
        outlier_dict[label].append(grouped_paths[label][idx])

# for label, outlier_imgs in tqdm(outlier_dict.items(), desc='saving outliers'):
#     for i, (img, idx) in enumerate(outlier_imgs):
#         try:
#             image = Image.open(img).convert('RGB')
#             char = idx_to_class[label]
#             if char == '/':
#                 char = 'slash'
#             save_dir = f"/home/jyryu/workspace/DiG/customUtils/outliers/char_{char}"
#             os.makedirs(save_dir, exist_ok=True)
#             save_path = os.path.join(save_dir, f"char_{char}_outlier_{i}_idx_{idx}.png")
#             image.save(save_path)
#         except Exception as e:
#             print(f"[ERROR] char: '{char}'")#, image: '{img}' — {e}")

# plot every chars in one 
# reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
# tsne_result = reducer.fit_transform(embeds_rslt)

unique_labels = sorted(set(labels_rslt))
num_classes = len(unique_labels)
color_map = cm.get_cmap('gist_ncar', num_classes)
label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

outlier_indices = set()
for label, outlier_paths in outlier_dict.items():
    for path in outlier_paths:
        global_idx = imgpaths.index(path)        
        outlier_indices.add(global_idx)

plt.figure(figsize=(12, 8))

for i, (point, label) in enumerate(zip(tsne_result, labels_rslt)):
    if i in outlier_indices:
        plt.scatter(point[0], point[1], color=label_to_color[label], marker='x', s=50) #, label='outlier' if i == list(outlier_indices)[0] else "")
    else:
        plt.scatter(point[0], point[1], color=label_to_color[label], s=30, alpha=0.7) #, label=label if i == all_labels.index(label) else "")


plt.title("t-SNE Visualization with Outliers")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("./customUtils/tsne_visual/tsne/base_feat_with_outliers.png", dpi=300)

