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
import random

def get_imgkeys(input_txt_path):
    
    imgkeys = []

    with open(input_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                
                match = re.match(r'(image-\d+)\s*\|\s*GT:\s*(.+?)\s*\|\s*Pred:\s*(.+)', line)
                image_id, _, _ = match.groups()
                
                if match:
                    imgkeys.append(image_id)
                    
            except Exception as e:
                print(f"[Warning] Failed to parse line: {line}")
                continue
    return imgkeys

classes = list(string.printable[:-6])
classes.append('EOS')
classes.append('PADDING')
classes.append('UNKNOWN')

idx_to_class = dict(zip(range(len(classes)), classes))
class_to_idx = dict(zip(classes, range(len(classes))))

embeds_rslt = np.load('npy_files_exif/embeddings_base_feat.npy')
labels_rslt = np.load('npy_files_exif/labels_base_feat.npy')
imgkeys = np.load('npy_files_exif/imgkeys_base_feat.npy')
char_indices = np.load('npy_files_exif/char_idx_base_feat.npy')


file_path = "output/mpsc/train/250809_1700/eval/wrong_predictions.txt" 

wrongp_keys = get_imgkeys(file_path)


imgpaths = []


print('finished loading embeddings')

img_root = '/home/jyryu/workspace/DiG/dataset/Reann_MPSC/jyryu/image/test'

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
grouped_imgkeys = defaultdict(list)

reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
tsne_result = reducer.fit_transform(embeds_rslt)


# for embed, label, path, index in zip(embeds_rslt, labels_rslt, imgpaths, char_indices):
#     grouped_features[label].append(embed)
#     grouped_paths[label].append(path)
#     grouped_char_idx[label].append(index)
for embed, label, path, index, keys in zip(tsne_result, labels_rslt, imgpaths, char_indices, imgkeys):
    grouped_features[label].append(embed)
    grouped_paths[label].append(path)
    grouped_char_idx[label].append(index)
    grouped_imgkeys[label].append(keys)

for label, feats in grouped_features.items():
    # tsne_result = reducer.fit_transform(feats)

    # tsne_result = np.stack(tsne_result)
    # z_scores = zscore(tsne_result, axis=0)
    z_scores = zscore(feats, axis=0)
    # anomaly_scores = np.max(np.abs(z_scores), axis=1)
    anomaly_scores = np.linalg.norm(z_scores, axis=1) # L2 norm
    # mask = anomaly_scores > 3.0
    mask = anomaly_scores > 2.466

    outlier_idx = np.where(mask)[0] 
    inlier_idx = np.where(~mask)[0]
    for idx in outlier_idx:
        key = grouped_imgkeys[label][idx]
        
        if key not in wrongp_keys:
            
            continue
        outlier_dict[label].append((grouped_paths[label][idx], grouped_char_idx[label][idx], grouped_imgkeys[label][idx]))
cnt = 0
# landscapes = set()
# portraits = set()
# squares = set()
# # land_keys = []
# port_keys  = []

for label, outlier_imgs in tqdm(outlier_dict.items(), desc='saving outliers'):

    # if len(outlier_imgs) >= 2:
    #     sampled_imgs = random.sample(outlier_imgs, 2)
    # else:
    #     sampled_imgs = outlier_imgs

    # for i, (img, idx, key) in enumerate(sampled_imgs):
    for i, (img, idx, key) in enumerate(outlier_imgs):
        cnt += 1

        try:
            image = Image.open(img).convert('RGB')

            # width, height = image.size
            # aspect_ratio = width / height

            # # 가로로 긴 이미지인지 세로로 긴 이미지인지 구분
            # if aspect_ratio > 1:
            #     # orientation = 'landscape'  # 가로로 긴
            #     # landscapes+=1
            #     landscapes.add(img)
            #     # land_keys.append(key)
            # elif aspect_ratio < 1:
            #     # orientation = 'portrait'   # 세로로 긴
            #     # portraits+=1
            #     portraits.add(img)
            #     port_keys.append(key)
            # else:
            #     # orientation = 'square'     # 정사각형
            #     # squares+=1
            #     squares.add(img)

            char = idx_to_class[label]
            if char == '/':
                char = 'slash'
            # save_dir = f"/home/jyryu/workspace/DiG/customUtils/tsne_outs_random_wrongP/char_{char}"
            save_dir = f"/home/jyryu/workspace/DiG/customUtils/tsne_outs_random_wrongP"
            os.makedirs(save_dir, exist_ok=True)
            # if f"char_{char}_outlier_{i}_idx_{idx}.png" == 'char_i_outlier_1_idx_0.png' :
            #     print(img)
                # import pdb;pdb.set_trace()
            save_path = os.path.join(save_dir, f"{cnt}_char_{char}_idx_{idx}.png")
            image.save(save_path)
        except Exception as e:
            print(f"[ERROR] char: '{char}'")

# out_imgkeys = list(set(port_keys))
# with open('port_keys.txt', 'w') as f:
#     for item in out_imgkeys:
#         pass
        # f.write(f"{item}\n")
# print(len(landscapes))
# print(len(portraits))
# print(len(squares))

print(cnt)
# # plot every chars in one 
# reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
# tsne_result = reducer.fit_transform(embeds_rslt)

# unique_labels = sorted(set(labels_rslt))
# num_classes = len(unique_labels)
# color_map = cm.get_cmap('gist_ncar', num_classes)
# label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

# outlier_indices = set()
# for label, outlier_paths in outlier_dict.items():
#     for path in outlier_paths:
#         global_idx = imgpaths.index(path)        
#         outlier_indices.add(global_idx)

# plt.figure(figsize=(12, 8))

# for i, (point, label) in enumerate(zip(tsne_result, labels_rslt)):
#     if i in outlier_indices:
#         plt.scatter(point[0], point[1], color='red', marker='x', s=50) #, label='outlier' if i == list(outlier_indices)[0] else "")
#     else:
#         plt.scatter(point[0], point[1], color=label_to_color[label], s=30, alpha=0.7) #, label=label if i == all_labels.index(label) else "")


# plt.title("t-SNE Visualization with Outliers")
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./customUtils/tsne_visual/tsne/base_feat_with_outliers.png", dpi=300)