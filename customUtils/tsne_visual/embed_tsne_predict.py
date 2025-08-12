import numpy as np
import umap
import matplotlib.pyplot as plt
import string
import seaborn as sns
from openai import OpenAI
from sklearn.manifold import TSNE
import matplotlib.cm as cm


classes = list(string.printable[:-6])
classes.append('EOS')
classes.append('PADDING')
classes.append('UNKNOWN')

idx_to_class = dict(zip(range(len(classes)), classes))
class_to_idx = dict(zip(classes, range(len(classes))))

# 모델에서 뽑은 임베딩 로딩
# embeds_rslt = np.load('embeddings_char.npy')
# labels_rslt = np.load('labels_char.npy')

embeds_rslt = np.load('npy_files_exif/embeddings_base_feat.npy')
labels_rslt = np.load('npy_files_exif/labels_base_feat.npy')

# embeds_rslt = np.load('embeddings_base_feat.npy')
# labels_rslt = np.load('labels_base_feat.npy')

print('finished loading embeddings')

# # 임베딩 모델로 부터 당장 뽑을 것 준비
# sim_chars_str = '0OQD1lLuvUVijMWmnN5SsJT.,;:'
sim_chars_str = '0OQDC'
# sim_chars_str  = '1lL'
# sim_chars_str  = 'uv'
# sim_chars_str  = 'ij'
# sim_chars_str  = '5Ss'


# reducer = umap.UMAP(n_components=2, random_state=42)
reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
embeddings_2d = reducer.fit_transform(embeds_rslt)

unique_labels = np.unique(labels_rslt) # 정수 인덱스 라벨 가지고 있음
# sim_chars_list = []

# for char in sim_chars_str:
#     sim_chars_list.append(class_to_idx[char])

# unique_labels_rslt = []
# for label in unique_labels:
#     if label in sim_chars_list:
#         unique_labels_rslt.append(label)

# no_colors_rslt = len(unique_labels_rslt)
# colors = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 10) # 30개 까지 구분 가능함!!
num_classes = len(unique_labels)
color_map = cm.get_cmap('gist_ncar', num_classes)
label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

plt.figure(figsize=(12, 8))
plt.grid(True)
plt.xlim(-100, 100)   # 원하는 범위로 설정
plt.ylim(-100, 100)

for i, label in enumerate(unique_labels): 
    idxs = labels_rslt == label
    x = embeddings_2d[idxs, 0]
    y = embeddings_2d[idxs, 1]
    # color = colors[i]
    color = label_to_color[label]
    

    plt.scatter(x, y, label=str(idx_to_class[label]), alpha=0.6, s=10, color=color)
    

    # 🛠️ 첫 번째 점 옆에만 텍스트 추가
    if len(x) > 0:
        plt.text(x[0] + 0.2, y[0], str(idx_to_class[label]), fontsize=9, alpha=1, color=color, bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
)

plt.title("t-SNE Projection of character features from DiG")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.grid(True)

plt.legend(
    loc='upper right', 
    title='Characters',
    fontsize=9,
    title_fontsize=10,
    markerscale=2
)

plt.savefig("customUtils/tsne_visual/tsne/base_feat/tsne_feat_baseline_250811.png", dpi=300)
print('saved tsne results')