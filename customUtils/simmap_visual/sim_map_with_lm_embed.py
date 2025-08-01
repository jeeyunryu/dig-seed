import numpy as np
# import umap
import matplotlib.pyplot as plt
import string
import seaborn as sns
from openai import OpenAI
# from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv() # loads variable from .env file into environment
openai.api_key = os.getenv("OPENAI_API_KEY")

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

embeds_rslt = np.load('./npy_files/embeddings_char_embed.npy')
labels_rslt = np.load('./npy_files/labels_char_embed.npy')

print('finished loading embeddings')

# # 임베딩 모델로 부터 당장 뽑을 것 준비
# sim_chars_str = '0OQD1lLuvUVijMWmnN5SsJT.,;:'
sim_chars_str = '0OQDC'
# sim_chars_str  = '1lL'
# sim_chars_str  = 'uv'
# sim_chars_str  = 'ij'
# sim_chars_str  = '5Ss'

client = OpenAI()
# sim_chars_str = '0OQD1lLuvUVijMWmnN5SsJT.,;:'
sim_chars_list = list(sim_chars_str)
no_embed_openai = len(sim_chars_list)

embeds_openai = []
for char in sim_chars_list:
    response = client.embeddings.create(
        input=char,
        model="text-embedding-3-small",
        dimensions=300
    )
    embeds_openai.append(response.data[0].embedding)

embeds_openai = np.stack(embeds_openai)

selected_embeddings = []
char_labels = []

for char in sim_chars_str:
    label_idx = class_to_idx[char]
    idxs = np.where(labels_rslt == label_idx)[0]
    if len(idxs) > 0:
        selected_embeddings.append(embeds_rslt[idxs[0]])  # 첫 번째 것만 사용
        char_labels.append(char)


selected_embeddings = np.array(selected_embeddings)
sim_matrix = cosine_similarity(selected_embeddings, embeds_openai)

df_sim = pd.DataFrame(sim_matrix, index=char_labels, columns=char_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(df_sim, annot=True, cmap="coolwarm", vmin=0, vmax=1, square=True)
plt.title("Cosine Similarity between Characters (semantic embed vs. LM embed.)")
plt.tight_layout()
plt.savefig("./heatmaps/cosine_heatmap_semantic_and_LM.png", dpi=300)
plt.show()



# # reducer = umap.UMAP(n_components=2, random_state=42)
# reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
# embeddings_2d = reducer.fit_transform(embeds_rslt)

# unique_labels = np.unique(labels_rslt) # 정수 인덱스 라벨 가지고 있음
# sim_chars_list = []

# for char in sim_chars_str:
#     sim_chars_list.append(class_to_idx[char])

# unique_labels_rslt = []
# for label in unique_labels:
#     if label in sim_chars_list:
#         unique_labels_rslt.append(label)


# no_colors_rslt = len(unique_labels_rslt)
# colors = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 10) # 30개 까지 구분 가능함!!

# plt.figure(figsize=(12, 8))
# plt.grid(True)
# plt.xlim(-100, 100)   # 원하는 범위로 설정
# plt.ylim(-100, 100)

# for i, label in enumerate(unique_labels_rslt): 
#     idxs = labels_rslt == label
#     x = embeddings_2d[idxs, 0]
#     y = embeddings_2d[idxs, 1]
#     color = colors[i]
    

#     plt.scatter(x, y, label=str(idx_to_class[label]), alpha=0.6, s=10, color=color)
    

#     # 🛠️ 첫 번째 점 옆에만 텍스트 추가
#     if len(x) > 0:
#         plt.text(x[0] + 0.2, y[0], str(idx_to_class[label]), fontsize=9, alpha=1, color=color, bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
# )

# plt.title("t-SNE Projection of Embeddings (dig + seed (feat))")
# plt.xlabel("tsne-1")
# plt.ylabel("tsne-2")
# plt.grid(True)

# plt.legend(
#     loc='upper right', 
#     title='Characters',
#     fontsize=9,
#     title_fontsize=10,
#     markerscale=2
# )

# plt.savefig("./tsne/tsne_char_embed_feat_0OQDC_limit.png", dpi=300)