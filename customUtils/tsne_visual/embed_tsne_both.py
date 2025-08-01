import numpy as np
import umap
import matplotlib.pyplot as plt
import string
import seaborn as sns
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from dotenv import load_dotenv
import os

load_dotenv() # loads variable from .env file into environment
openai.api_key = os.getenv("OPENAI_API_KEY")


scaler = StandardScaler()



client = OpenAI()

classes = list(string.printable[:-6])
classes.append('EOS')
classes.append('PADDING')
classes.append('UNKNOWN')

idx_to_class = dict(zip(range(len(classes)), classes))
class_to_idx = dict(zip(classes, range(len(classes))))

# 모델에서 뽑은 임베딩 로딩
embeds_rslt = np.load('/home/jyryu/workspace/DiG/npy_files/embeddings_char_embed.npy')
labels_rslt = np.load('/home/jyryu/workspace/DiG/npy_files/labels_char_embed.npy')
print('finished loading embeddings')

# 임베딩 모델로 부터 당장 뽑을 것 준비
sim_chars_str = '0OQD1lLuvUVijMWmnN5SsJT.,;:'
sim_chars = list(sim_chars_str)
no_embeds_openai = len(sim_chars)

embeds_openai = []
for char in sim_chars:
    response = client.embeddings.create(
        input=char,
        model="text-embedding-3-small",
        dimensions=300
    )
    embeds_openai.append(response.data[0].embedding)

embeds_openai = np.stack(embeds_openai)

embeds_combi = np.vstack([embeds_openai, embeds_rslt])
# import pdb;pdb.set_trace()
labels_combined = sim_chars + list(labels_rslt)

embeds_combi_scaled = scaler.fit_transform(embeds_combi)

# embeddings = np.load('embeddings_base_feat.npy')
# labels = np.load('labels_base_feat.npy')

# reducer = umap.UMAP(n_components=2, random_state=42)
reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
# embeddings_2d = reducer.fit_transform(embeds_combi)
embeddings_2d = reducer.fit_transform(embeds_combi_scaled)

unique_labels = np.unique(labels_rslt)

sim_chars_list = []
for char in sim_chars_str:
    sim_chars_list.append(class_to_idx[char])

unique_labels_rslt = []
for i in unique_labels:
    if i in sim_chars_list:
        unique_labels_rslt.append(i)

# import pdb;pdb.set_trace()
# intersection = list(set(a) & set(b))



# no_colors_rslt = len(unique_labels_rslt)
colors = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 10) # 30개 까지 구분 가능함!!

plt.figure(figsize=(12, 8))
plt.grid(True)


for i in range(no_embeds_openai):
    x, y = embeddings_2d[i]
    plt.scatter(x, y, label=labels_combined[i], alpha=0.7, color='black')  # 자동 색상
    plt.text(x + 0.3, y, labels_combined[i], fontsize=9, color='black')
# import pdb;pdb.set_trace()

for i, label in enumerate(unique_labels_rslt): 
    # if label not in filter_idx:
    #     continue
    idxs = labels_combined[no_embeds_openai:] == label
    x = embeddings_2d[no_embeds_openai:][idxs, 0]
    y = embeddings_2d[no_embeds_openai:][idxs, 1]
    color = colors[i]
    

    plt.scatter(x, y, label=str(label), alpha=0.6, s=10, color=color) 
    plt.text(x[0]+0.5, y[0], str(idx_to_class[label]), fontsize=6, alpha=0.8, color=color)
# import pdb;pdb.set_trace()
# for i, label in enumerate(unique_labels): 
#     idxs = labels == label
#     x = embeddings_2d[idxs, 0]
#     y = embeddings_2d[idxs, 1]
#     color = colors[i]
    

#     plt.scatter(embeddings_2d[idxs, 0],
#                 embeddings_2d[idxs, 1],
#                 label=str(label),
                
#                 alpha=0.6,
#                 s=10, color=color) 
#     plt.text(x[0]+0.5, y[0], str(idx_to_class[label]), fontsize=6, alpha=0.8, color=color)


plt.title("t-SNE dig only")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.grid(True)


plt.savefig("tsne_dig_with_charembed.png", dpi=300)