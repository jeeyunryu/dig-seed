import numpy as np
import umap
import matplotlib.pyplot as plt
import string
import seaborn as sns
from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv() # loads variable from .env file into environment
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

sim_chars_str = '0OQD1lLuvUVijMWmnN5SsJT.,;:'
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


voc = list(string.printable[:-6])
voc.append('EOS')
voc.append('PADDING')
voc.append('UNKNOWN')

classes = voc
idx_to_class = dict(zip(range(len(classes)), classes))
class_to_idx = dict(zip(classes, range(len(classes))))

reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeds_openai)




for i, label in enumerate(sim_chars_list):

    x, y = embeddings_2d[i]

    plt.scatter(x, y, label=label, alpha=0.6, s=10) 
    plt.text(x+0.1, y+0.1, label, fontsize=6, alpha=0.8)


plt.title("UMAP of Embeddings from openai")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)

plt.savefig("umap_openai_char_level.png", dpi=300)