from coreference_resolution.src.coref import CorefScore, Trainer
from coreference_resolution.src.loader import load_glove, creat_dataset
import random
import numpy as np
import torch
SEED = 1234
import matplotlib.pyplot as plt
import networkx as nx

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

path_dir = '../experiment_7_loss/'

# savemodel = '2020-11-17 09:35:41.547827.pth'
savemodel = path_dir+'2020-11-26 12:10:20.829183.pth'

# test_corpus = creat_dataset('Я увидел, что Иванов подошёл ко мне. Он предложил нам пойти в кино.')
test_corpus = creat_dataset('Привет, меня зовут Михаил, мне 30 лет и я живу в Москве. Давайте cейчас расскажу про отца. Он был страшный и темный. Его я постараюсь забыть.' )
GLOVE = load_glove(path_dir)
device = torch.device('cuda:0')
# device = torch.device('cpu')

model = CorefScore(embeds_dim=350, hidden_dim=100, GLOVE = GLOVE, device=device)

trainer = Trainer(model, [], test_corpus, steps=100, device=device)
trainer.load_model(savemodel, device=device)


predicted_docs = [trainer.predict(doc) for doc in test_corpus]
print(' '.join(['{}{}'.format(x,y) for (x,y) in zip(predicted_docs[0].tokens, predicted_docs[0].tags)]))


G = predicted_docs[0].graf

# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
e1 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.2]
e2 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.4 and d["weight"]>0.2]
e3 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.6 and d["weight"]>0.4]
e4 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.8 and d["weight"]>0.6]
e5 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 1 and d["weight"]>0.8]

pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=e1, width=1, edge_color="r")
nx.draw_networkx_edges(G, pos, edgelist=e2, width=1, edge_color="r")
nx.draw_networkx_edges(G, pos, edgelist=e3, width=3, edge_color="b")
nx.draw_networkx_edges(G, pos, edgelist=e4, width=3, edge_color="b")
nx.draw_networkx_edges(G, pos, edgelist=e5, width=4, edge_color="g")

# labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()