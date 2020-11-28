#!/usr/bin/env python3

import argparse
import random
import numpy as np
import torch
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import pymorphy2
import subprocess

utils = importr('udpipe')


SEED = 1234
import matplotlib.pyplot as plt
import networkx as nx
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

NPRO = {'я': ('он','она'),
        'меня': ('его','её'),
        'мне': ('ему','ей'),
        'мной': ('им','ею'),
        'мною': ('им','ею'),
        'обо мне': ('о нем','о ней'),
        'мы': ('они','они'),
        'нас': ('их','их'),
        'ей': ('им','им'),
        'вами': ('ими','ими'),
        'о нас': ('о них','о них'),
        'мой': ('его','её'),
        'нам': ('им','им'),
        'мои': ('его','её'),
        'моего': ('его','её'),
        }


def onestep():
    from coreference_resolution.src.coref import CorefScore, Trainer
    from coreference_resolution.src.loader import load_glove, creat_dataset

    path_dir = '/home/alex/PycharmProjects/coref/coreference_resolution/experiment_7_loss/'

    savemodel = path_dir + '2020-11-26 12:10:20.829183.pth'

    # test_corpus = creat_dataset('Я увидел, что Иванов подошёл ко мне. Он предложил нам пойти в кино.')
    file = opt.outdir+'input.txt'
    with open(file, 'r') as f:
        text = f.read()

    test_corpus = creat_dataset(text)
    GLOVE = load_glove(path_dir)
    device = torch.device('cuda:0')

    model = CorefScore(embeds_dim=350, hidden_dim=100, GLOVE=GLOVE, device=device)

    trainer = Trainer(model, [], test_corpus, steps=1, device=device)
    trainer.load_model(savemodel, device=device)

    predicted_docs = [trainer.predict(doc) for doc in test_corpus]

    print(predicted_docs[0].tokens)
    print(predicted_docs[0].tags)

    with open(opt.outdir+'cluster.txt', 'w') as f:
        f.write('%'.join(predicted_docs[0].tokens))
        f.write('\n')
        f.write('%'.join(predicted_docs[0].tags_small))
        f.write('\n')
        f.write(str(predicted_docs[0].nun_clusters))

    return predicted_docs[0].graf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--readfile', type=str, default='TRUE', help='')
    # parser.add_argument('--file', type=str, default='./flask/static/input.txt', help='')
    # parser.add_argument('--string', type=str, default=text, help='')

    # parser.add_argument('--outdir', type=str, default='./flask/static/', help='')
    parser.add_argument('--outdir', type=str, default='./outdataoffice/', help='')

    opt = parser.parse_args()
    # кластера кореференции
    G = onestep()
    plt.figure(figsize=(5, 5))

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
    plt.savefig(opt.outdir+'graf.png')