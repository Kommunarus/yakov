print('Initializing...')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.vocab import Vectors

import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from random import sample
from datetime import datetime
from subprocess import Popen, PIPE
from boltons.iterutils import pairwise
from coreference_resolution.src.loader import *
from coreference_resolution.src.utils import *
import subprocess
import re
import collections
from boltons.iterutils import chunked_iter


rx_metric = re.compile('METRIC ([a-z]+):')
rx_score = re.compile('([A-Za-z\- ]+): Recall:.* ([0-9\.]+)%\tPrecision:.* ([0-9\.]+)%\tF1:.* ([0-9\.]+)%')

path_dir = '/home/alex/PycharmProjects/coref/coreference_resolution/experiment_7_loss/'
os.makedirs(path_dir, exist_ok=True)
os.makedirs(os.path.join(path_dir, 'cache'), exist_ok=True)


class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim, hidden_dim=150):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)


class Distance(nn.Module):
    """ Learned, continuous representations for: span widths, distance
    between spans
    """

    bins = [1,2,3,4,8,16,32,64]

    def __init__(self, device, distance_dim=20):
        super().__init__()
        self.device = device
        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return to_cuda(torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        ), self.device)


class Genre(nn.Module):
    """ Learned continuous representations for genre. Zeros if genre unknown.
    """

    genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
    _stoi = {genre: idx+1 for idx, genre in enumerate(genres)}

    def __init__(self, genre_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(len(self.genres)+1, genre_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, labels):
        """ Embedding table lookup """
        return self.embeds(self.stoi(labels))

    def stoi(self, labels):
        """ Locate embedding id for genre """
        indexes = [self._stoi.get(gen) for gen in labels]
        return to_cuda(torch.tensor([i if i is not None else 0 for i in indexes]), device)


class Speaker(nn.Module):
    """ Learned continuous representations for binary speaker. Zeros if speaker unknown.
    """

    def __init__(self, speaker_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(3, speaker_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, speaker_labels):
        """ Embedding table lookup (see src.utils.speaker_label fnc) """
        return self.embeds(to_cuda(torch.tensor(speaker_labels), device))


class CharCNN(nn.Module):
    """ Character-level CNN. Contains character embeddings.
    """

    unk_idx = 1
    pad_size = 15

    def __init__(self, filters, device, char_dim=8):
        super().__init__()

        with open(os.path.join(path_dir, 'train_char_vocab.txt'), 'r') as f:
            char_vocab = []
            for line in f.read():
                char_vocab.append(line)

        self.vocab = set(char_vocab)
        self._stoi = {char: idx + 2 for idx, char in enumerate(self.vocab)}

        self.device = device
        self.embeddings = nn.Embedding(len(self.vocab)+2, char_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.pad_size,
                                              out_channels=filters,
                                              kernel_size=n) for n in (3,4,5)])

    def forward(self, sent):
        """ Compute filter-dimensional character-level features for each doc token """
        embedded = self.embeddings(self.sent_to_tensor(sent))
        convolved = torch.cat([F.relu(conv(embedded)) for conv in self.convs], dim=2)
        pooled = F.max_pool1d(convolved, convolved.shape[2]).squeeze(2)
        return pooled

    def sent_to_tensor(self, sent):
        """ Batch-ify a document class instance for CharCNN embeddings """
        tokens = [self.token_to_idx(t) for t in sent]
        batch = self.char_pad_and_stack(tokens)
        return batch

    def token_to_idx(self, token):
        """ Convert a token to its character lookup ids """
        return to_cuda(torch.tensor([self.stoi(c) for c in token]),  self.device)

    def char_pad_and_stack(self, tokens):
        """ Pad and stack an uneven tensor of token lookup ids """
        skimmed = [t[:self.pad_size] for t in tokens]

        lens = [len(t) for t in skimmed]

        padded = [F.pad(t, (0, self.pad_size-length))
                  for t, length in zip(skimmed, lens)]

        return torch.stack(padded)

    def stoi(self, char):
        """ Lookup char id. <PAD> is 0, <UNK> is 1. """
        idx = self._stoi.get(char)
        return idx if idx else self.unk_idx


class DocumentEncoder(nn.Module):
    """ Document encoder for tokens
    """
    def __init__(self, hidden_dim, char_filters, GLOVE, device, n_layers=2):
        super().__init__()

        self.GLOVE = GLOVE
        self.device = device
        # Unit vector embeddings as per Section 7.1 of paper
        glove_weights = F.normalize(self.GLOVE.weights())
        # turian_weights = F.normalize(TURIAN.weights())

        # GLoVE
        self.glove = nn.Embedding(glove_weights.shape[0], glove_weights.shape[1])
        self.glove.weight.data.copy_(glove_weights)
        self.glove.weight.requires_grad = False

        # Turian
        # self.turian = nn.Embedding(turian_weights.shape[0], turian_weights.shape[1])
        # self.turian.weight.data.copy_(turian_weights)
        # self.turian.weight.requires_grad = False

        # Character
        self.char_embeddings = CharCNN(char_filters, device)

        # Sentence-LSTM
        # self.lstm = nn.LSTM(glove_weights.shape[1]+turian_weights.shape[1]+char_filters,
        # self.lstm = nn.LSTM(glove_weights.shape[1]+char_filters,
        #                     hidden_dim,
        #                     num_layers=n_layers,
        #                     bidirectional=True,
        #                     batch_first=True,
        #                     )
        self.lstm = nn.GRU(glove_weights.shape[1]+char_filters,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True,
                            )

        # Dropout
        self.emb_dropout = nn.Dropout(0.60)
        self.lstm_dropout = nn.Dropout(0.20)

    def forward(self, doc):
        """ Convert document words to ids, embed them, pass through LSTM. """

        # Embed document
        embeds = self.embed(doc).unsqueeze_(1)
        # embeds = [self.embed(s) for s in doc.sents]

        # Batch for LSTM
        # packed, reorder = pack(embeds)

        # Apply embedding dropout
        embdropout = self.emb_dropout(embeds)

        # Pass an LSTM over the embeds
        output, _ = self.lstm(embdropout)

        # Apply dropout
        lstmdropout = self.lstm_dropout(output)

        # Undo the packing/padding required for batching
        # states = unpack_and_unpad(output, reorder)

        return lstmdropout.squeeze_(1), embeds.squeeze_(1)
        # return torch.cat(lstmdropout, dim=0), torch.cat(embeds, dim=0)


    def embed(self, doc):
        """ Embed a sentence using GLoVE, Turian, and character embeddings """
        sents = [s for s in doc.sents]
        sent = []
        for i in range(len(sents)):
            sent += sents[i]
        # Embed the tokens with Glove
        glove_embeds = self.glove(lookup_tensor(sent, self.GLOVE, self.device))

        # Embed again using Turian this time
        # tur_embeds = self.turian(lookup_tensor(sent, TURIAN))

        # Character embeddings
        char_embeds = self.char_embeddings(sent)

        # Concatenate them all together
        # embeds = torch.cat((glove_embeds, tur_embeds, char_embeds), dim=1)
        embeds = torch.cat((glove_embeds, char_embeds), dim=1)

        return embeds


class MentionScore(nn.Module):
    """ Mention scoring module
    """
    def __init__(self, gi_dim, attn_dim, distance_dim,  device):
        super().__init__()
        self.device = device
        self.attention = Score(attn_dim)
        self.width = Distance(device,distance_dim)
        self.score = Score(gi_dim)

    def forward(self, states, embeds, doc, is_eval, K=250):
        """ Compute unary mention score for each span
        """

        # Initialize Span objects containing start index, end index, genre, speaker
        spans = [Span(i1=i[0], i2=i[-1], id=idx,
                      speaker=doc.speaker(i), genre=doc.genre)
                 for idx, i in enumerate(compute_idx_spans(doc.sents, L=2))]

        # Compute first part of attention over span states (alpha_t)
        alfa_t = self.attention(states)

        n_i = len(spans)
        n_t = states.size()[0]
        a_t = to_cuda(torch.zeros((n_i,n_t)), self.device)
        ches = torch.exp(alfa_t)
        for i in range(n_i):
            s = spans[i]
            su = torch.sum(torch.exp(alfa_t[s.i1:s.i2+1]))
            a_t[i,:] = torch.squeeze(ches/ su, 1)

        attn_embeds = to_cuda(torch.zeros((n_i,350)), self.device)
        for i in range(n_i):
            s = spans[i]
            b = torch.unsqueeze(a_t[i,s.i1:s.i2+1],1)
            attn_embeds[i,:] = torch.sum(b*embeds[s.i1:s.i2+1,:],0)

        # Regroup attn values, embeds into span representations
        # TODO: figure out a way to batch
        # span_attns, span_embeds = zip(*[(attns[s.i1:s.i2+1], embeds[s.i1:s.i2+1])
        #                                 for s in spans])
        #
        # # Pad and stack span attention values, span embeddings for batching
        # padded_attns, _ = pad_and_stack(span_attns, value=-1e10)
        # padded_embeds, _ = pad_and_stack(span_embeds)
        #
        # # Weight attention values using softmax
        # attn_weights = F.softmax(padded_attns, dim=1)
        #
        # # Compute self-attention over embeddings (x_hat)
        # attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)


        # Compute span widths (i.e. lengths), embed them
        widths = self.width([len(s) for s in spans])

        # Get LSTM state for start, end indexes
        # TODO: figure out a way to batch
        start_end = torch.stack([torch.cat((states[s.i1], states[s.i2]))
                                 for s in spans])

        # Cat it all together to get g_i, our span representation
        g_i = torch.cat((start_end, attn_embeds, widths), dim=1)

        # Compute each span's unary mention score
        mention_scores = self.score(g_i)

        # Update span object attributes
        # (use detach so we don't get crazy gradients by splitting the tensors)
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores.detach())
        ]

        # Prune down to LAMBDA*len(doc) spans
        if is_eval:
            spans = prune(spans, len(doc), LAMBDA = 1)
        else:
            spans = prune(spans, len(doc), LAMBDA = 1)

        # Update antencedent set (yi) for each mention up to K previous antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx-K):idx])
            for idx, span in enumerate(spans)
        ]

        return spans, g_i, mention_scores


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module
    """
    def __init__(self, gij_dim, distance_dim, device):
        super().__init__()
        self.device = device
        self.distance = Distance(device, distance_dim)

        self.score = Score(gij_dim)

    def forward(self, spans, g_i, mention_scores):
        """ Compute pairwise score for spans and their up to K antecedents
        """

        # Extract raw features
        mention_ids, antecedent_ids, distances = zip(*[(i.id, j.id,
                                                i.i2-j.i1)
                                                         for i in spans
                                                         for j in i.yi])

        # For indexing a tensor efficiently
        mention_ids = to_cuda(torch.tensor(mention_ids), self.device)
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids), self.device)

        # Embed them
        phi = self.distance(distances)

        # Extract their span representations from the g_i matrix
        i_g = torch.index_select(g_i, 0, mention_ids)
        j_g = torch.index_select(g_i, 0, antecedent_ids)

        # Create s_ij representations
        pairs = torch.cat((i_g, j_g, i_g*j_g, phi), dim=1)

        # Extract mention score for each mention and its antecedents
        s_i = torch.index_select(mention_scores, 0, mention_ids)
        s_j = torch.index_select(mention_scores, 0, antecedent_ids)

        # Score pairs of spans for coreference link
        s_ij = self.score(pairs)

        # Compute pairwise scores for coreference links between each mention and
        # its antecedents
        coref_scores = torch.sum(torch.cat((s_i, s_j, s_ij), dim=1), dim=1, keepdim=True)

        # Update spans with set of possible antecedents' indices, scores
        spans = [
            attr.evolve(span,
                        yi_idx=[((y.i1, y.i2), (span.i1, span.i2)) for y in span.yi]
                        )
            for span, score, (i1, i2) in zip(spans, coref_scores, pairwise_indexes(spans))
        ]

        # Get antecedent indexes for each span
        antecedent_idx = [len(s.yi) for s in spans if len(s.yi)]

        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores = [to_cuda(torch.tensor([]), self.device)] + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(torch.tensor([[0.]]), self.device)
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        with_epsilon_my = coref_scores.repeat(1,2)
        with_epsilon_my[:,0]=0

        # Batch and softmax
        # get the softmax of the scores for each span in the document given
        probs = [F.softmax(tensr, dim=0) for tensr in with_epsilon]

        # pad the scores for each one with a dummy value, 1000 so that the tensors can 
        # be of the same dimension for calculation loss and what not. 
        probs, _ = pad_and_stack(probs, value=1000)
        probs = probs.squeeze()

        return spans, probs, with_epsilon_my


class CorefScore(nn.Module):
    """ Super class to compute coreference links between spans
    """
    def __init__(self, embeds_dim,
                       hidden_dim,
                       GLOVE,
                       device,
                       char_filters=50,
                       distance_dim=20,
                       ):

        super().__init__()

        # Forward and backward pass over the document
        attn_dim = hidden_dim*2

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = attn_dim*2 + embeds_dim + distance_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim*3 + distance_dim

        # self.GLOVE = GLOVE
        # Initialize modules
        self.encoder = DocumentEncoder(hidden_dim, char_filters, GLOVE, device)
        self.score_spans = MentionScore(gi_dim, attn_dim, distance_dim, device)
        self.score_pairs = PairwiseScore(gij_dim, distance_dim, device)

    def forward(self, doc, is_eval = False):
        """ Enocde document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(doc)

        # Get mention scores for each span, prune
        spans, g_i, mention_scores = self.score_spans(states, embeds, doc, is_eval)

        # Get pairwise scores for each span combo
        spans, coref_scores, with_epsilon_my = self.score_pairs(spans, g_i, mention_scores)

        return spans, coref_scores, with_epsilon_my


class Trainer:
    """ Class dedicated to training and evaluating the model
    """
    def __init__(self, model, train_corpus, val_corpus, device,
                    lr=1e-3, steps=100):

        self.__dict__.update(locals())

        self.train_corpus = list(self.train_corpus)
        self.val_corpus = self.val_corpus

        self.model = to_cuda(model, device)

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad],
                                    lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=0.96)

    def train(self, num_epochs, eval_interval=10, save_interval=10, *args, **kwargs):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)

            # Save often

            if epoch % save_interval == 0:
                self.save_model(path_dir+'/'+str(datetime.now()) + '.pth')

            # Evaluate every eval_interval epochs
            if epoch % eval_interval == 0:
                # print('\n\nEVALUATION\n\n')
                print('epoch: {}'.format(epoch))
                self.model.eval()
                results = self.evaluate(self.val_corpus)
                print(results)

    def train_epoch(self, epoch):
        """ Run a training epoch over 'steps' documents """

        # Set model to train (enables dropout)
        self.model.train()

        self.train_corpus = [doc for doc in self.train_corpus if doc.sents]
        # self.val_corpus = [doc for doc in self.val_corpus if doc.sents]

        # Randomly sample documents from the train corpus
        # batch = chunked_iter(self.train_corpus,2)
        batch = random.sample(self.train_corpus, self.steps)

        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []

        for doc in batch:
        # for document in tqdm(batch):

            # Randomly truncate document to up to 50 sentences
            # doc = document.truncate()

            # Compute loss, number gold links found, total gold links
            loss, mentions_found, total_mentions, corefs_found, total_corefs, corefs_chosen = self.train_doc(doc)

            # Track stats by document for debugging
            # print(document, '| Loss: %f | Mentions: %d/%d | Coref recall: %d/%d | Corefs precision: %d/%d' \
            #     % (loss, mentions_found, total_mentions,
            #         corefs_found, total_corefs, corefs_chosen, total_corefs))

            epoch_loss.append(loss)
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            epoch_corefs.append(safe_divide(corefs_found, total_corefs))
            epoch_identified.append(safe_divide(corefs_chosen, total_corefs))

        # Step the learning rate decrease scheduler
        self.scheduler.step()

        # print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref precision: %f' \
        #         % (epoch, np.mean(epoch_loss), np.mean(epoch_mentions),
        #             np.mean(epoch_corefs), np.mean(epoch_identified)))

    def train_doc(self, document):
        """ Compute loss for a forward pass over a document """
        # with torch.autograd.set_detect_anomaly(True):
        # Extract gold coreference links
        gold_corefs, total_corefs, gold_mentions, total_mentions = extract_gold_corefs(document)
        weights = [5, 1]
        class_weights = to_cuda(torch.FloatTensor(weights), device)
        loss_cross = nn.CrossEntropyLoss(weight=class_weights)
        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        mentions_found, corefs_found, corefs_chosen = 0, 0, 0

        # Predict coref probabilites for each span in a document
        spans, probs, with_epsilon_my = self.model(document)

        # Get log-likelihood of correct antecedents implied by gold clustering
        gold_indexes = to_cuda(torch.zeros_like(probs), device)
        target = to_cuda(torch.zeros(with_epsilon_my.size()[0],dtype=torch.long), device)
        nom = 0
        for idx, span in enumerate(spans):
            # Log number of mentions found
            if (span.i1, span.i2) in gold_mentions:
                mentions_found += 1

                # Check which of these tuples are in the gold set, if any
                golds = [
                    i for i, link in enumerate(span.yi_idx)
                    if link in gold_corefs
                ]

                # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
                if golds:
                    gold_indexes[idx, golds] = 1

                    target[[nom+gold for gold in golds]] = 1

                    # Progress logging for recall
                    corefs_found += len(golds)
                    found_corefs = sum((probs[idx, golds] > probs[idx, len(span.yi_idx)])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    # Otherwise, set gold to dummy
                    gold_indexes[idx, len(span.yi_idx)] = 1
            nom += len(span.yi_idx)

        # Negative marginal log-likelihood
        eps = 1e-8
        # loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps), dim=0) * -1)
        # loss =   torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps)), dim=0) * -1
        loss = loss_cross(with_epsilon_my, target)
        # Backpropagate
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, corefs_chosen)

    def evaluate(self, val_corpus, eval_script='../src/eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        # Predict files
        print('Evaluating on validation corpus...')
        predicted_docs = [self.predict(doc) for doc in val_corpus]
        val_corpus.docs = predicted_docs

        # Output results
        golds_file, preds_file = self.to_conll(val_corpus, eval_script)

        # Run perl script
        # print('Running Perl evaluation script...')
        # p = Popen([eval_script, 'all', golds_file, preds_file], stdout=PIPE)
        # stdout, stderr = p.communicate()
        # results = str(stdout).split('TOTALS')[-1]
        # metrics = ('all')
        metrics = ('muc', 'bcub', 'ceafm')
        results = ''
        for metric in metrics:
            # Run perl script
            scorer_params = ['perl',
                             eval_script,
                             metric,
                             golds_file,
                             preds_file,
                             'none']

            # print('Running Perl evaluation script...')
            output = subprocess.check_output(scorer_params).decode('utf-8').split('\n')
            scores = collections.defaultdict(dict)
            for line in output:
                line = line.strip('\r\n')
                m = rx_metric.match(line)
                if m:
                    metric = m.group(1)
                m = rx_score.match(line)
                if m:
                    scores[metric][m.group(1)] = (m.group(2), m.group(3), m.group(4))
            results += str(scores)+'\n'
            # print(scores)
        # Write the results out for later viewing
        with open('../preds/results.txt', 'w+') as f:
            f.write(results)
            f.write('\n\n\n')

        return results

    def predict(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()
        with torch.no_grad():
        # Initialize graph (mentions are nodes and edges indicate coref linkage)
            graph = nx.Graph()
            viz = nx.Graph()

            # Pass the document through the model
            spans, probs, with_epsilon_my = self.model(doc, is_eval = True)

            # Cluster found coreference links
            for i, span in enumerate(spans):

                # Loss implicitly pushes coref links above 0, rest below 0
                found_corefs = [(idx,  probs[i, idx])
                                for idx, _ in enumerate(span.yi_idx)
                                if probs[i, idx] > 0.01
                                # if probs[i, idx] > probs[i, len(span.yi_idx)]
                                ]

                # If we have any
                if any(found_corefs):

                    # Add edges between all spans in the cluster
                    for (coref_idx, p) in found_corefs:
                        if p>0.0:
                            link = spans[coref_idx]
                            graph.add_edge((span.i1, span.i2), (link.i1, link.i2),  weight=p)
                            viz.add_edge((' '.join(doc.tokens[span.i1:span.i2+1])+'({})'.format(span.i1)), (' '.join(doc.tokens[link.i1:link.i2+1])+'({})'.format(link.i1)),  weight=p)

            # Extract clusters as nodes that share an edge
            clusters = list(nx.connected_components(graph))

            # Initialize token tags
            token_tags = [[] for _ in range(len(doc))]
            token_tags_small = [[] for _ in range(len(doc))]
            # Add in cluster ids for each cluster of corefs in place of token tag
            for idx, cluster in enumerate(clusters):
                for i1, i2 in cluster:
                    if i1 == i2:
                        token_tags[i1].append(f'({idx})')
                        token_tags_small[i1].append(idx)

                    else:
                        token_tags[i1].append(f'({idx}')
                        token_tags[i2].append(f'{idx})')
                        for iii in range(i1,i2+1):
                            token_tags_small[iii].append(idx)

            doc.tags = ['|'.join(t) if t else '-' for t in token_tags]
            doc.graf = viz
            doc.nun_clusters = len(clusters)
            doc.tags_small = [str(t[0]) if t else '-1' for t in token_tags_small]

        return doc

    def to_conll(self, val_corpus, eval_script):
        """ Write to out_file the predictions, return CoNLL metrics results """

        # Make predictions directory if there isn't one already
        golds_file, preds_file = '../preds/golds.txt', '../preds/predictions.txt'
        if not os.path.exists('../preds/'):
            os.makedirs('../preds/')

        # Combine all gold files into a single file (Perl script requires this)
        golds_file_content = flatten([doc.raw_text for doc in val_corpus])
        with io.open(golds_file, 'w', encoding='utf-8', errors='strict') as f:
            currdoc = 'qq'
            first = True
            for line in golds_file_content:
                docline = line.split()[1]
                if docline != currdoc:
                    currdoc = docline
                    if first:
                        first = False
                    else:
                        f.write('\n')
                        f.write('#end document\n')
                    f.write('#begin document (Doc{});\n'.format(docline))
                f.write(line)

        # Dump predictions
        with io.open(preds_file, 'w', encoding='utf-8', errors='strict') as f:

            currdoc = 'qq'
            first = True
            val_content = flatten([doc.raw_text for doc in val_corpus])
            for line in val_content:
                docline = line.split()[1]
                if docline != currdoc:
                    currdoc = docline
                    if first:
                        first = False
                    else:
                        f.write('\n')
                        f.write('#end document\n')
                    f.write('#begin document (Doc{});\n'.format(docline))
                    doc = val_corpus[int(currdoc)]
                    current_idx = 0
                tokens = line.split()
                tokens[-1] = doc.tags[current_idx]
                current_idx += 1
                f.write('\t'.join(tokens))
                f.write('\n')

                # for line in doc.raw_text:

                    # Indicates start / end of document or line break
                    # if line.startswith('#begin') or line.startswith('#end') or line == '\n':
                    #     f.write(line)
                    #     continue
                    # else:
                        # Replace the coref column entry with the predicted tag
                        # tokens = line.split()
                        # tokens[-1] = doc.tags[current_idx]
                        #
                        # # Increment by 1 so tags are still aligned
                        # current_idx += 1
                        #
                        # # Rewrite it back out
                        # f.write('\t'.join(tokens))
                    # f.write('\n')

        return golds_file, preds_file

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath, device):
        """ Load state dictionary into model """
        state = torch.load(loadpath,  map_location=device)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model, device)

if __name__ == '__main__':
    # Initialize model, train

    train_corpus, val_corpus, _ = load_dataset(True, True, False, path_dir)
    GLOVE = load_glove(path_dir)
    device = torch.device('cuda:1')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CorefScore(embeds_dim=350, hidden_dim=100, GLOVE=GLOVE, device=device)
    # model = nn.DataParallel(model)
    # model = CorefScore(embeds_dim=400, hidden_dim=200)
    # ?? train for 150 epochs, each each train 100 documents
    trainer = Trainer(model, train_corpus, val_corpus, steps=20, device=device)
    trainer.train(num_epochs=150, eval_interval=10, save_interval=50)
