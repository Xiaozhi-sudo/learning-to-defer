from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

vocabfile = "twitteraae_models/model_vocab.txt" # change path if needed, path inside twitteraae repo is twitteraae/model/model_vocab.txt
modelfile = "twitteraae_models/model_count_table.txt" # change path if needed, path inside twitteraae repo is twitteraae/model/model_vocab.txt

# the following functions are copied from twitteraae for convenience
K=0; wordprobs=None; w2num=None

def load_model():
    """Idempotent"""
    global vocab,w2num,N_wk,N_k,wordprobs,N_w,K, modelfile,vocabfile
    if wordprobs is not None:
        # assume already loaded
        return

    N_wk = np.loadtxt(modelfile)
    N_w = N_wk.sum(1)
    N_k = N_wk.sum(0)
    K = len(N_k)
    wordprobs = (N_wk + 1) / N_k

    vocab = [L.split("\t")[-1].strip() for L in open(vocabfile,encoding="utf8")]
    w2num = {w:i for i,w in enumerate(vocab)}
    assert len(vocab) == N_wk.shape[0]

def infer_cvb0(invocab_tokens, alpha, numpasses):
    global K,wordprobs,w2num
    doclen = len(invocab_tokens)

    # initialize with likelihoods
    Qs = np.zeros((doclen, K))
    for i in range(0,doclen):
        w = invocab_tokens[i]
        Qs[i,:] = wordprobs[w2num[w],:]
        Qs[i,:] /= Qs[i,:].sum()
    lik = Qs.copy()  # pertoken normalized but proportionally the same for inference

    Q_k = Qs.sum(0)
    for itr in range(1,numpasses):
        # print "cvb0 iter", itr
        for i in range(0,doclen):
            Q_k -= Qs[i,:]
            Qs[i,:] = lik[i,:] * (Q_k + alpha)
            Qs[i,:] /= Qs[i,:].sum()
            Q_k += Qs[i,:]

    Q_k /= Q_k.sum()
    return Q_k

def predict_lang(tokens, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
    invocab_tokens = [w.lower() for w in tokens if w.lower() in w2num]
    # check that at least xx tokens are in vocabulary
    if len(invocab_tokens) < thresh1:
        return None
    # check that at least yy% of tokens are in vocabulary
    elif len(invocab_tokens) / len(tokens) < thresh2:
        return None
    else:
        posterior = infer_cvb0(invocab_tokens, alpha=alpha, numpasses=numpasses)
        return posterior


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax()

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        out = self.fc(cat)
        out = self.softmax(out)
        return out


class CNN_rej(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.embedding_rej = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs_rej = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc_rej = nn.Linear(len(filter_sizes) * n_filters, 1)

        self.dropout_rej = nn.Dropout(dropout)

        self.softmax = nn.Softmax()

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        embedded_rej = self.embedding_rej(text)

        # embedded = [batch size, sent len, emb dim]

        embedded_rej = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved_rej = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs_rej]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_rej = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat_rej = self.dropout_rej(torch.cat(pooled, dim=1))

        out_rej = self.fc_rej(cat_rej)
        # cat = [batch size, n_filters * len(filter_sizes)]

        out = self.fc(cat)
        out = torch.cat((out, out_rej), 1)

        out = self.softmax(out)
        return out