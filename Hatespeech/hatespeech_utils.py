from __future__ import division
import numpy as np
import torch
from torchtext import data
from torchtext import datasets


def load_data():
    labeled_data_path = "data/labeled_data.csv"  # change path if needed
    TEXT = data.Field(tokenize='spacy', batch_first=True)
    LABEL = data.LabelField(dtype=torch.long, sequential=False, use_vocab=False)
    EXPERT = data.LabelField(dtype=torch.long, sequential=False, use_vocab=False)
    GROUP = data.LabelField(dtype=torch.long, sequential=False, use_vocab=False)
    EXPERTLABEL = data.LabelField(dtype=torch.long, sequential=False, use_vocab=False)

    fields = [(None, None), (None, None), ('expertlabel', EXPERTLABEL), ('group', GROUP), ('expert', EXPERT),
              ('label', LABEL), ('text', TEXT)]

    train_data_orig = data.TabularDataset.splits(
        path='',
        train=labeled_data_path,
        format='csv',
        fields=fields,
        skip_header=True)

    # build expert data
    all_data = train_data_orig[0]

    p = 0.75  # expert probability of being correct for AA tweeet
    q = 0.9  # expert probability of being correct for AA tweeet

    # tracker variables for statistics
    sum = 0
    total = 0
    i = 0
    aa_frac = 0
    for example in all_data:
        lang = predict_lang(vars(example)['text'])
        aa = 0
        try:
            if lang[0] >= 0.5:
                aa = 1
        except:
            print("error processing tweet: " + str(vars(example)['text']))
        label = vars(example)['label']
        exp = 0  # 0: expert wrong, 1: expert is right
        exp_label = 0
        if aa == 1:  # if tweet is african american

            coin = np.random.binomial(1, p)
            if coin:
                exp = 1
                exp_label = np.long(label)
            else:
                exp_label = np.long(np.argmax(np.random.multinomial(1, [1 / 3] * 3, size=1)))
                exp = 0
        else:
            coin = np.random.binomial(1, q)
            if coin:
                exp = 1  # is right 90% of time
                exp_label = np.long(label)
            else:
                exp_label = np.long(np.argmax(np.random.multinomial(1, [1 / 3] * 3, size=1)))
                exp = 0
        # if label =='2' : # 2: neither, 1: offensive, 0: hate speech
        #    aa = 1
        vars(all_data[i])['expertlabel'] = exp_label
        vars(all_data[i])['group'] = str(aa)
        vars(all_data[i])['expert'] = exp
        aa_frac += aa
        i += 1
        total += 1
        sum += exp

    LABEL.build_vocab(all_data)
    EXPERT.build_vocab(all_data)
    GROUP.build_vocab(all_data)
    EXPERTLABEL.build_vocab(all_data)
    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(all_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    train_data, test_data, valid_data = all_data.split(split_ratio=[0.6, 0.1, 0.3])

    return TEXT, train_data, test_data, valid_data

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]]).to(device)

