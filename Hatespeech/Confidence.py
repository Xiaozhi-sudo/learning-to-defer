from __future__ import division
import numpy as np
import sys,os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import time
import random
import torch.optim as optim
import copy, time
from scipy.special import softmax
from hatespeech_utils import load_data, categorical_accuracy, epoch_time
from hatespeech_model import load_model, infer_cvb0, predict_lang

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN_(nn.Module):
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
        return out


def train_expert(model_exp, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model_exp.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model_exp(batch.text)

        loss = criterion(predictions, batch.expert)

        acc = categorical_accuracy(predictions, batch.expert)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_expert(model_exp, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model_exp.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model_exp(batch.text)
            loss = criterion(predictions, batch.expert)
            acc = categorical_accuracy(predictions, batch.expert)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train(model_class, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model_class.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model_class(batch.text)
        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model_class, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model_class.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model_class(batch.text)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def metrics_print_confid(net_class, net_exp, loader):
    net_class.eval()
    net_exp.eval()
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    with torch.no_grad():
        for data in loader:
            outputs_class = net_class(data.text)
            outputs_exp = net_exp(data.text)
            _, predicted = torch.max(outputs_class.data, 1)
            batch_size = outputs_class.size()[0]  # batch_size
            for i in range(0, batch_size):
                arr = [outputs_class.data[i][0].item(), outputs_class.data[i][1].item(),
                       outputs_class.data[i][2].item()]
                arr = softmax(arr)
                r_score = 1 - np.max(arr)  # outputs_class.data[i][predicted[i].item()].item()
                arr_exp = [outputs_exp.data[i][0].item(), outputs_exp.data[i][1].item()]
                arr_exp = softmax(arr_exp)
                r_score = r_score - arr_exp[0]
                r = 0
                if r_score >= 0:
                    r = 1
                if r == 0:
                    total += 1
                    correct += (predicted[i] == data.label[i]).item()
                    correct_sys += (predicted[i] == data.label[i]).item()
                if r == 1:
                    exp += data.expert[i].item()
                    correct_sys += data.expert[i].item()
                    exp_total += 1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001),
                "alone classifier": 100 * alone_correct / real_total}
    print(to_print)
    return [100 * total / real_total, 100 * correct_sys / real_total, 100 * exp / (exp_total + 0.0002),
            100 * correct / (total + 0.0001)]


def metrics_print_confid_fairness(net_class, net_exp, loader):
    net_class.eval()
    net_exp.eval()
    group_1 = 0
    group_1_counts = 0
    group_0 = 0
    group_0_counts = 0
    with torch.no_grad():
        for data in loader:
            outputs_class = net_class(data.text)
            outputs_exp = net_exp(data.text)
            _, predicted = torch.max(outputs_class.data, 1)
            batch_size = outputs_class.size()[0]  # batch_size
            for i in range(0, batch_size):
                arr = [outputs_class.data[i][0].item(), outputs_class.data[i][1].item(),
                       outputs_class.data[i][2].item()]
                arr = softmax(arr)
                r_score = 1 - np.max(arr)  # outputs_class.data[i][predicted[i].item()].item()
                arr_exp = [outputs_exp.data[i][0].item(), outputs_exp.data[i][1].item()]
                arr_exp = softmax(arr_exp)
                r_score = r_score - arr_exp[0]
                r = 0
                if r_score >= 0:
                    r = 1
                prediction = 0
                if r == 0:
                    prediction = predicted[i]
                if r == 1:
                    prediction = data.expertlabel[i].item()

                if data.group[i].item() == 0:
                    if data.label[i].item() == 2:
                        group_0_counts += 1
                        if prediction == 1 or prediction == 0:
                            group_0 += 1
                else:
                    if data.label[i].item() == 2:
                        group_1_counts += 1
                        if prediction == 1 or prediction == 0:
                            group_1 += 1
    print(group_1_counts)
    print(group_0_counts)

    to_print = {"group0": group_0 / (group_0_counts + 0.0001), "group1": group_1 / (group_1_counts + 0.0001),
                "discrimination": group_0 / (group_0_counts + 0.0001) - group_1 / (group_1_counts + 0.0001)}
    print(to_print)
    return [group_0 / (group_0_counts + 0.0001), group_1 / (group_1_counts + 0.0001),
            abs(group_0 / (group_0_counts + 0.0001) - group_1 / (group_1_counts + 0.0001))]


def main():
    TEXT, train_data, test_data, valid_data = load_data()

    BATCH_SIZE = 64

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort=False,
        batch_size=BATCH_SIZE,
        device=device)


    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 300
    FILTER_SIZES = [3,4,5]
    OUTPUT_DIM = 2
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model_expert = CNN_(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 2, DROPOUT, PAD_IDX)
    pretrained_embeddings = TEXT.vocab.vectors

    model_expert.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model_expert.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model_expert.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = torch.optim.Adam(model_expert.parameters())
    criterion = nn.CrossEntropyLoss()

    model_expert = model_expert.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train_expert(model_expert, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate_expert(model_expert, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_expert.state_dict(), 'tut3-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        N_FILTERS = 300
        FILTER_SIZES = [3, 4, 5]
        OUTPUT_DIM = 3
        DROPOUT = 0.5
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model_class = CNN_(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
        pretrained_embeddings = TEXT.vocab.vectors

        model_class.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model_class.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model_class.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        import torch.optim as optim

        optimizer = optim.Adam(model_class.parameters())
        criterion = nn.CrossEntropyLoss()

        model_class = model_class.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 5

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_acc = train(model_class, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model_class, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model_class.state_dict(), 'tut3-model.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    metrics_print_confid(model_class, model_expert, test_iterator)
    print(metrics_print_confid_fairness(model_class, model_expert, test_iterator))



if __name__ == '__main__':
    main()