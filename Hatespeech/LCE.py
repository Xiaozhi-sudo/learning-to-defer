from __future__ import division
import numpy as np
import sys,os
import numpy as np
import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import time
import random
import torch.optim as optim
import copy, time
from hatespeech_model import load_model, infer_cvb0, predict_lang, CNN_rej, CNN
from hatespeech_utils import load_data, categorical_accuracy, epoch_time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    '''
    The L_{CE} loss implementation for hatespeech, identical to CIFAR implementation
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]  # batch_size
    rc = [n_classes] * batch_size
    rc = torch.tensor(rc)
    outputs = -m * torch.log2(outputs[range(batch_size), rc]) - m2 * torch.log2(
        outputs[range(batch_size), labels])  # pick the values corresponding to the labels
    return torch.sum(outputs) / batch_size


def train_reject(model, iterator, optimizer, alpha):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        predictions = model(batch.text)
        batch_size = predictions.size()[0]
        # get expert predictions and costs
        m = (batch.expert) * 1.0  # expert agreement with label: I_{m=y}
        m2 = [1] * batch_size
        m2 = torch.tensor(m2)
        for j in range(0, batch_size):
            exp = m[j].item()
            if exp:
                m2[j] = alpha
            else:
                m2[j] = 1

        m2 = m2.to(device)

        loss = reject_CrossEntropyLoss(predictions, m, batch.label, m2, 3)

        acc = categorical_accuracy(predictions, batch.label.to(device))

        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_reject(model, iterator):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            batch_size = predictions.size()[0]  # batch_size
            m = batch.expert
            m2 = [1] * batch_size
            m2 = torch.tensor(m2)
            m2 = m2.to(device)
            loss = reject_CrossEntropyLoss(predictions, m, batch.label, m2, 3)
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def metrics_print(net, loader):
    net.eval()
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    with torch.no_grad():
        for data in loader:
            outputs = net(data.text)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]            # batch_size
            for i in range(0,batch_size):
                r = (predicted[i].item() == 3)
                if r==0:
                    total += 1
                    correct += (predicted[i] == data.label[i]).item()
                    correct_sys += (predicted[i] == data.label[i]).item()
                if r==1:
                    exp +=  data.expert[i].item()
                    correct_sys += data.expert[i].item()
                    exp_total+=1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001), "alone classifier": 100*alone_correct/real_total }
    print(to_print)
    return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]


def metrics_print_fairness(net, loader):
    net.eval()
    group_1 = 0
    group_1_counts = 0
    group_0 = 0
    group_0_counts = 0

    with torch.no_grad():
        for data in loader:
            outputs = net(data.text)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]            # batch_size
            for i in range(0,batch_size):
                r = (predicted[i].item() == 3)
                prediction = 0
                if r==0:
                    prediction = predicted[i]
                if r==1:
                    prediction = data.expertlabel[i].item()

                if  data.group[i].item() == 0:
                    if data.label[i].item() == 2:
                        group_0_counts += 1
                        if prediction == 1 or prediction ==0:
                            group_0 += 1
                else:
                    if data.label[i].item() == 2:
                        group_1_counts += 1
                        if prediction == 1 or prediction ==0:
                            group_1 += 1


    to_print={"group0":group_0/(group_0_counts+0.0001), "group1": group_1/(group_1_counts+0.0001), "discrimination":group_0/(group_0_counts+0.0001)- group_1/(group_1_counts+0.0001) }
    return [group_0/(group_0_counts+0.0001), group_1/(group_1_counts+0.0001), abs(group_0/(group_0_counts+0.0001)- group_1/(group_1_counts+0.0001))]


def main():
    load_model()

    BATCH_SIZE = 64

    TEXT, train_data, test_data, valid_data = load_data()

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort=False,
        batch_size=BATCH_SIZE,
        device=device)

    # build model
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100  # fixed
    N_FILTERS = 300  # hyperparameterr
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 4
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    model = CNN_rej(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 3, DROPOUT, PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())
    model = model.to(device)

    for i in range(0, 11):
        model = CNN_rej(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 3, DROPOUT, PAD_IDX)

        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        model = model.to(device)
        alpha = i / 10
        N_EPOCHS = 5

        best_valid_loss = 0
        best_model = None
        for epoch in range(N_EPOCHS):

            start_time = time.time()
            train_loss, train_acc = train_reject(model, train_iterator, optimizer, alpha)

            valid_loss = metrics_print(model, valid_iterator)[1]

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss >= best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(model)

        print(metrics_print(best_model, valid_iterator))

    metrics_print_fairness(best_model, test_iterator)
    metrics_print(best_model, test_iterator)


if __name__ == '__main__':
    main()
