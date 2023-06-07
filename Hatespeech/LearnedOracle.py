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
from hatespeech_model import load_model, infer_cvb0, predict_lang, CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def metrics_print_oracle(net_class, loader):
    # prints classification metrics for Oracle baseline
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
            _, predicted = torch.max(outputs_class.data, 1)
            batch_size = outputs_class.size()[0]  # batch_size
            for i in range(0, batch_size):
                r = 0
                arr = [outputs_class.data[i][0].item(), outputs_class.data[i][1].item(),
                       outputs_class.data[i][2].item()]
                arr = softmax(arr)
                # r = (data.group[i].item() == 0)
                if data.group[i].item() == 0:
                    if np.max(arr) <= 0.90:
                        r = 1
                else:
                    if np.max(arr) <= 0.75:
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


def metrics_print_oracle_fairness(net_class, loader):
    net_class.eval()
    group_1 = 0
    group_1_counts = 0
    group_0 = 0
    group_0_counts = 0
    with torch.no_grad():
        for data in loader:
            outputs_class = net_class(data.text)
            _, predicted = torch.max(outputs_class.data, 1)
            batch_size = outputs_class.size()[0]  # batch_size
            for i in range(0, batch_size):
                r = 0
                arr = [outputs_class.data[i][0].item(), outputs_class.data[i][1].item(),
                       outputs_class.data[i][2].item()]
                arr = softmax(arr)
                # r = (data.group[i].item() == 0)
                if data.group[i].item() == 0:
                    if np.max(arr) <= 0.90:
                        r = 1
                else:
                    if np.max(arr) <= 0.75:
                        r = 1
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

    to_print = {"group0": group_0 / (group_0_counts + 0.0001), "group1": group_1 / (group_1_counts + 0.0001),
                "discrimination": group_0 / (group_0_counts + 0.0001) - group_1 / (group_1_counts + 0.0001)}
    return [group_0 / (group_0_counts + 0.0001), group_1 / (group_1_counts + 0.0001),
            abs(group_0 / (group_0_counts + 0.0001) - group_1 / (group_1_counts + 0.0001))]

def metrics_print_classifier(net_class, loader):
    # print classification metrics of the classifier alone on all the dataset
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
            _, predicted = torch.max(outputs_class.data, 1)
            batch_size = outputs_class.size()[0]            # batch_size
            for i in range(0,batch_size):
                total += 1
                correct += (predicted[i] == data.label[i]).item()
                correct_sys += (predicted[i] == data.label[i]).item()
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001), "alone classifier": 100*alone_correct/real_total }
    print(to_print)
    return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]


def metrics_print_classifier_fairness(net_class, loader):
    # print fairness metrics of the classifier alone on all the dataset
    net_class.eval()
    group_1 = 0
    group_1_counts = 0
    group_0 = 0
    group_0_counts = 0
    with torch.no_grad():
        for data in loader:
            outputs_class = net_class(data.text)
            _, predicted = torch.max(outputs_class.data, 1)
            batch_size = outputs_class.size()[0]            # batch_size
            for i in range(0,batch_size):
                prediction = predicted[i]
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
    print(to_print)
    return [group_0/(group_0_counts+0.0001), group_1/(group_1_counts+0.0001), abs(group_0/(group_0_counts+0.0001)- group_1/(group_1_counts+0.0001))]


def metrics_print_expert_fairness(loader):
    # print fairness metrics of the expert on all the dataset
    group_1 = 0
    group_1_counts = 0
    group_0 = 0
    group_0_counts = 0
    with torch.no_grad():
        for data in loader:
            batch_size = len(data)  # batch_size
            for i in range(0, batch_size):
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

    to_print = {"group0": group_0 / (group_0_counts + 0.0001), "group1": group_1 / (group_1_counts + 0.0001),
                "discrimination": group_0 / (group_0_counts + 0.0001) - group_1 / (group_1_counts + 0.0001)}
    print(to_print)
    return [group_0 / (group_0_counts + 0.0001), group_1 / (group_1_counts + 0.0001),
            abs(group_0 / (group_0_counts + 0.0001) - group_1 / (group_1_counts + 0.0001))]

def main():

    TEXT, train_data, test_data, valid_data = load_data()

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 300
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 3
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model_class = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
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

    BATCH_SIZE = 64

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort=False,
        batch_size=BATCH_SIZE,
        device=device)

    metrics_print_expert_fairness(test_iterator)
    metrics_print_classifier_fairness(model_class, test_iterator)
    metrics_print_oracle(model_class, test_iterator)
    metrics_print_oracle_fairness(model_class, test_iterator)


if __name__ == '__main__':
    main()