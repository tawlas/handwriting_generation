#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import torch
import numpy as np

specifications_filename = "specs.json"
logs_filename = "Logs.pth"

def one_hot_encode(sentence, char_set):
        one_hot_sentence = []
        for value in sentence:
            one_hot = [0 for _ in range(len(char_set))]
            one_hot[value-1] = 1
            one_hot_sentence.append(one_hot)
        return one_hot_sentence

def split_dataset(dataset, train_split, val_split):
        train_set = dataset[:int(train_split*len(dataset))]
        val_set = dataset[int(train_split*len(dataset)):int((train_split+val_split)*len(dataset))]
        if train_split + val_split == 1:
            return train_set, val_set
        else:
            test_set = dataset[int((train_split+val_split)*len(dataset)):]
            return train_set, val_set, test_set

def preprocess(sentences, char_to_int):
    sentences_ints = []
    for sentence in sentences:
        sentences_ints.append([char_to_int[char] for char in sentence])
    one_hot_sentences = [torch.Tensor(one_hot_encode(sentence, char_to_int)) for sentence in sentences_ints]
    return one_hot_sentences

def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))

def load_logs(logs_dir):

    full_filename = os.path.join(logs_dir, logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["val_loss"],
        data["epoch"],
    )

def clip_logs(loss_log, val_loss_log, epoch):

    loss_log = loss_log[: epoch]
    val_loss_log = val_loss_log[: epoch]
    return (loss_log, val_loss_log)

#normalize x and y coordinates to 0 mean 1 std
def normalize(data):
    data_concat = np.concatenate(data, axis=0)
    means = np.mean(data_concat, axis=0)
    stds = np.std(data_concat, axis=0)
    x_mean = means[1]
    y_mean = means[2]
    x_std = stds[1]
    y_std = stds[2]
    for element in data:
        element[:,1] = (element[:,1] - x_mean) / x_std
        element[:,2] = (element[:,2] - y_mean) / y_std
