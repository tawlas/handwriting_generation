import torch, torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import os
import utils.workspace as ws
from utils.workspace import normalize


class Preprocess:

    def __init__(self, data, train_split, val_split):
        self.strokes = data
        self.train_split = train_split
        self.val_split = val_split
        self.train_set, self.val_set = self.preprocess(data)

    def even_seq(self, data, seq_len=800):
        """ pad data to seq_len size """
        data = np.array(data)
        shape = data.shape
        if shape[0] > seq_len:
            data = data[:seq_len]
            return data
        elif shape[0] < seq_len:
            last = np.zeros_like(data, shape=(seq_len-shape[0], *shape[1:]))
            data = np.vstack([data,last])
            return data
        else:
            return data

    def preprocess(self, strokes):

        # Shuffle the dataset
        np.random.seed(1)
        np.random.shuffle(self.strokes)

        # Pad the dataset
        strokes = np.array([self.even_seq(stroke) for stroke in strokes])

        # Splitting the dataset
        train_set, val_set = ws.split_dataset(strokes, self.train_split, self.val_split)

        print("Training set: {}\nValidation set: {}".format(len(train_set), len(val_set)))

        #normalize x and y coordinates to 0 mean and 1 std
        normalize(train_set)
        normalize(val_set)

        # From numpy arrays to pytorch tensors
        train_set = torch.from_numpy(train_set)
        val_set = torch.from_numpy(val_set)
        return train_set, val_set
