#Import statements
import torch, torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import matplotlib.pyplot as plt
import os
import utils.workspace as ws

use_cuda = torch.cuda.is_available()

# Model definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_first = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
    
    def forward(self, x, h):
        # Packing the sequences
        x = rnn_utils.pack_sequence(x)

        x, h = self.lstm(x, h)
        x, batch_len = rnn_utils.pad_packed_sequence(x, padding_value=np.inf, batch_first=True)

        # Unpacking them now (sequences)
        unpacked = []
        for k in range(len(x)):
            unpacked.append(x[k][:batch_len[k]])
        x = torch.cat(unpacked).contiguous().view(-1, self.hidden_dim)
        x = self.dropout_layer(x)
        x = self.fc(x)
        return x, h
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if use_cuda:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
    
    def dataloader(self, dataset, batch_size):
        ''' Loads the dataset '''
        n = len(dataset)
        if batch_size > n:
            raise Exception('Batch size must be less than dataset size')
        i = 0
        while i < n:
            if i + batch_size <= n:
                yield dataset[i:i+batch_size]
                i += batch_size
            else:
                # Dropping last uncomplete batch
                i = n
                
    def save_model(self, model_params_dir, filename, epoch):
        ''' Saves the weiths of the model '''

        if not os.path.isdir(model_params_dir):
            os.makedirs(model_params_dir)
            
        torch.save(
            {"epoch": epoch, "model_state_dict": self.state_dict()},
            os.path.join(model_params_dir, filename),
        )
        
    def save_logs(self, logs_dir, loss_log, val_loss_log, epoch):
        ''' Saves the logs of the model '''

        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)
        torch.save(
            { "epoch": epoch, "loss": loss_log, "val_loss": val_loss_log},
            os.path.join(logs_dir, ws.logs_filename),
        )

    def load_model_parameters(self, model_params_dir, checkpoint):
        ''' Loads the weiths of the model and return the corresponding epoch number'''

        filename = os.path.join(model_params_dir, checkpoint + ".pth")

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)

        self.load_state_dict(data["model_state_dict"])

        return data["epoch"]

