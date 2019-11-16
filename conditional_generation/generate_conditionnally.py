#Import statements
import torch, torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import matplotlib.pyplot as plt
import os
import utils.workspace as ws

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Model definition
class LSTM(nn.Module):
    #corriger input dim etc..
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, window_parameter_dim, char_to_int):
        super().__init__()
        self.batch_first = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.window_parameter_dim = window_parameter_dim
        self.n_layers = n_layers
        self.char_to_int = char_to_int
        self.csl = len(self.char_to_int)
        self.lstm_cell = nn.LSTMCell(input_dim+self.csl, hidden_dim) #inputdim is onehotencode size for x and softwindow
        self.lstm = nn.LSTM(input_dim+self.csl + hidden_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 3*window_parameter_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    #use broadcasting for cleaner code
    def compute_soft_window(self, h, sentences, prev_p=None):
        K = self.window_parameter_dim
        sentences = sentences.to(device)
        if type(prev_p) == type(None):
            prev_p = torch.zeros(h.size(0), 3*K).to(device)
        # This machinery is for getting around in_place operation which destroy coherent backpropagation
        intermediate_p = torch.exp(self.fc1(h.float()))
        p_first = intermediate_p[:,:2*K]
        p_last = intermediate_p[:,2*K:] - prev_p[:, 2*K:]
        p = torch.cat([p_first, p_last], dim=-1)
        p = p.unsqueeze(1)
        alpha, beta, kappa = torch.split(p, K, -1)
        u = torch.arange(1, sentences.size(1)).unsqueeze(-1).to(device)
        phi = alpha*torch.exp(-beta*(kappa-u)**2)
        phi = phi.sum(dim=-1)
        phi = phi.unsqueeze(-1)
        soft_window = phi*sentences
        soft_window = soft_window.sum(1)

        # all_wt = []
        # for i in range(len(sentences)):
        #     c = sentences[i]
        #     u = torch.ones(c.size(0), alpha.size(-1))
        #     if use_cuda:
        #         c = c.cuda()
        #         u = u.cuda()
        #     for k in range(c.size(0)):
        #         u[k] *= k+1
        #     phi = torch.zeros(c.size(0), alpha.size(-1))
        #     if use_cuda:
        #         phi = phi.cuda()
        #     for k in range(phi.size(0)):
        #         phi[k] = alpha[i]*torch.exp(-beta[i]*(kappa[i]-u[k])**2)
        #     phi = torch.sum(phi, dim=-1)
        #     wt = phi.matmul(c)
        #     all_wt.append(wt) 
        # soft_window = torch.stack(all_wt)

        return  soft_window, p.squeeze(1)

    def forward(self, x, sentences, h1, h2):
        p = None
        hx = torch.zeros(h1[0].size(0), x.size(1), h1[0].size(-1)).to(device)
        all_windows = torch.zeros(h1[0].size(0), x.size(1), self.csl).to(device)
        
        for k in range(x.size(1)):
            x_t = x[:, k, :]
            soft_window, p = self.compute_soft_window(h1[0],sentences, p)
            x_t = torch.cat([x_t, soft_window], dim=-1)
            h1 = self.lstm_cell(x_t, h1)
            hx[:, k, :] = h1[0]
            all_windows[:, k, :] = soft_window
        
        x = torch.cat([x, all_windows, hx], dim=-1)
        x, h2 = self.lstm(x, h2)
        x = x.contiguous().view(-1, self.hidden_dim)
        out = self.fc2(x)
        return out, h1, h2
    

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

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

