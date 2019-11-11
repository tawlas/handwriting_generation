#Import statements
import torch, torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import matplotlib.pyplot as plt
import os
import utils.workspace as ws
from utils.workspace import normalize
from unconditional_generation.generate_unconditionnally import LSTM as lstm

# Loss functions
def loss_function(y_pred, y_true):
    
        lift_true = y_true[:, 0]
        lift_pred = y_pred[:, 0]
        lift_loss = nn.BCEWithLogitsLoss()(lift_pred, lift_true)
        l1_loss = nn.L1Loss()(y_pred[:, 1:] , y_true[:, 1:])
        loss = lift_loss + l1_loss
        return loss

def main(experiment_directory, continue_from):
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Training on GPU")
    else:
        print("Training on CPU")

    specs = ws.load_experiment_specifications(experiment_directory)

    # Loading the dataset
    datapath = specs["DataPath"]
    strokes = np.load(datapath, allow_pickle=True)
    # Shuffle the dataset
    np.random.seed(1)
    np.random.shuffle(strokes)

    # Splitting the dataset
    train_split = specs["TrainSplit"]
    val_split = specs["ValSplit"]
    train_set, val_set = ws.split_dataset(strokes, train_split, val_split)
    train_set = np.array(sorted(train_set, key=len, reverse=True))
    val_set = np.array(sorted(val_set, key=len, reverse=True))

    print("Training set: {}\nValidation set: {}".format(len(train_set), len(val_set)))

    #normalize x and y coordinates to 0 mean and 1 std
    normalize(train_set)
    normalize(val_set)

    # From numpy arrays to pytorch tensors
    dataset = [train_set, val_set]
    for k in range(len(dataset)):
        dataset[k] = [torch.from_numpy(x) for x in dataset[k]]
        train_set, val_set = tuple(dataset)

    #Instantiating the model
    net_specs = specs["NetworkSpecs"]
    input_dim = net_specs["InputDim"]
    hidden_dim = net_specs["HiddenDim"]
    n_layers = net_specs["NumLayersLSTM"]
    output_dim = net_specs["OutputDim"]
    dropout = net_specs["Dropout"]
    model = lstm(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=output_dim, dropout=dropout)
    if use_cuda:
        model = model.cuda()

    # training parameters optimization function
    lr=specs["LearningRate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = specs["NumEpochs"]
    start_epoch = 1
    batch_size = specs["BatchSize"]
    clip = specs["GradClip"]
    log_frequency = specs["LogFrequency"]
    logs_dir = specs["LogsDir"]
    model_params_dir = specs["ModelDir"]
    checkpoints = list(
        range(
            specs["CheckpointFrequency"],
            specs["NumEpochs"] + 1,
            specs["CheckpointFrequency"],
        )
    )

    loss_log = []
    val_loss_log = []

    # Resuming the training from a particular saved checkpoint
    if continue_from is not None:

        print('Continuing from "{}"'.format(continue_from))

        model_epoch = model.load_model_parameters(model_params_dir, continue_from)
        loss_log, val_loss_log, log_epoch = ws.load_logs(logs_dir)

        if not log_epoch == model_epoch:
            loss_log, val_loss_log = ws.clip_logs(
                loss_log, val_loss_log, model_epoch
            )

        start_epoch = model_epoch + 1

        print("Model loaded from epoch: {}".format(model_epoch))

    print("Starting training")
    train(model, train_set, val_set, start_epoch, epochs, batch_size, optimizer,
        clip, log_frequency, logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log, use_cuda)


def train(model, train_set, val_set,start_epoch, epochs, batch_size, optimizer, clip, log_frequency, 
    logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log, use_cuda=False):
    for epoch in range(start_epoch, epochs+1):
        epoch_loss = 0
        epoch_val_loss = 0
        print("epoch: ", epoch)
        # initialize hidden state
        model.train()
        h = model.init_hidden(batch_size)

        # Looping the whole dataset
        for inputs in model.dataloader(train_set, batch_size):
            if use_cuda:
                inputs = [x.cuda() for x in inputs]

            labels = [torch.zeros_like(k) for k in inputs]
            for i in range(len(labels)):
                labels[i][:-1, :] = inputs[i][1:, :]
                labels[i][-1, :] = inputs[i][0, :]
            labels = torch.cat(labels)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = loss_function(output, labels)
            epoch_loss += loss.item()
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        loss_log.append(epoch_loss/len(train_set))

        # Get validation loss
        val_h = model.init_hidden(batch_size)
        model.eval()
        for val_inputs in model.dataloader(val_set, batch_size):
            if use_cuda:
                val_inputs = [x.cuda() for x in val_inputs]
            val_labels = [torch.zeros_like(k) for k in val_inputs]
            for i in range(len(val_labels)):
                val_labels[i][:-1, :] = val_inputs[i][1:, :]
                val_labels[i][-1, :] = val_inputs[i][0, :]
            val_labels = torch.cat(val_labels)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            val_h = tuple([each.data for each in val_h])
            val_output, val_h = model(val_inputs, val_h)
            val_loss = loss_function(val_output, val_labels)
            epoch_val_loss += val_loss.item()

        val_loss_log.append(epoch_val_loss/len(val_set))

        model.train()


        if epoch in checkpoints:
            model.save_model(model_params_dir, str(epoch)+".pth", epoch)

        if epoch % log_frequency == 0:
            model.save_model(model_params_dir, "latest.pth", epoch)
            model.save_logs(
                logs_dir,
                loss_log,
                val_loss_log,
                epoch,
            )

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train an Unconditional Generator")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json"
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )


    args = arg_parser.parse_args()

    main(args.experiment_directory, args.continue_from)