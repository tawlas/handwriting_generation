#Import statements
import torch, torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import matplotlib.pyplot as plt
import os
import utils.workspace as ws
from utils.workspace import normalize
from unconditional_generation.generate_unconditionnally import LSTM as lstm
from preprocess_ugm import Preprocess as Preprocess

# Loss functions
def loss_function(y_pred, y_true):
    # make sure down that x1 is of shape b*seq*1
    # print(y_pred)
    # print(y_true)

    param_dim = net_specs["ParamDim"]
    pi, mu1, mu2, sigma1, sigma2, rho, pred_e_hat = y_pred.split(param_dim, dim=-1)
    pred_e = 1/(1+torch.exp(pred_e_hat))
    pi = nn.functional.softmax(pi, dim=-1)
    sigma1 = torch.exp(sigma1)
    sigma2 = torch.exp(sigma2)
    rho = torch.tanh(rho)

    end_stroke, x1, x2 = y_true.split(1, dim=-1)
    # x1 = x1.expand_as(mu1)
    # x2 = x2.expand_as(mu2)

    Z = ((x1-mu1)**2)/(sigma1**2) + ((x2-mu2)**2)/(sigma2**2) - (2*rho*(x1-mu1)*(x2-mu2))/(sigma1*sigma2)
    #Since loss is blowing because of z i devide it by 1000
    Z = Z / 1000
    from_z = torch.exp(-0.5*Z/(1-rho**2))
    print("mm", from_z.min())
    from_rho = 1/torch.sqrt(1-rho**2)
    from_sigma = 0.5/(np.pi*sigma1*sigma2)
    gaussian = from_sigma*from_rho*from_z
    mask_e1 = end_stroke == 1
    mask_e2 = end_stroke == 0
    from_e1 = pred_e*mask_e1
    from_e2 = 1 - pred_e
    from_e2 = from_e2*mask_e2
    from_e = from_e1 + from_e2
    # from_e = from_e.expand_as(pi)

    bivariate_loss = pi*gaussian*from_e
    bivariate_loss = torch.mean(bivariate_loss, dim=-1)
    
    bivariate_loss = torch.mean(-torch.log(bivariate_loss))
    # print("min", Z.max())
    # bivariate_loss = torch.mean(-torch.log(bivariate_loss)/bivariate_loss.size(-1))
    end_stroke_loss = nn.BCEWithLogitsLoss()(pred_e_hat.squeeze(), end_stroke.squeeze())
    # print(bivariate_loss)
    # print(end_stroke_loss)
    loss = end_stroke_loss + bivariate_loss
    # print(loss)
    return loss

def main(experiment_directory, continue_from):
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    specs = ws.load_experiment_specifications(experiment_directory)

    # Loading the dataset
    datapath = specs["DataPath"]
    strokes = np.load(datapath, allow_pickle=True)

    # Preprocessing the dataset
    train_split = specs["TrainSplit"]
    val_split = specs["ValSplit"]
    preprocess = Preprocess(strokes, train_split, val_split)
    train_set = preprocess.train_set.to(device)
    val_set = preprocess.val_set.to(device)
    # print("",train_set)
    # # Shuffle the dataset
    # np.random.seed(1)
    # np.random.shuffle(strokes)

    # # Splitting the dataset
    # train_split = specs["TrainSplit"]
    # val_split = specs["ValSplit"]
    # train_set, val_set = ws.split_dataset(strokes, train_split, val_split)
    # train_set = np.array(sorted(train_set, key=len, reverse=True))
    # val_set = np.array(sorted(val_set, key=len, reverse=True))

    # print("Training set: {}\nValidation set: {}".format(len(train_set), len(val_set)))

    # #normalize x and y coordinates to 0 mean and 1 std
    # normalize(train_set)
    # normalize(val_set)

    # # From numpy arrays to pytorch tensors
    # dataset = [train_set, val_set]
    # for k in range(len(dataset)):
    #     dataset[k] = [torch.from_numpy(x) for x in dataset[k]]
    #     train_set, val_set = tuple(dataset)

    #Instantiating the model
    global net_specs
    net_specs = specs["NetworkSpecs"]
    input_dim = net_specs["InputDim"]
    hidden_dim = net_specs["HiddenDim"]
    n_layers = net_specs["NumLayersLSTM"]
    # output_dim = net_specs["OutputDim"]
    output_dim = 1 + 6*net_specs["ParamDim"]
    dropout = net_specs["Dropout"]
    model = lstm(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=output_dim, dropout=dropout)
    model = model.to(device)

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
        clip, log_frequency, logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log, device)


def train(model, train_set, val_set,start_epoch, epochs, batch_size, optimizer, clip, log_frequency, 
    logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log, device):
    for epoch in range(start_epoch, epochs+1):
        epoch_loss = 0
        epoch_val_loss = 0
        print("epoch: ", epoch)
        # initialize hidden state
        model.train()
        h = model.init_hidden(batch_size)
        h = (each.to(device) for each in h)

        # Looping the whole dataset
        for inputs in model.dataloader(train_set, batch_size):
            # inputs = [x.to(device) for x in inputs]

            labels = torch.zeros_like(inputs)
            for i in range(len(labels)):
                labels[i][:-1, :] = inputs[i][1:, :]
                labels[i][-1, :] = inputs[i][0, :]

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
        # print(epoch_loss)
        loss_log.append(epoch_loss/len(train_set))

        # Get validation loss
        val_h = model.init_hidden(batch_size)
        val_h = (each.to(device) for each in val_h)
        model.eval()
        for val_inputs in model.dataloader(val_set, batch_size):
            val_labels = torch.zeros_like(val_inputs)
            for i in range(len(val_labels)):
                val_labels[i][:-1, :] = val_inputs[i][1:, :]
                val_labels[i][-1, :] = val_inputs[i][0, :]

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