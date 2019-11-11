# import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import utils.workspace as ws


def load_logs(experiment_directory):
    specs = ws.load_experiment_specifications(experiment_directory)
    logs = torch.load(os.path.join(specs["LogsDir"], ws.logs_filename))

    print("latest epoch is {}".format(logs["epoch"]))

    num_epoch = len(logs["val_loss"])

    fig, ax = plt.subplots()

    ax.plot(
        np.arange(1, num_epoch+1),
        logs["loss"],
        "#000000",
        label="Training Loss"
    )
    
    ax.plot(
        np.arange(1, num_epoch+1),
        logs["val_loss"],
        "#999999",
        label="Validation Loss"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    ax.set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Losses")

    ax.grid()
    plt.show()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Plot DeepSDF training logs")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment "
        + "specifications in 'specs.json', and logging will be done in this directory "
    )

    args = arg_parser.parse_args()

    load_logs(args.experiment_directory)
