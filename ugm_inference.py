#Import statements
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import utils.workspace as ws
import numpy
import matplotlib
import sys
sys.path.insert(0,'..')
from utils import plot_stroke

experiment_directory = "../unconditional_generation/"
specs = ws.load_experiment_specifications(experiment_directory)
specs_eval = specs["eval"]

def generate_unconditionally(random_seed=1):
    """ 
    Input:
       random_seed - integer

    Output:
       stroke - numpy 2D-array (T x 3)
    return stroke
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(random_seed)
    model = torch.load(specs_eval["ModelPath"]).to(device)
    model.eval()

    stroke_length = np.random.randint(specs_eval["StrokeMinLength"], specs_eval["StrokeMaxLength"]+1)
    batch_size = 1
    stroke = torch.zeros(stroke_length, batch_size, 3).to(device)
    hidden = model.init_hidden(batch_size)
    hidden = (hidden[0].to(device), hidden[1].to(device))
    point = torch.Tensor([[[1,0,0]]]).to(device)
    
    # Randomize Length of stroke with random seed
    for k in range(stroke_length):
        model.eval()

        point, hidden = model(point,hidden)
        point[:, 0] = torch.sigmoid(point[:, 0])
#         point[:, 0] = torch.Tensor([1 if x > 0.5 else 0 for x in point[:, 0]])
        point[:, 0] = torch.Tensor(np.random.binomial(1,point[:, 0].data.cpu().numpy())).to(device)
        stroke[k] = point
        point = point.unsqueeze(0)

    return np.array(stroke.squeeze().data.cpu())




if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Generate a random stroke")
    arg_parser.add_argument(
        "--random_seed",
        "-r",
        dest="random_seed",
        required=False,
        help="The random seed"
    )


    args = arg_parser.parse_args()
    random_seed = args.random_seed if args.random_seed else 1
    stroke = generate_unconditionally(random_seed)
    print('Random seed:', random_seed)
    plot_stroke(stroke)