#Import statements
import torch
import numpy as np
import os
import utils.workspace as ws
import numpy
import matplotlib
import sys 
sys.path.insert(0,'..')
from utils import plot_stroke

# if experimenting from command line interface
experiment_directory = "../conditional_generation"
specs = ws.load_experiment_specifications(experiment_directory)
specs_eval = specs["eval"]

def generate_conditionally(text):
    """ 
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = torch.load(specs_eval["ModelPath"]).to(device)
    model.eval()

    stroke_length = specs["NumTimestep"]
    batch_size = 1
    stroke = torch.zeros(stroke_length, batch_size, 3).to(device)

    h1 = model.init_hidden(batch_size)
    h1 = (h1[0].squeeze(0).to(device), h1[1].squeeze(0).to(device))
    h2 = model.init_hidden(batch_size)
    h2 = (h2[0].to(device), h2[1].to(device))
    text = [text]
    text = ws.preprocess(text, model.char_to_int)
    point = torch.Tensor([[[1,0,0]]]).to(device)
    
    for k in range(stroke_length):

        point, h1, h2 = model(point,text, h1, h2)
        point[:, 0] = torch.sigmoid(point[:, 0])
#         point[:, 0] = torch.Tensor([1 if x > 0.2 else 0 for x in point[:, 0]])
        point[:, 0] = torch.Tensor(np.random.binomial(1,point[:, 0].data.cpu().numpy())).to(device)
        stroke[k] = point
        point = point.unsqueeze(0)

    return np.array(stroke.squeeze().data.cpu())



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Infer a conditional stroke")
    arg_parser.add_argument(
        "--sentence",
        "-s",
        dest="sentence",
        required=True,
        help="The conditional sentence "
    )


    args = arg_parser.parse_args()
    sentence = args.sentence
    stroke = generate_conditionally(sentence)
    print('TEXT:', sentence)
    plot_stroke(stroke)
    
