import click
import numpy as np
import pandas as pd

from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from multinetwork import MultiMLPNetwork
from utils import fit_models, eval_models, eval_model
from data import load_data

def create_model(input_size):
    return MultiMLPNetwork(input_size, repeat=10)

    
@click.command()
@click.option("--batch_size", default=32)
@click.option("--learning_rate", default=5e-3)
@click.argument("data_file")
def main(batch_size, learning_rate, data_file):

    train_data, val_data, test_data, input_size = load_data(data_file,
                                                            batch_size)

    net = create_model(input_size)
    
    fit_models(net, train_data, val_data,
               learning_rate=learning_rate,
               batch_size=batch_size)
    _, winner = eval_models(net, val_data)
    
    winner = net.body[winner]

    eval_model(winner, val_data)
    eval_model(winner, test_data)

    
    x = torch.Tensor(np.linspace(0, 1, 10000)).reshape(10000, 1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    outputs = winner(x)
    
    df = pd.DataFrame()
    df["X"] = x.detach().cpu().numpy().flatten()
    df["Y"] = outputs.detach().cpu().numpy().flatten()

    df.to_csv("result.csv", index=False)
    
if __name__ == "__main__":
    main()


