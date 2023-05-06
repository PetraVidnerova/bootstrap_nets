import click
import numpy as np
import pandas as pd

from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from multinetwork import MultiMLPNetwork, MultiModel
from utils import fit_models, eval_models, eval_model, fit_multi_model
from data import load_data, BootstrapDataSetGenerator


def create_model(input_size):
    return MultiMLPNetwork(input_size, repeat=10)

    
@click.command()
@click.option("--batch_size", default=32)
@click.option("--learning_rate", default=1e-1)
@click.argument("data_file")
def main(batch_size, learning_rate, data_file):

    model_batch = 100
    repeat = 10
    
    dataset = BootstrapDataSetGenerator(data_file, batch_size=batch_size, repeat=repeat)
    train_set, val_set, input_size = dataset.get_data(samples=model_batch)

    multi_model = MultiModel(
        MultiMLPNetwork,
        {"input_size": input_size, "repeat": repeat},
        size=model_batch
    )

    fit_multi_model(
        multi_model,
        train_set,
        val_set,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    

    # train_data, val_data, test_data, input_size = load_data(data_file,
    #                                                         batch_size)

    # net = create_model(input_size)
    
    # fit_models(net, train_data, val_data,
    #            learning_rate=learning_rate,
    #            batch_size=batch_size)
    # _, winner = eval_models(net, val_data)
    
    # winner = net.body[winner]

    # eval_model(winner, val_data)
    # eval_model(winner, test_data)

    
    # x = torch.Tensor(np.linspace(0, 1, 10000)).reshape(10000, 1)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # x = x.to(device)
    # outputs = winner(x)
    
    # df = pd.DataFrame()
    # df["X"] = x.detach().cpu().numpy().flatten()
    # df["Y"] = outputs.detach().cpu().numpy().flatten()

    # df.to_csv("result.csv", index=False)
    
if __name__ == "__main__":
    main()


