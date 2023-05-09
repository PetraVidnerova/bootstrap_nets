import click
from functools import partial
import numpy as np
import pandas as pd


from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from multinetwork import MultiMLPNetwork
from utils import fit_models, eval_models, eval_model
from data import load_data

def create_model(input_size):
    return MultiMLPNetwork(input_size, repeat=10)

def do_the_work(batch_size, learning_rate, data_file,  i):
    train_data, val_data, test_data, input_size = load_data(data_file,
                                                            batch_size=batch_size)
    net = create_model(input_size)
    
    fit_models(net, train_data, val_data,
               learning_rate=learning_rate,
               batch_size=batch_size)
    _, winner = eval_models(net, val_data)
    
    winner = net.body[winner]

    residuals = eval_model(winner, test_data)

    residuals2 = eval_model(winner.half(), test_data, half=True)

    
    return residuals, residuals2
    
#print(f"********** FINISHED {i} ************* ")
    # x = torch.Tensor(np.linspace(0, 1, 10000)).reshape(10000, 1)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # x = x.to(device)
    # outputs = winner(x)
    
    # df = pd.DataFrame()
    # df["X"] = x.detach().cpu().numpy().flatten()
    # df["Y"] = outputs.detach().cpu().numpy().flatten()

    # #    df.to_csv("result.csv", index=False)
    

@click.command()
@click.option("--steps", default=1000)
@click.option("--batch_size", default=32)
@click.option("--learning_rate", default=1e-1)
@click.argument("data_file")
def main(steps, batch_size, learning_rate, data_file):

    pool = mp.Pool(5)
    
    residuals = []
    residuals2 = [] 
    for res, res2 in  tqdm(pool.imap_unordered(
            partial(do_the_work, batch_size, learning_rate, data_file),
            range(steps)
    ), total = steps):
        residuals.append(res)
        residuals2.append(res2)
        
    result, result2 = torch.cat(residuals, dim=1), torch.cat(residuals2, dim=1)
    torch.save(result, "residuals.pt")
    torch.save(result2, "residuals2.pt")
    
if __name__ == "__main__":
    main()


