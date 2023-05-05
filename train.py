import click
import numpy as np
import pandas as pd

from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class MultiMLPNetwork(nn.Module):
    def __init__(self,  input_size, repeat=10):
        super().__init__()
        
        self.input_size = input_size
        self.repeat = repeat

        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in range(self.repeat)
        ])

            

    def forward(self, x):
        return torch.cat([
            net(x)
            for net in self.body
        ], dim=1)
        
    def split_eval(self, x):
        return [ net(x)
                 for net in self.body
        ]

    
    
def create_model(input_size):
    return MultiMLPNetwork(input_size, repeat=10)


def fit_models(model, train_dl, val_dl, learning_rate=0.001, batch_size=32):

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.7) 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    
    model.train()

    for epoch in range(1000):
        loss_ep = 0

        print(f"Epoch {epoch} ... ")
        #        with tqdmtotal=len(train_dl)) as t:
        for batch_idx, (data, targets) in enumerate(train_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            multi_targets = targets.repeat(1, model.repeat)
            ## Forward Pass
            scores = model(data)
            loss = criterion(scores, multi_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            # if epoch > 0 and epoch % 300 == 0:
            #     for g in optimizer.param_groups:
            #         print(f"**** {g['lr']}")
            #         g['lr'] = 0.9*g['lr']
    
        
            
        print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        with torch.no_grad():
            sum_loss = 0
            print("Computing validation accuracy ...")
            #            with tqdm(total=len(val_dl)) as t:
            for batch_idx, (data,targets) in enumerate(val_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                multi_targets = targets.repeat(1, model.repeat)
                ## Forward Pass
                scores = model(data)
#                print(scores, targets)
                sum_loss += criterion(scores, multi_targets).item() #((scores-targets)**2).sum()
            #       t.update()
            print(
                f"VAL loss: {float(sum_loss) / len(val_dl)  :.2f}"
            )

def eval_models(model, test_dl):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss()

    model.eval()
    sum_loss = torch.zeros(model.repeat).to(device)
    with torch.no_grad():
        for data, targets in test_dl:
            data = data.to(device=device)
            targets = targets.to(device=device)
            # Forward Pass
            scores = model.split_eval(data)
            # geting predictions
            sum_loss += torch.cat([
                criterion(single_scores, targets).reshape(1)
                for single_scores in scores
            ])
    sum_loss /= len(test_dl)
    print(
        f"TEST accuracy: {sum_loss.min() :.2f}"
    )
    print(sum_loss)
    return sum_loss.min(), sum_loss.argmin()
   
def eval_model(model, test_dl):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    
    model.eval()
    sum_loss = 0
    outputs = []
    with torch.no_grad():
        for data, targets in test_dl:
            data = data.to(device=device)
            targets = targets.to(device=device)
            # Forward Pass
            scores = model(data)
            outputs.append(scores)
            # geting predictions
            sum_loss += criterion(scores, targets).item()
    print(
        f"TEST accuracy: {float(sum_loss) / len(test_dl) :.2f}"
    )
    return outputs


def load_data(data_file, batch_size=32):
    df = pd.read_csv(data_file)

    df = df.sample(frac=1)

    Y = df.pop("Y").to_numpy().reshape(-1,1)
    X = df.to_numpy()

    input_size = len(df.columns)
    
    X_train, X_val, X_test = X[:700], X[700:800], X[800:] 
    Y_train, Y_val, Y_test = Y[:700], Y[700:800], Y[800:] 

    X_train, X_val, X_test = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(X_test)
    Y_train, Y_val, Y_test = torch.Tensor(Y_train), torch.Tensor(Y_val), torch.Tensor(Y_test)
    
    
    train_data = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size)
    val_data = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)
    test_data = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)

    return train_data, val_data, test_data, input_size

    
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


