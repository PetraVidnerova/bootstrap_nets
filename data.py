import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

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
