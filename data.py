import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

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


class MultiTensorDataSet(Dataset):
    
    def __init__(self, x_list, y_list):
        super().__init__()
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.x_list[0])

    def __getitem__(self, index):
        return (
            [x[index] for x in self.x_list],
            [y[index] for y in self.y_list]
        )
                

class BootstrapDataSetGenerator():
    def __init__(self, data_file, batch_size):
        self.batch_size = batch_size
        df = pd.read_csv(data_file).sample(frac=1)

        train_size = int(len(df)*0.8)
        self.df_train = df[:train_size]
        self.df_test = df[train_size:]

        self.input_size = len(df.columns)-1
        
    def get_X_Y(self, df):
        Y = df.pop("Y").to_numpy().reshape(-1,1)
        X = df.to_numpy()
        return (torch.Tensor(X), torch.Tensor(Y))
        
    def get_data(self, samples=10):
        train_x_list, train_y_list = [], []
        val_x_list, val_y_list = [], []
        
        for i in range(samples):
            df = self.df_train.sample(frac=1, replace=True)

            train_size = int(len(df)*7/8)

            train_X, train_Y = self.get_X_Y(df[:train_size])
            val_X, val_Y  = self.get_X_Y(df[train_size:])

            train_x_list.append(train_X)
            train_y_list.append(train_Y)

            val_x_list.append(val_X)
            val_y_list.append(val_Y)
            
        train_dataset = MultiTensorDataSet(train_x_list, train_y_list)
        val_dataset = MultiTensorDataSet(val_x_list, val_y_list)

        return (
            DataLoader(train_dataset, batch_size=self.batch_size),
            DataLoader(val_dataset, batch_size=self.batch_size),
            self.input_size
        )
