import torch
import torch.nn as nn

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
