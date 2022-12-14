import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatSupplyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(in_features=9, out_features=500, bias=True)
        self.hidden2 = nn.Linear(500, 250)
        self.hidden3 = nn.Linear(250, 100)
        self.hidden4 = nn.Linear(100, 50)
        self.hidden5 = nn.Linear(50, 50)
        self.hidden6 = nn.Linear(50, 50)
        self.hidden7 = nn.Linear(50, 25)
        self.hidden8 = nn.Linear(25, 25)
        self.predict = nn.Linear(25, 1)

    def forward(self, x):
        x = F.sigmoid(self.hidden1(x))
        x = F.sigmoid(self.hidden2(x))
        x = F.sigmoid(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.sigmoid(self.hidden8(x))
        output = self.predict(x)
        return output[:, 0]