import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

def read():
    dir=r"C:\Users\Cabbage\Desktop\ARTS1425-HeatSupply\data\train.xlsx"
    data = pd.read_excel(io=dir, sheet_name=0,header=0)
    need_=data.iloc[:,4:]
    need=need_.to_numpy()
    # print(need_)
    # print(need)
    need_delay=need[1:]
    need2=np.hstack((need[:-1],need_delay))
    # print("done")
    # print(need2)
    return need2

class Net(nn.Module):
    def __init__(self, n_feature,n_output):
        super(Net,self).__init__()
        self.net_1 = nn.Sequential(
            # nn.Linear(n_feature, 20),
            # nn.ReLU(),
            nn.Linear(n_feature, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.Sigmoid(),
            nn.Linear(20, n_output)
        )
    
    def forward(self,x):
        x = self.net_1(x)
        return x

def train(net,loss_func,optimizer,epochs,dataloader):
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    for epoch in range(epochs):
        avg_loss=0
        cnt=0
        for step, (x, y) in enumerate(dataloader):
            x=torch.Tensor(x).to(device,dtype=torch.float32)
            y=torch.Tensor(y).to(device,dtype=torch.float32)
            prediction = net(x)
            loss = loss_func(prediction,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cnt+=x.size(0)
            avg_loss+=loss.item()*x.size(0)

        scheduler.step()
        avg_loss/=cnt
        print("Epoch ",epoch,avg_loss)#?

class MyDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x=x
        self.y=y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device='cpu'
    print(f"Using {device} device")

    data=read()
    x_,y_=np.hsplit(data,[9])
    mydataset=MyDataset(x_,y_)
    # x=torch.Tensor(x_)
    # y=torch.Tensor(y_)
    # x=x.cuda()
    # y=y.cuda()

    net = Net(9,1)
    print(net)

    net=net.cuda()
    # optimizer = torch.optim.SGD(net.parameters(),lr = 1e-5)
    optimizer = torch.optim.Adam(net.parameters(),lr=0.1)
    loss_func = torch.nn.MSELoss()

    data_l=DataLoader(dataset=mydataset,batch_size=64,shuffle=True,drop_last=True)
    train(net,loss_func,optimizer,500,data_l)

    