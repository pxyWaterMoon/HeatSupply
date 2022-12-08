import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def read():
    dir=r"data/train.xlsx"
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
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net,self).__init__()
        self.net_1 = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
    
    def forward(self,x):
        x = self.net_1(x)
        return x

def test(net, loss_func, dataloader):
    test_loss = 0
    test_num = 0
    for step, (data, label) in enumerate(dataloader):
        data=torch.Tensor(data).to(device,dtype=torch.float32)
        label=torch.Tensor(label).to(device,dtype=torch.float32)
        output = net(data)
        loss = loss_func(output, label)
        test_loss += loss.item() * data.size(0)
        test_num += data.size(0)
    print("Test error:", test_loss / test_num)

def train(net,loss_func,optimizer,epochs,train_dataloader, test_dataloader):
    for epoch in range(epochs):
        avg_loss=0
        cnt=0
        for step, (x, y) in enumerate(train_dataloader):
            x=torch.Tensor(x).to(device,dtype=torch.float32)
            y=torch.Tensor(y).to(device,dtype=torch.float32)
            prediction = net(x)
            loss = loss_func(prediction,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cnt+=x.size(0)
            avg_loss+=loss.item()*x.size(0)
        avg_loss/=cnt
        print("Epoch ",epoch,avg_loss)#?
        if epoch % 10 == 0:
            test(net, loss_func, test_dataloader)


    
class MyDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x=x
        self.y=y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def data_split(data, rate):
    train_l = int(len(data) * rate)
    test_l = len(data) -  train_l
    train_set, test_set = torch.utils.data.random_split(data, [train_l, test_l])
    return train_set, test_set

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device='cpu'
    print(f"Using {device} device")

    data=read()
    x_,y_=np.hsplit(data,[9])
    mydataset=MyDataset(x_,y_)
    train_dataset, test_dataset = data_split(mydataset, 0.8)
    # x=torch.Tensor(x_)
    # y=torch.Tensor(y_)
    # x=x.cuda()
    # y=y.cuda()

    net = Net(9,10,1)
    print(net)

    net=net.cuda()
    # optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss()

    train_data_l=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,drop_last=True)
    
    test_data_l=DataLoader(dataset=test_dataset,batch_size=1,shuffle=True,drop_last=True)
    train(net,loss_func,optimizer,200,train_data_l, test_data_l)


    