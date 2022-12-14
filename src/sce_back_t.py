import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def read():
    dir=r"data/train.xlsx"
    data = pd.read_excel(io=dir, sheet_name=0,header=0)
    need_ = data.iloc[1:, 4:]
    need=need_.to_numpy()
    data2 = pd.read_excel(io = dir, sheet_name=1, header=0)
    need1 = data2.iloc[1:, 1].to_numpy()
    need1 = np.append(need1, 0.0)
    
    need11 = [0.0]
    need11 = np.append(need11, [np.linspace(need1[i], need1[i+1], 7)[:-1] for i in range(len(need1) - 1)])
    
    need1 = need11.reshape((-1, 1))
    need_delay=need[1:]
    need2 = np.hstack((need[:-1], need1[:-1]))
    need3 = np.hstack((need2,need_delay[:, 1].reshape((-1, 1))))
    return need3

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net,self).__init__()
        self.net_1 = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),           
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
    return test_loss / test_num

def train(net,loss_func,optimizer,scheduler,epochs,train_dataloader, test_dataloader):
    saved_loss = 1000.0
    state = net.state_dict()
    for epoch in range(epochs):
        avg_loss=0
        cnt=0
        for step, (x, y) in enumerate(train_dataloader):  # x : sec_supp_{t-1}, sec_back_{t-1}, sec_follow_{t-1}, outdoor, indoor, iradiance, sec_supp_t, sec_back_t, sec_follow_t, outdoor_t -> indoor
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
        scheduler.step()
        # print("lr:", scheduler.get_last_lr())
        # print("Epoch ",epoch,avg_loss)#?
        if epoch % 10 == 0:
            test_loss = test(net, loss_func, test_dataloader)
            if test_loss < saved_loss:
                state = net.state_dict()
                saved_loss = test_loss
    torch.save(state, 'model/sec_back_t_MLP_parament.pth')


    
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
    x_,y_=np.hsplit(data,[6])
    mydataset=MyDataset(x_,y_)
    train_dataset, test_dataset = data_split(mydataset, 0.8)

    net = Net(6,256,1)
    print(net)

    net=net.cuda()
    # optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
    optimizer = torch.optim.Adam(net.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30 , gamma=0.1) # 0.02
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    loss_func = torch.nn.MSELoss()

    train_data_l=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,drop_last=True)
    
    test_data_l=DataLoader(dataset=test_dataset,batch_size=1,shuffle=True,drop_last=True)
    train(net,loss_func,optimizer,scheduler,500,train_data_l, test_data_l)
    test(net, loss_func, test_data_l)

    