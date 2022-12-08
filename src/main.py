import dataloader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MLP
import numpy as np



def test_loss_calc():
    test_loss = 0
    test_num = 0
    for step, (data, label) in enumerate(data_test_loader):
        output = net(data)
        loss = loss_func(output, label)
        test_loss += loss.item() * data.size(0)
        test_num += data.size(0)
    return test_loss,test_num

#载入数据
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = dataloader.HeatSupplyDataSet('data/train.xlsx', device)
data_train, data_test = dataloader.data_split(data, 0.8)
# print(data_train.shape())
batchSize = 60

data_train_loader = DataLoader(data_train, batch_size=batchSize)
data_test_loader = DataLoader(data_test, batch_size=1)

# # 指定规模
# input_size = 9
# hidden_size = 128
# output_size = 1

# # 搭建网络
# my_nn = torch.nn.Sequential(
#     torch.nn.Linear(input_size,hidden_size),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(hidden_size,output_size),
# ).to(device=device)

# # 定义损失函数
# cost = torch.nn.MSELoss(reduction = 'mean').to(device= device)
# optimizer = torch.optim.Adam(my_nn.parameters(),lr = 0.001)

# # 训练网络
# losses = []
# for i in range(1000):
#     batch_loss = 0
#     train_cnt = 0
#     # 小批量随机梯度下降进行训练
#     for step, (data, label) in enumerate(data_test_loader):
#         prediction = my_nn(data).squeeze(-1)
#         loss = cost(prediction, label)
#         optimizer.zero_grad()
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         batch_loss += loss.item()
#         train_cnt += 1
        
        
#     # 打印损失
#     # 打印损失值
#     if i % 10 == 0:
#         batch_loss_mean = batch_loss / train_cnt
#         losses.append(batch_loss_mean)
#         print(i, batch_loss_mean)


net = MLP.HeatSupplyMLP().to(device=device)
optimizer = torch.optim.SGD(net.parameters(), lr = 0.02)
loss_func = nn.MSELoss().to(device= device)

for epoch in range(1000):
    train_loss = 0
    train_num = 0
    for step, (data, label) in enumerate(data_train_loader):
        output = net(data).squeeze(-1)
        # print(output)
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        train_num += data.size(0)
        print(data.size(0))
    # print("At", epoch, ", MSE is", train_loss / train_num);
    # train_loss_all.append(train_loss / train_num)

    if epoch%10==0:
        test_loss,test_num=test_loss_calc()
        print("After", epoch, ": Test Set MSE is", test_loss / test_num)



# test_loss = 0
# test_num = 0
# net.eval()
# j = 0
# total = len(data_test)
# for step, (data, label) in enumerate(data_test_loader):
#     output = net(data)
#     if abs(output - label.data) < 0.01:
#         j += 1
# acc = j / total
# print("The ACC is", acc)
#     loss = loss_func(output, label)
#     test_loss += loss.item() * data.size(0)
#     test_num += data.size(0)
    
# print("Test Set MSE is", test_loss / test_num)
