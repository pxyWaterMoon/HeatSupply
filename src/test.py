import torch
import torch.nn as nn
import indoort_MLP
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
# device='cpu'
print(f"Using {device} device")

net = indoort_MLP.Net(6,256,1)
net.load_state_dict(torch.load('model/indoor_MLP_parament.pth'))
net.eval()

net=net.cuda()


data=indoort_MLP.read()
x_,y_=np.hsplit(data,[6])
mydataset=indoort_MLP.MyDataset(x_,y_)
train_dataset, test_dataset = indoort_MLP.data_split(mydataset, 0.8)
loss_func = torch.nn.MSELoss()
test_data_l=indoort_MLP.DataLoader(dataset=test_dataset,batch_size=1,shuffle=True,drop_last=True)
indoort_MLP.test(net, loss_func, test_data_l, device)

