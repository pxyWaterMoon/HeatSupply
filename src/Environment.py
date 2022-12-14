import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import pandas as pd

def read():
    dir=r"data/train.xlsx"
    data = pd.read_excel(io=dir, sheet_name=0,header=0)
    need_ = data.iloc[1:, 4:]
    need=need_.to_numpy()
    data2 = pd.read_excel(io = dir, sheet_name=1, header=0)
    need1 = data2.iloc[1:, 1].to_numpy()
    need1 = np.append(need1, 0.0)
    
    need11 = []
    need11 = np.append(need11, [np.linspace(need1[i], need1[i+1], 7)[:-1] for i in range(len(need1) - 1)])
    
    need1 = need11.reshape((-1, 1))
    # print(need1.shape[0])
    need_delay=need[1:]
    # print(need.shape[0])
    need2 = np.hstack((need[:-1], need1))
    need3 = np.hstack((need2,need_delay))
    return need3

class testevn (gym.Env) :
    def __init__(self, sec_back_net, indoor_net, initial_t, data) -> None:
        self.sce_back_net = sec_back_net
        self.indoor_net = indoor_net
        self.initial_t = initial_t
        self.data = data
        self.state = None
    
    def step(self, action): #action 是二次供水温度
        line, pre_indoor, pre_back_t, now_supp_t, now_back_t, pre_sup_t , done = self.state
        data = self.data[line]
        new_data = self.data[line + 1]
        x = [pre_sup_t, pre_back_t, data[2], data[3], pre_indoor, data[5], now_supp_t, now_back_t, data[8], data[9]]
        now_indoor = self.indoor_net(x)
        x = [now_supp_t, now_back_t, data[8], data[9], data[5], action] #?
        new_back_t = self.sce_back_net(x)
        
        if now_indoor < 20.0 or now_indoor > 24.0:
            done = True
            reward = -20000
        else:
            reward = 0.5 * (action - new_data[6]) + 0.5 * abs(now_indoor - pre_indoor)

        new_state = [line + 1, now_indoor, now_back_t, action, new_back_t, now_supp_t, done]
        self.state = new_state
        return new_state, reward, done, {} 
    
    def reset(self):
        line = 0
        data = self.data[line]
        origin_state = [0, data[4], data[1], data[6], data[7], data[0], False]
        self.state = origin_state
        return


sec_back_net = torch.load('sec_back_t_MLP.pkl')
indoor_net = torch.load('indoor_MLP.pkl')
data = read()

env = testevn(sec_back_net ,indoor_net, 24.435, data)