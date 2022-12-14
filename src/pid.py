from simple_pid import PID
import Environment as env
import torch
import os
def pid_control(env,supp_t_0):
    pid=PID(1,0.1,0.05,setpoint=23,sample_time=None)
    while True:
        supp_t_diff=pid(temp)
        temp=env.step(supp_t_0+supp_t_diff)
        print(supp_t_0+supp_t_diff)
        os.system("pause")


if __name__=="__main__":
    sec_back_net = torch.load('C:\\Users\\Cabbage\\Desktop\\ARTS1425-HeatSupply\\model\\sec_back_t_MLP.pkl')
    print(sec_back_net)
    indoor_net = torch.load('C:\\Users\\Cabbage\\Desktop\\ARTS1425-HeatSupply\\indoor_MLP.pkl')
    data = env.read()

    Envir = env.testevn(sec_back_net ,indoor_net, 24.435, data)
    pid_control(Envir,35.266)