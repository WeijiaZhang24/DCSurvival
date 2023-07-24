import torch
import numpy as np

def Survival(model, estimate, x, steps=200):
    device = x.device
    u = torch.ones((x.shape[0],), device=device)*0.001
    time_steps = torch.linspace(1e-4,1,steps,device=device).reshape(1,-1).repeat(x.shape[0],1)
    t_max_model = model.rvs(x, u)
    #t_max_estimate = estimate.rvs(x, u)#.reshape(-1,1).repeat(1,100)
    #e = (t_max_model < t_max_estimate).type(torch.float32)
    t_max = t_max_model#e * t_max_estimate + (1-e) * t_max_model
    t_max = t_max.reshape(-1,1).repeat(1,steps)
    time_steps = t_max * time_steps
    surv1 = torch.zeros((x.shape[0], steps),device=device)
    surv2 = torch.zeros((x.shape[0], steps),device=device)
    for i in range(steps):
        surv1[:,i] = model.survival(time_steps[:,i],x)
        surv2[:,i] = estimate.survival(time_steps[:,i], x)
    return surv1, surv2, time_steps, t_max_model

def surv_diff(model, estimate, x, steps):
    surv1, surv2, time_steps, t_m = Survival(model, estimate, x, steps)
    integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1), device=x.device), time_steps], dim=1))*(torch.abs(surv1-surv2)), dim=1)
    #integ2 = torch.sum(torch.diff(torch.cat([torch.zeros(surv1.shape[0],1), time_steps], dim=1))*(torch.abs(surv1)), dim=1)
    #return torch.mean(integ/integ2)
    #print(torch.std(integ/t_m))
    #print(integ.shape)
    #print((integ/t_m).shape)
    return torch.mean(integ/t_m)#time_steps[:,-1])#add std