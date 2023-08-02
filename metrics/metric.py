import torch
import numpy as np
from tqdm import tqdm
import torch
import copy

def Survival(truth_model, estimate, x, time_steps):
    """
    model: the learned survival model
    truth: the true survival model
    """
    device = torch.device("cpu")
    estimate = copy.deepcopy(estimate).to(device)
    surv1_estimate = torch.zeros((x.shape[0], time_steps.shape[0]),device=device)
    surv1_truth = torch.zeros((x.shape[0], time_steps.shape[0]),device=device)
    x = torch.tensor(x)
    time_steps = torch.tensor(time_steps)
    # use tqdm to show progress
    for i in range(time_steps.shape[0]):
        surv1_estimate[:,i] = estimate.survival(time_steps[i], x)
        surv1_truth[:,i] = truth_model.survival(time_steps[i], x)
    return surv1_truth, surv1_estimate, time_steps, time_steps.max()


def surv_diff(truth_model, estimate, x, steps):
    device = torch.device("cpu")    
    surv1, surv2, time_steps, t_m = Survival(truth_model, estimate, x, steps)
    # integ = torch.abs(surv1-surv2).sum()
    integ = torch.sum( torch.diff(torch.cat([torch.zeros(1), time_steps])) * torch.abs(surv1-surv2)   )
    #integ2 = torch.sum(torch.diff(torch.cat([torch.zeros(surv1.shape[0],1), time_steps], dim=1))*(torch.abs(surv1)), dim=1)
    #return torch.mean(integ/integ2)
    #print(torch.std(integ/t_m))
    #print(integ.shape)
    #print((integ/t_m).shape)
    return (integ/t_m/x.shape[0]).detach().numpy() # t_max and N are the same for all patients


def Survival_aws(truth_model, estimate, x, time_steps):
    """
    model: the learned survival model
    truth: the true survival model
    """
    estimate_device = device = next(estimate.parameters()).device
    # estimate = copy.deepcopy(estimate).to(device)
    surv1_estimate = torch.zeros((x.shape[0], time_steps.shape[0]),device=estimate_device)
    surv1_truth = torch.zeros((x.shape[0], time_steps.shape[0]),device=torch.device("cpu"))
    x = torch.tensor(x)
    time_steps = torch.tensor(time_steps)
    # use tqdm to show progress
    for i in range(time_steps.shape[0]):
        with torch.no_grad():
            surv1_estimate[:,i] = estimate.survival(time_steps[i], x.to(estimate_device))
        surv1_truth[:,i] = truth_model.survival(time_steps[i], x)
    return surv1_truth, surv1_estimate, time_steps, time_steps.max()


def surv_diff_aws(truth_model, estimate, x, steps):
    device = torch.device("cpu")    
    surv1, surv2, time_steps, t_m = Survival_aws(truth_model, estimate, x, steps)
    surv2 = surv2.to(device)
    # integ = torch.abs(surv1-surv2).sum()
    integ = torch.sum( torch.diff(torch.cat([torch.zeros(1), time_steps])) * torch.abs(surv1-surv2)   )
    #integ2 = torch.sum(torch.diff(torch.cat([torch.zeros(surv1.shape[0],1), time_steps], dim=1))*(torch.abs(surv1)), dim=1)
    #return torch.mean(integ/integ2)
    #print(torch.std(integ/t_m))
    #print(integ.shape)
    #print((integ/t_m).shape)
    return (integ/t_m/x.shape[0]).detach().numpy() # t_max and N are the same for all patients

def C_index(t, x, e, model):
    t_s,indices = torch.sort(t)
    x_s = x[indices]
    e_s = e[indices]
    t_mat = t_s.reshape(-1,1) < t_s.reshape(1,-1)
    e_mat = e_s.reshape(-1,1).repeat(1,t.shape[0])
    t_mat = t_mat * e_mat
    s_1 = model.survival(t_s,x_s).reshape(-1,1).repeat(1,t.shape[0])
    t_rep = t_s.reshape(-1,1).repeat(1, t.shape[0])
    s_2 = torch.zeros_like(t_rep, device=x.device)
    for i in range(t_rep.shape[0]):
        s_2[i, :] = model.survival(t_rep[i, :], x_s)

    compare = (s_1 < s_2).type(torch.float32)
    compare = compare * t_mat
    return torch.sum(compare)/torch.sum(t_mat)

def BS(t, event_t, x, model):
    S = model.survival(t, x)
    tmp = ((event_t > t).type(torch.float32) - S)**2
    return torch.mean(tmp)

def BS_censored(t, event_t, x, model,e, km_h, km_p):
    S = model.survival(t, x)
    tmp = ((event_t > t).type(torch.float32) - S)**2
    t_ind1 = (t >= event_t).type(torch.float32) * e
    t_ind2 = (t < event_t).type(torch.float32)
    G_t = KM_evaluater(t, km_h, km_p).clamp(0.01,1)
    G_event = KM_evaluater(event_t, km_h, km_p).clamp(0.01,100)
    
    #print(torch.sum(t_ind1/(G_event+1e-9*(G_event==0))), torch.sum(t_ind2/(G_t+1e-9*(G_t==0))))
    return torch.mean((tmp *(t_ind1/(G_event+1e-9*(G_event==0))))  + (tmp * (t_ind2/(G_t+1e-9*(G_t==0)))))

def IBS(event_t, x, model, t_max,e, km_h, km_p, n_bins=100):
    len_bin = t_max / (n_bins-1)
    ibs = 0
    for t_ in torch.linspace(0, t_max, n_bins):
        #tmp = BS(torch.ones_like(event_t)*t_, event_t, x, model)
        tmp = BS_censored(torch.ones_like(event_t,device=event_t.device)*t_, event_t, x, model, e, km_h, km_p)
        ibs += tmp * len_bin
    return ibs/t_max

def IBS_plain(event_t, x, model, t_max, n_bins=100):
    len_bin = t_max / (n_bins-1)
    ibs = 0
    for t_ in torch.linspace(0, t_max, n_bins):
        tmp = BS(torch.ones_like(event_t, device=event_t.device)*t_, event_t, x, model)
        
        ibs += tmp * len_bin
    return ibs/t_max



def evaluate_c_index(dep_model, indep_model, dgp, test_dict, E_reverse = False):
    E = test_dict['E']
    if E_reverse:
        E = 1-test_dict['E']
    dgp_obs = C_index(test_dict['T'], test_dict['X'], E, dgp).cpu().detach().numpy().item()
    dep_obs = C_index(test_dict['T'], test_dict['X'], E, dep_model).cpu().detach().numpy().item()
    indep_obs = C_index(test_dict['T'], test_dict['X'], E, indep_model).cpu().detach().numpy().item()
    aux_e = torch.ones_like(E, device = E.device)
    t = test_dict['t1']
    if E_reverse:
        t = test_dict['t2']
    dgp_tot = C_index(t, test_dict['X'], aux_e, dgp).cpu().numpy().item()
    dep_tot = C_index(t, test_dict['X'], aux_e, dep_model).cpu().numpy().item()
    indep_tot = C_index(t, test_dict['X'], aux_e, indep_model).cpu().numpy().item()
    return [[dgp_obs, dep_obs, indep_obs], [dgp_tot, dep_tot, indep_tot]]
    
    

def evaluate_IBS(dep_model, indep_model, dgp, test_dict,km_h, km_p, E_reverse):
    t = test_dict['t1']
    if E_reverse:
        t = test_dict['t2']
    dgp_tot = IBS_plain(t, test_dict['X'], dgp, torch.max(t), n_bins=100).cpu().numpy().item()
    dep_tot = IBS_plain(t, test_dict['X'], dep_model, torch.max(t), n_bins=100).cpu().numpy().item()
    indep_tot = IBS_plain(t, test_dict['X'], indep_model, torch.max(t), n_bins=100).cpu().numpy().item()
    E = test_dict['E']
    if E_reverse:
        E = 1-test_dict['E']
    dgp_obs = IBS(test_dict['T'], test_dict['X'], dgp, torch.max(test_dict['T']), E, km_h, km_p).cpu().numpy().item()
    dep_obs = IBS(test_dict['T'], test_dict['X'], dep_model, torch.max(test_dict['T']), E, km_h, km_p).cpu().numpy().item()
    indep_obs = IBS(test_dict['T'], test_dict['X'], indep_model, torch.max(test_dict['T']), E, km_h, km_p).cpu().numpy().item()
    return [[dgp_obs, dep_obs, indep_obs], [dgp_tot, dep_tot, indep_tot]]
    
def KM(t, e):
    device= t.device
    t = t.cpu().numpy().reshape(-1,)
    e = e.cpu().numpy().reshape(-1,)
    indices = np.argsort(t)
    t_sorted = t[indices]
    e_sorted = e[indices]
    event_times = np.unique(t_sorted[e_sorted==1])
    n_events = np.zeros_like(event_times)
    n_at_risk = np.zeros_like(event_times)
    for i in range(len(event_times)):
        et = event_times[i]
        n_events[i] = np.sum((t_sorted * e_sorted)==et)
        n_at_risk[i] = np.sum(t_sorted >= et)
    prob = np.cumprod(1-(n_events/n_at_risk))
    prob_end = np.ones(prob.shape[0]+1)
    prob_end[1:] = prob
    event_times_end = np.zeros(event_times.shape[0]+1)
    event_times_end[1:] = event_times
    return torch.from_numpy(event_times_end).to(device), torch.from_numpy(prob_end).to(device)
    
def KM_evaluater(t, h, p):
    device = t.device
    h = h.cpu().numpy()
    p = p.cpu().numpy()
    t = t.cpu().numpy()
    if len(t.shape) == 1:
        idx = np.digitize(t, h, False)
        return torch.from_numpy(p[idx-1]).to(device)
    else:
        idx = np.digitize(t, h, False)
        prob = np.zeros_like(t)
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prob[i, j] = p[idx[i,j]-1]
        return torch.from_numpy(prob).to(device)