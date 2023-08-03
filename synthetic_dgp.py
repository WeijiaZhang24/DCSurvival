import numpy as np
import torch
# import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula, FrankCopula, ClaytonCopula, IndependenceCopula)
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.allow_tf32 = False

def LOG(x):
    return np.log(x+1e-20*(x<1e-20))

# Generate according to Algorithm 2 in "Copula-based Deep Survival Models for Dependent Censoring"
def inverse_transform(value, risk, shape, scale):
    return (-LOG(value)/np.exp(risk))**(1/shape)*scale       
    # return (-np.log(1-value)/np.exp(risk))**(1/shape)*scale

def linear_dgp( copula_name= 'Frank', sample_size = 30000, covariate_dim= 10, theta=10, rng = np.random.default_rng(), verbose=True):
    # Generate synthetic data (time-to-event and censoring indicator)
    v_e=4; rho_e=14; v_c=3; rho_c=16

    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = rng.uniform(0, 1, (sample_size, covariate_dim))
    # generate censoring risk coefficient beta from 10 dimensional uniforma distribution from 0 to 1
    beta_c = rng.uniform(0, 1, (covariate_dim, ))
    # generate event risk coefficient beta_e from 10 dimensional uniforma distribution from 0 to 1
    beta_e = rng.uniform(0, 1, (covariate_dim,))
    # multiply beta_e with X to get event risk
    event_risk = np.matmul(X, beta_e).squeeze()
    # multiple beta_c with X to get censoring risk
    censoring_risk = np.matmul(X, beta_c).squeeze()

    if copula_name=='Frank':
        copula = FrankCopula(theta=theta)
    elif copula_name=='Gumbel':
        copula = GumbelCopula(theta=theta)
    elif copula_name=='Clayton':
        copula = ClaytonCopula(theta=theta)
    elif copula_name=="Independent":
        copula = IndependenceCopula()
    else:
        raise ValueError('Copula not implemented')
    sample = copula.rvs(sample_size, random_state=rng)
    u = sample[:, 0]
    v = sample[:, 1]

    event_time = inverse_transform(u, event_risk, v_e, rho_e)
    censoring_time = inverse_transform(v, censoring_risk, v_c, rho_c)

    # if verbose==True:
    #     print(copula_name)
    #     # check censoring rate 
    #     print("{:.5f}".format(np.sum(event_time<censoring_time)/len(event_time)))
    #     plt.scatter(u, v, s=15)
    #     plt.savefig('/home/weijia/code/SurvivalACNet_sumo/sample_figs/true_sample_'+ copula_name + '_' +str(theta)  +'.png')
    #     plt.clf()

    # create observed time 
    observed_time = np.minimum(event_time, censoring_time)
    event_indicator = (event_time<censoring_time).astype(int)

    return X, observed_time, event_indicator, event_time, censoring_time, beta_e


def nonlinear_dgp( copula_name= 'Frank', sample_size = 30000, theta=10, rng = np.random.default_rng(142857), verbose=True):
    # Generate synthetic data (time-to-event and censoring indicator)
    v_e=4; rho_e=17; v_c=3; rho_c=16

    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = np.random.uniform(0, 1, (sample_size, 1))
    # # generate censoring risk coefficient beta from 10 dimensional uniforma distribution from 0 to 1
    # beta_c = np.random.uniform(0, 1, (covariate_dim, ))
    # # generate event risk coefficient beta_e from 10 dimensional uniforma distribution from 0 to 1
    # beta_e = np.random.uniform(0, 1, (covariate_dim,))
    # multiply beta_e with X to get event risk
    event_risk = 2* np.sin(X*np.pi).squeeze() 
    # multiple beta_c with X to get censoring risk
    censoring_risk = 2* np.sin(X*np.pi+0.5).squeeze()

    if copula_name=='Frank':
        copula = FrankCopula(theta=theta)
    elif copula_name=='Gumbel':
        copula = GumbelCopula(theta=theta)
    elif copula_name=='Clayton':
        copula = ClaytonCopula(theta=theta)
    elif copula_name=="Independent":
        copula = IndependenceCopula()
    else:
        raise ValueError('Copula not implemented')
    sample = copula.rvs(sample_size, random_state=rng)
    u = sample[:, 0]
    v = sample[:, 1]

    event_time = inverse_transform(u, event_risk, v_e, rho_e)
    censoring_time = inverse_transform(v, censoring_risk, v_c, rho_c)

    # if verbose==True:
    #     print(copula_name)
    #     # check censoring rate 
    #     print("{:.5f}".format(np.sum(event_time<censoring_time)/len(event_time)))
    #     plt.scatter(u, v, s=15)
    #     plt.savefig('/home/weijia/code/SurvivalACNet_sumo/sample_figs/true_sample_'+ copula_name + '_' +str(theta)  +'.png')
    #     plt.clf()

    # create observed time 
    observed_time = np.minimum(event_time, censoring_time)
    event_indicator = (event_time<censoring_time).astype(int)

    return X, observed_time, event_indicator, event_time, censoring_time