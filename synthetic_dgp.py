import numpy as np
import torch
import numpy as np
from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula, FrankCopula, ClaytonCopula, IndependenceCopula)
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.allow_tf32 = False


# Generate according to Algorithm 2 in "Copula-based Deep Survival Models for Dependent Censoring"
def inverse_transform(value, risk, shape, scale):
    return (-np.log(value)/np.exp(risk))**(1/shape)*scale       
    # return (-np.log(1-value)/np.exp(risk))**(1/shape)*scale

def linear_dgp( copula= 'Frank', sample_size = 30000, covariate_dim= 10, theta=10, rng = np.random.default_rng(142857)):
    # seed = 142857

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

    if copula=='Frank':
        copula = FrankCopula(theta=theta)
        print('Frank!\n')
    elif copula=='Gumbel':
        copula = GumbelCopula(theta=theta)
        print("Gumbel!\n")
    elif copula=='Clayton':
        copula = ClaytonCopula(theta=theta)
        print("Clayton!\n")
    elif copula=="Independent":
        copula = IndependenceCopula()
        print("Independent!\n")
    else:
        raise ValueError('Copula not implemented')
    sample = copula.rvs(sample_size, random_state=rng)
    u = sample[:, 0]
    v = sample[:, 1]

    event_time = inverse_transform(u, event_risk, v_e, rho_e)
    censoring_time = inverse_transform(v, censoring_risk, v_c, rho_c)
    # check censoring rate 
    print("{:.5f}".format(np.sum(event_time<censoring_time)/len(event_time)))

    # create observed time 
    observed_time = np.minimum(event_time, censoring_time)
    event_indicator = (event_time<censoring_time).astype(int)

    return X, observed_time, event_indicator, event_time, censoring_time


def nonlinear_dgp( copula= 'Frank', sample_size = 30000, covariate_dim= 10, theta=10, random_seed = 142857):
    # seed = 142857
    rng = np.random.default_rng(random_seed)

    # Generate synthetic data (time-to-event and censoring indicator)
    v_e=4; rho_e=17; v_c=3; rho_c=16

    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = np.random.uniform(0, 1, (sample_size, covariate_dim))
    # generate censoring risk coefficient beta from 10 dimensional uniforma distribution from 0 to 1
    beta_c = np.random.uniform(0, 1, (covariate_dim, ))
    # generate event risk coefficient beta_e from 10 dimensional uniforma distribution from 0 to 1
    beta_e = np.random.uniform(0, 1, (covariate_dim,))
    # multiply beta_e with X to get event risk
    event_risk = np.matmul(X, beta_e).squeeze()
    # multiple beta_c with X to get censoring risk
    censoring_risk = np.matmul(X, beta_c).squeeze()

    if copula=='Frank':
        copula = FrankCopula(theta=theta)
        print('Frank!\n')
    elif copula=='Gumbel':
        copula = GumbelCopula(theta=theta)
        print("Gumbel!\n")
    elif copula=='Clayton':
        copula = ClaytonCopula(theta=theta)
        print("Clayton!\n")
    elif copula=="Independent":
        copula = IndependenceCopula()
        print("Independent!\n")
    else:
        raise ValueError('Copula not implemented')
    sample = copula.rvs(sample_size)
    u = sample[:, 0]
    v = sample[:, 1]

    event_time = inverse_transform(u, event_risk, v_e, rho_e)
    censoring_time = inverse_transform(v, censoring_risk, v_c, rho_c)
    # check censoring rate 
    print(np.sum(event_time<censoring_time)/len(event_time))

    # create observed time 
    observed_time = np.minimum(event_time, censoring_time)
    event_indicator = (event_time<censoring_time).astype(int)

    return X, observed_time, event_indicator, event_time, censoring_time