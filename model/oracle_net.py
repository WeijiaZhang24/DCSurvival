import torch.nn as nn
import torch

# Define our Weibull survival model
class WeibullModel_indep(nn.Module):
    def __init__(self, num_features, hidden_size = 32):
        super(WeibullModel_indep, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.shape = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def log_likelihood(self, x, t, c):
        x_beta = self.net(x).squeeze()
        log_t_lambda = (self.shape - 1) * torch.log(t / self.scale)
        log_lambda = torch.log(self.scale)
        exp_term = - torch.exp(x_beta) * (t / self.scale) ** self.shape # Survival Function

        logL = torch.mean(c * (torch.log(self.shape) - torch.log(self.scale) \
            + x_beta + log_t_lambda + exp_term) + (1-c)* exp_term)

        return logL
    

# According to UAI2023 paper
def log_clayton_partial_u(u, v, theta):
    result = torch.where ( (u.pow(-theta) + v.pow(-theta)) > 1, (-theta-1)*torch.log(u)  + (-(theta+1)/(theta))*torch.log(u.pow(-theta) + v.pow(-theta) - 1), 0)
    return result
# if u or v is too small, then u.pow(-theta) wil be inf

def log_survival(t, shape, scale, risk):
    return -(torch.exp(risk + shape*torch.log(t) - shape*torch.log(scale))) # used log transform to avoid numerical issue

def survival(t, shape, scale, risk):
    return torch.exp(log_survival(t, shape, scale, risk))

def log_density(t,shape,scale,risk):
    log_hazard = risk + shape*torch.log(t) - shape*torch.log(scale )\
         + torch.log(1/t) + torch.log(shape)
    return log_hazard + log_survival(t, shape, scale, risk)

# Define our Weibull survival model with Clayton Copula
class WeibullModelClayton(nn.Module):
    def __init__(self, num_features, hidden_size = 32):
        super(WeibullModelClayton, self).__init__()
        self.net_t = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.net_c = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.shape_t = nn.Parameter(torch.tensor(1.0)) # Event Weibull Shape
        self.scale_t = nn.Parameter(torch.tensor(1.0)) # Event Weibull Scale
        self.shape_c = nn.Parameter(torch.tensor(1.0)) # Censoring Weibull Shape   
        self.scale_c = nn.Parameter(torch.tensor(1.0)) # Censoring Weibull Scale
        self.theta = nn.Parameter(torch.tensor(1.0)) # Clayton Copula Theta
    def log_likelihood(self, x, t, c):
        x_beta_t = self.net_t(x).squeeze()
        x_beta_c = self.net_c(x).squeeze()

        # In event density, censoring entries should be 0
        event_log_density = c * log_density(t, self.shape_t, self.scale_t, x_beta_t) 
        censoring_log_density = (1-c) * log_density(t, self.shape_c, self.scale_c, x_beta_c)

        S_E = survival(t, self.shape_t, self.scale_t, x_beta_t)
        S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        event_partial_copula = c * log_clayton_partial_u(S_E, S_C, self.theta)
        censoring_partial_copula = (1-c) * log_clayton_partial_u(S_C, S_E, self.theta)
        
        logL = event_log_density + event_partial_copula + censoring_log_density + censoring_partial_copula
    
        return torch.mean(logL), torch.mean(event_log_density), torch.mean(event_partial_copula), torch.mean(censoring_log_density), torch.mean(censoring_partial_copula)
    

    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t,x))
    
    def hazard(self, t, x):
        x_beta_t = self.net_t(x).squeeze()
        return ((self.shape_t/self.scale_t)*((t/self.scale_t)**(self.shape_t-1))) * torch.exp(x_beta_t)
        
    def cum_hazard(self, t, x):
        x_beta_t = self.net_t(x).squeeze()
        return ((t/self.scale_t)**self.shape_t) * torch.exp(x_beta_t)