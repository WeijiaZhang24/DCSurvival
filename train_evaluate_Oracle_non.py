import torchtuples as tt
import torch # For building the networks 
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm

from model.oracle_net import WeibullModelCopula, WeibullModel_indep
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model.truth_net import Weibull_linear, Weibull_nonlinear
from metrics.metric import surv_diff, surv_diff_aws
from synthetic_dgp import linear_dgp, nonlinear_dgp
from sklearn.model_selection import train_test_split

sample_size=30000
torch.set_num_threads(6)
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
likelihood_threshold = 1e-10
num_epochs = 5000


method ='uai2023'
risk = 'linear'
print(method, risk)

def main():
    for theta_true in [0,2,4,6,8,10,12,14,16,18,20]:
        survival_l1 = []
        for repeat in range(5):
            seed = 142857 + repeat
            rng = np.random.default_rng(seed)
            if theta_true==0:
                copula_form = "Independent"
            else:
                copula_form = "Clayton"
                print(copula_form)
            if risk == 'linear':
                X, observed_time, event_indicator, _, _, beta_e = linear_dgp( copula_name=copula_form, theta=theta_true, sample_size=sample_size, rng=rng, verbose=False)
            elif risk == 'nonlinear':
                X, observed_time, event_indicator, _, _ = nonlinear_dgp( copula_name=copula_form, theta=theta_true, sample_size=sample_size, rng=rng, verbose=False)
            # split train test
            X_train, X_test, y_train, y_test, indicator_train, indicator_test = train_test_split(X, observed_time, event_indicator, test_size=0.33, 
                                                                                                 stratify= event_indicator, random_state=repeat)
            # split train val
            X_train, X_val, y_train, y_val, indicator_train, indicator_val = train_test_split(X_train, y_train, indicator_train, test_size=0.33)

            patience = 20  # Or whatever value you choose: how many epochs to wait for improvement in validation loss before stopping
            best_val_loss = float('inf')  # Initialize best validation loss as infinity
            counter = 0  # Initialize counter
            if method=='uai2023':
                times_tensor_train = torch.tensor(y_train).to(device)
                event_indicator_tensor_train = torch.tensor(indicator_train).to(device)
                covariate_tensor_train = torch.tensor(X_train).to(device)
                dataset = TensorDataset(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train)     
                val_dataset = TensorDataset(torch.tensor(X_val).to(device), torch.tensor(y_val).to(device), torch.tensor(indicator_val).to(device))
                batch_size = 8192  # set your batch size
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                model = WeibullModelCopula(num_features = X.shape[1], copula=copula_form).to(device)
                optimizer_event = optim.Adam([{"params": [model.scale_t], "lr": 0.01},
                                            {"params": [model.shape_t], "lr": 0.01},
                                            {"params": model.net_t.parameters(), "lr": 0.01},
                                        ])
                optimizer_censoring = optim.Adam([{"params": [model.scale_c], "lr": 0.01},
                                            {"params": [model.shape_c], "lr": 0.01},
                                            {"params": model.net_c.parameters(), "lr": 0.01},
                                        ])
                optimizer_theta = optim.Adam([{"params": [model.theta], "lr": 0.01}])  
                # Train the model
                for epoch in range(num_epochs):
                    for covariates, times, events in dataloader:  # iterate over batches
                        optimizer_theta.zero_grad()
                        optimizer_event.zero_grad()
                        optimizer_censoring.zero_grad()
                        log_likelihood, event_log_density, event_partial_copula, censoring_log_density, censoring_partial_copula = \
                            model.log_likelihood(covariates, times, events)
                        (-log_likelihood).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        optimizer_censoring.step()
                        optimizer_event.step()
                        if epoch > 1000: 
                            optimizer_theta.step()                       
                        # Evaluate validation loss
                        val_loss = 0
                        with torch.no_grad():
                            for covariates, times, events in val_loader:
                                log_likelihood, _, _, _, _ = model.log_likelihood(covariates, times, events)
                                val_loss += -log_likelihood.item()
                                # print("val_loss = ", val_loss)
                        val_loss /= len(val_loader)

                        # If the validation loss has decreased, save the model and reset the counter
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), 'best_model.pt')  # Or any other filename or path
                            counter = 0
                        else:  # If the validation loss did not decrease, increment the counter
                            counter += 1
                            if counter >= patience:  # If counter reaches patience, stop training
                                break

            if risk == "linear":
                truth_model = Weibull_linear(num_feature= X_test.shape[1], shape = 4, scale = 14, device = torch.device("cpu"), coeff = beta_e)
            elif risk == "nonlinear":
                truth_model = Weibull_nonlinear(shape = 4, scale = 17, device = torch.device("cpu"))

            # calculate survival_l1 based on ground truth survival function
            steps = np.linspace(y_test.min(), y_test.max(), 10000)
            performance = surv_diff_aws(truth_model, model, X_test, steps)
            print(performance)
            survival_l1.append(performance)
        print("theta_true = ", theta_true, "survival_l1 = ", np.nanmean(survival_l1), "+-", np.nanstd(survival_l1))

if __name__ == "__main__":
    main()
