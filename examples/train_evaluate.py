import warnings

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.metrics.metric import surv_diff
from dcsurvival.model.truth_net import Weibull_linear, Weibull_nonlinear
from dcsurvival.survival import DCSurvival
from dcsurvival.synthetic_dgp import linear_dgp, nonlinear_dgp

sample_size=30000
torch.set_num_threads(24)
torch.set_default_tensor_type(torch.DoubleTensor)
warnings.simplefilter(action="ignore", category=FutureWarning)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

method ="ours"
risk = "linear"
print(method, risk)

depth = 2
widths = [100, 100]
lc_w_range = (0, 1.0)
shift_w_range = (0., 2.0)
num_epochs = 5000
batch_size = 30000
early_stop_epochs = 100

def main() -> None:
    for theta_true in [5]:
        survival_l1 = []
        copula_form = "Independent" if theta_true == 0 else "Frank"
        print(copula_form)
        for repeat in range(5):
            seed = 142857 + repeat
            rng = np.random.default_rng(seed)
            if risk == "linear":
                X, observed_time, event_indicator, _, _, beta_e = linear_dgp( copula_name=copula_form,
                                                                             theta=theta_true, sample_size=sample_size, rng=rng, verbose=False)
            elif risk == "nonlinear":
                X, observed_time, event_indicator, _, _ = nonlinear_dgp(copula_name=copula_form,
                                                                         theta=theta_true, sample_size=sample_size, rng=rng, verbose=False)
            # split train test
            X_train, X_test, y_train, y_test, indicator_train, indicator_test = train_test_split(X, observed_time, event_indicator, test_size=0.33, stratify= event_indicator, random_state=repeat)
            # split train val
            X_train, X_val, y_train, y_val, indicator_train, indicator_val = train_test_split(X_train, y_train, indicator_train, test_size=0.33, stratify= indicator_train, random_state=repeat)

            if risk == "linear":
                truth_model = Weibull_linear(num_feature= X_test.shape[1], shape = 4, scale = 14, device = torch.device("cpu"), coeff = beta_e)
            elif risk == "nonlinear":
                truth_model = Weibull_nonlinear(shape = 4, scale = 17, device = torch.device("cpu"))

            times_tensor_train = torch.tensor(y_train).to(device)
            event_indicator_tensor_train = torch.tensor(indicator_train).to(device)
            covariate_tensor_train = torch.tensor(X_train).to(device)

            times_tensor_val = torch.tensor(y_val).to(device)
            event_indicator_tensor_val = torch.tensor(indicator_val).to(device)
            covariate_tensor_val = torch.tensor(X_val).to(device)

            phi = DiracPhi(depth, widths, lc_w_range, shift_w_range, device, tol = 1e-14).to(device)
            model = DCSurvival(phi, device = device, num_features=X.shape[1], tol=1e-14).to(device)
            # optimizer = optim.Adam(model.parameters(), lr = 0.001)
            optimizer = optim.Adam([{"params": model.sumo_e.parameters(), "lr": 1e-3},
                                    {"params": model.sumo_c.parameters(), "lr": 1e-3},
                                    {"params": model.phi.parameters(), "lr": 1e-4},
                                ])
            # Train the model
            best_val_loglikelihood = float("-inf")
            epochs_no_improve = 0
            for epoch in tqdm(range(num_epochs)):
            # for epoch in range(num_epochs):
                optimizer.zero_grad()
                logloss = model(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train, max_iter = 10000)
                (-logloss).backward()
                optimizer.step()

                if epoch % 10 == 0:
                    val_loglikelihood = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter = 1000)
                    if val_loglikelihood > (best_val_loglikelihood + 1):
                        best_val_loglikelihood = val_loglikelihood
                        epochs_no_improve = 0
                        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),"loss": best_val_loglikelihood,
                                    }, "/home/DCSurvival/checkpoints/ours_linear_"+copula_form + "_" +str(theta_true)+".pth")
                    else:
                        if val_loglikelihood > best_val_loglikelihood:
                            best_val_loglikelihood = val_loglikelihood
                        epochs_no_improve = epochs_no_improve + 10
                # Early stopping condition
                if epochs_no_improve == early_stop_epochs:
                    # print('Early stopping triggered at epoch: %s' % epoch)
                    break
            # load the best model
            checkpoint = torch.load("/home/DCSurvival/checkpoints/ours_linear_"+copula_form + "_" +str(theta_true)+".pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            # calculate survival_l1 based on ground truth survival function
            steps = np.linspace(y_test.min(), y_test.max(), 1000)
            performance = surv_diff(truth_model, model, X_test, steps)
            survival_l1.append(performance)
            print(epoch, performance)
        print("theta_true = ", theta_true, "survival_l1 = ", np.mean(survival_l1), "+-", np.std(survival_l1))

if __name__ == "__main__":
    main()
