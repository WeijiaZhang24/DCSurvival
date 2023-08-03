import torchtuples as tt
import torch # For building the networks 
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pycox.models import LogisticHazard, CoxPH, CoxCC
from pycox.datasets import metabric

from model.truth_net import Weibull_linear, Weibull_nonlinear
from metrics.metric_pycox import surv_diff
from synthetic_dgp import linear_dgp, nonlinear_dgp
from sklearn.model_selection import train_test_split

torch.set_num_threads(32)
sample_size=30000

method ='CoxCC'
risk = 'nonlinear'
print(method, risk)
def main():
    for theta_true in [2,4,6,8,10,12,14,16,18,20,25,30]:
        survival_l1 = []
        if theta_true==0:
            copula_form='Independent'
        else:
            copula_form='Gumbel'
        print(copula_form)
        for repeat in range(5):
            seed = 142857 + repeat
            rng = np.random.default_rng(seed)   
            if risk=='linear':
                X, observed_time, event_indicator, _, _, beta_e = linear_dgp( copula_name=copula_form, theta=theta_true, sample_size=sample_size, rng=rng, verbose=False)
            elif risk=='nonlinear':
                X, observed_time, event_indicator, _, _ = nonlinear_dgp( copula_name=copula_form, theta=theta_true, sample_size=sample_size, rng=rng, verbose=False)
            # split train test
            X_train, X_test, y_train, y_test, indicator_train, indicator_test = train_test_split(X, observed_time, event_indicator, test_size=0.33, random_state=repeat)
            # split train val
            X_train, X_val, y_train, y_val, indicator_train, indicator_val = train_test_split(X_train, y_train, indicator_train, test_size=0.33,  random_state=repeat)

            y_train_tuple = (y_train, indicator_train)
            y_val_tuple = (y_val, indicator_val)
            val = X_val, y_val_tuple

            in_features = X_train.shape[1]
            num_nodes = [32, 32, 32]
            out_features = 1
            batch_norm = True
            dropout = 0.0
            output_bias = False

            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                        dropout, output_bias=output_bias)
            if method=="DeepSurv":
                model = CoxPH(net, tt.optim.Adam, device=torch.device("cpu"))
            elif method=="CoxCC":
                model = CoxCC(net, tt.optim.Adam, device=torch.device("cpu"))
            model.optimizer.set_lr(0.001)
            epochs = 1024
            callbacks = [tt.callbacks.EarlyStopping(patience=100)]
            verbose = False
            batch_size = 4096
            log = model.fit(X_train, y_train_tuple, batch_size, epochs, callbacks, verbose,
                    val_data=val, val_batch_size=batch_size)
            _ = model.compute_baseline_hazards()
            surv_prediction = model.predict_surv_df(X_test)
            prediction_timepoint = surv_prediction.index
            # define truth model
            if risk == "linear":
                truth_model = Weibull_linear(num_feature= X_test.shape[1], shape = 4, scale = 14, 
                                             device = torch.device("cpu"), coeff = beta_e)
            elif risk == "nonlinear":
                truth_model = Weibull_nonlinear(shape = 4, scale = 17, device = torch.device("cpu"))

            # calculate survival_l1 based on ground truth survival function
            performance = surv_diff(truth_model, surv_prediction, X_test, prediction_timepoint)
            survival_l1.append(performance)
        print("theta_true = ", theta_true, "survival_l1 = ", np.mean(survival_l1), "+-", np.std(survival_l1))

if __name__ == "__main__":
    main()