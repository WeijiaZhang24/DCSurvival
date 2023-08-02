import torchtuples as tt
import torch # For building the networks 
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm

from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
# from sksurv.datasets import load_gbsg2

from model.truth_net import Weibull_linear, Weibull_nonlinear
from metrics.metric_sksurv import surv_diff
from synthetic_dgp import linear_dgp,nonlinear_dgp
from sklearn.model_selection import train_test_split

sample_size=30000
num_threads = 32

risk="linear"
method ='RSF'
print(method)
print(risk)

def main():
    for theta_true in [0]:
        survival_l1 = []
        for repeat in range(5): 
            seed = 142857 + repeat
            rng = np.random.default_rng(seed)   
            if theta_true==0:
                copula_form = "Independent"
            else:
                copula_form = "Clayton"
                print(copula_form)
            if risk=="linear":
                X, observed_time, event_indicator, _, _, beta_e = linear_dgp( copula_name=copula_form, theta=theta_true, sample_size=sample_size, rng=rng, verbose=False)
            elif risk == "nonlinear":
                X, observed_time, event_indicator, _, _ = nonlinear_dgp( copula_name=copula_form, theta=theta_true, sample_size=sample_size, rng=rng, verbose=False)
            # split train test
            X_train, X_test, y_train, y_test, indicator_train, indicator_test = train_test_split(X, observed_time, event_indicator, test_size=0.33, 
                                                                                                 stratify = event_indicator, random_state= repeat)
            # split train val
            # X_train, X_val, y_train, y_val, indicator_train, indicator_val = train_test_split(X_train, y_train, indicator_train, test_size=0.33)
            
            bool_indicator_train =  [bool(x) for x in indicator_train]
            y_train_tuple =  c = np.array(list(zip(bool_indicator_train, y_train)), dtype=[('a', bool), ('b', float)])

            if method=='RSF':
                rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, 
                                           min_samples_leaf=15, n_jobs=num_threads)
                rsf.fit(X_train, y_train_tuple)
                surv_prediction = rsf.predict_survival_function(X_test, return_array=True)
                prediction_timepoint = rsf.unique_times_

            elif method == 'CoxPH':
                CoxPH = CoxPHSurvivalAnalysis()
                CoxPH.fit(X_train, y_train_tuple)
                surv_prediction = CoxPH.predict_survival_function(X_test, return_array=True)
                prediction_timepoint = CoxPH.unique_times_

            # surv_prediction = model.predict_surv_df(X_test)
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