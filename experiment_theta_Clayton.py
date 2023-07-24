import os, sys
# add path to import py file from parent directory


from synthetic_dgp import linear_dgp

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os
import numpy as np
# sys.path.append(os.path.abspath('../'))
from dirac_phi import DiracPhi
from survival import SurvivalCopula
from survival import sample


device = torch.device("cuda:0")
batch_size = 10000
num_epochs = 20000
copula_form = 'Clayton'


def main():
    for theta_true in [2,4,6,8,10,15,20]:
        X, observed_time, event_indicator, _, _ = linear_dgp( copula=copula_form, theta=theta_true)
        times_tensor = torch.tensor(observed_time, dtype=torch.float64).to(device)
        event_indicator_tensor = torch.tensor(event_indicator, dtype=torch.float64).to(device)
        covariate_tensor = torch.tensor(X, dtype=torch.float64).to(device)
        train_data = TensorDataset(covariate_tensor[0:20000], times_tensor[0:20000], event_indicator_tensor[0:20000])
        val_data = TensorDataset(covariate_tensor[20000:], times_tensor[20000:], event_indicator_tensor[20000:])


        train_loader = DataLoader(
            train_data, batch_size= batch_size, shuffle=True)

        val_loader = DataLoader(
            val_data, batch_size= batch_size, shuffle=True)

        # Early stopping
        best_val_loglikelihood = float('-inf')
        epochs_no_improve = 0
        early_stop_epochs = 5000

        # Parameters for ACNet
        depth = 2
        widths = [100, 100]
        lc_w_range = (0, 1.0)
        shift_w_range = (0., 2.0)

        phi = DiracPhi(depth, widths, lc_w_range, shift_w_range, device, tol = 1e-10).to(device)
        model = SurvivalCopula(phi, device = device, num_features=10, tol=1e-10).to(device)
        # optimizer = get_optim(optim_name, net, optim_args)

        optimizer_event = optim.Adam([{"params": [model.scale_t], "lr": 0.01},
                                    {"params": [model.shape_t], "lr": 0.01},
                                    {"params": model.net_t.parameters(), "lr": 0.01},
                                ])
        optimizer_censoring = optim.Adam([{"params": [model.scale_c], "lr": 0.01},
                                    {"params": [model.shape_c], "lr": 0.01},
                                    {"params": model.net_c.parameters(), "lr": 0.01},
                                ])
        optimizer_copula = optim.Adam([
                                    {"params": model.phi.parameters(), "lr": 0.01},
                                ])

        train_loss_per_epoch = []
        print("Start training!")
        for epoch in range(num_epochs):
            loss_per_minibatch = []
            for i, (x , t, c) in enumerate(train_loader, 0):
                optimizer_copula.zero_grad()
                optimizer_event.zero_grad()
                optimizer_censoring.zero_grad()

                p = model(x, t, c, max_iter = 10000)
                logloss = -p
                logloss.backward() 
                scalar_loss = (logloss/p.numel()).detach().cpu().numpy().item()

                optimizer_censoring.step()
                optimizer_event.step()
                optimizer_copula.step()
                
                loss_per_minibatch.append(scalar_loss)
            train_loss_per_epoch.append(np.mean(loss_per_minibatch))
            if epoch % 100 == 0:
                # Check if validation loglikelihood has improved
                for i, (x_val, t_val, c_val) in enumerate(val_loader, 0):
                    p_val = model(x_val, t_val, c_val, max_iter = 10000)
                    val_loglikelihood = p_val
                # print('Validation log-likelihood at epoch %s: %s' % (epoch, val_loglikelihood.cpu().detach().numpy().item()))
                if val_loglikelihood > best_val_loglikelihood:
                    best_val_loglikelihood = val_loglikelihood
                    epochs_no_improve = 0
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': best_val_loglikelihood,
                    }, '/home/weijia/code/SurvivalACNet/checkpoint_experiment_theta_clayton.pth')
            
            else:
                epochs_no_improve += 100

            # Early stopping condition
            if epochs_no_improve == early_stop_epochs:
                print('Early stopping triggered at epoch: %s' % epoch)
                break
            checkpoint = torch.load('/home/weijia/code/SurvivalACNet/checkpoint_experiment_theta_clayton.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
        print(model.shape_t.item())
        print(model.shape_c.item())
if __name__ == '__main__':
    main()