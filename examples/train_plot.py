
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival, sample
from dcsurvival.synthetic_dgp import linear_dgp

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_num_threads(24)

device = torch.device("cuda:0")
batch_size = 20000
num_epochs = 10000
copula_form = "Frank"
sample_size = 30000
val_size = 10000
seed = 142857
rng = np.random.default_rng(seed)

def main() -> None:
    for theta_true in [2]:
        X, observed_time, event_indicator, _, _, _ = linear_dgp( copula_name=copula_form, covariate_dim=10, theta=theta_true, sample_size=sample_size, rng=rng)
        times_tensor = torch.tensor(observed_time, dtype=torch.float64).to(device)
        event_indicator_tensor = torch.tensor(event_indicator, dtype=torch.float64).to(device)
        covariate_tensor = torch.tensor(X, dtype=torch.float64).to(device)
        train_data = TensorDataset(covariate_tensor[0:sample_size-val_size], times_tensor[0:sample_size-val_size], event_indicator_tensor[0:sample_size-val_size])
        val_data = TensorDataset(covariate_tensor[sample_size-val_size:], times_tensor[sample_size-val_size:], event_indicator_tensor[sample_size-val_size:])

        train_loader = DataLoader(train_data, batch_size= batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size= batch_size, shuffle=True)

        # Early stopping
        best_val_loglikelihood = float("-inf")
        epochs_no_improve = 0
        early_stop_epochs = 100

        # Parameters for ACNet
        depth = 2
        widths = [100, 100]
        lc_w_range = (0, 1.0)
        shift_w_range = (0., 2.0)

        phi = DiracPhi(depth, widths, lc_w_range, shift_w_range, device, tol = 1e-10).to(device)
        model = DCSurvival(phi, device = device, num_features=10, tol=1e-10).to(device)
        # separately optimize copula and survival parameters is sometimes helpful, but not necessary
        optimizer_survival = optim.Adam([{"params": model.sumo_e.parameters(), "lr": 0.001},
                                    {"params": model.sumo_c.parameters(), "lr": 0.001},
                                ])
        optimizer_copula = optim.SGD([{"params": model.phi.parameters(), "lr": 0.0005}])

        train_loss_per_epoch = []
        print("Start training!")
        for epoch in range(num_epochs):
            loss_per_minibatch = []
            for _i, (x , t, c) in enumerate(train_loader, 0):
                optimizer_copula.zero_grad()
                optimizer_survival.zero_grad()

                p = model(x, t, c, max_iter = 1000)
                logloss = -p
                logloss.backward()
                scalar_loss = (logloss/p.numel()).detach().cpu().numpy().item()

                optimizer_survival.step()

                if epoch > 200:
                    optimizer_copula.step()
                # optimizer_censoring.step()
                # optimizer.step()

                loss_per_minibatch.append(scalar_loss/batch_size)
            train_loss_per_epoch.append(np.mean(loss_per_minibatch))
            if epoch % 1 == 0:
                print(f"Training likilihood at epoch {epoch}: {-train_loss_per_epoch[-1]:.5f}")
                # Check if validation loglikelihood has improved
                for _i, (x_val, t_val, c_val) in enumerate(val_loader, 0):
                    val_loglikelihood = model(x_val, t_val, c_val, max_iter = 10000)/val_size
                print(f"Validation log-likelihood at epoch {epoch}: {val_loglikelihood.cpu().detach().numpy().item()}")
                if val_loglikelihood > best_val_loglikelihood:
                    best_val_loglikelihood = val_loglikelihood
                    epochs_no_improve = 0
                    torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": best_val_loglikelihood,
                    }, "/home/DCSurvival/checkpoints/checkpoint_experiment_"+copula_form + "_" +str(theta_true)+".pth")

                else:
                    epochs_no_improve += 1
                # Early stopping condition
                if epochs_no_improve == early_stop_epochs:
                    print("Early stopping triggered at epoch: %s" % epoch)
                    break
            # Plot Samples from the learned copula
            if epoch % 200 == 0:
                print("Scatter sampling")
                samples = sample(model, 2, sample_size, device =  device)
                plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
                plt.savefig("/home/DCSurvival/sample_figs/"+copula_form+"/"+str(theta_true)+"/epoch%s.png" %
                            (epoch))
                plt.clf()

        checkpoint = torch.load("/home/DCSurvival/checkpoints/checkpoint_experiment_"+copula_form + "_" +str(theta_true)+".pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        samples =  sample(model, 2, sample_size, device =  device)
        plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s = 15)
        plt.savefig("/home/DCSurvival/sample_figs/"+copula_form+"/"+str(theta_true)+"/best_epoch.png")
        plt.clf()


if __name__ == "__main__":
    main()
