import torch

from dcsurvival.survival import MixExpPhi, MixExpPhi2FixedSlope, PhiInv, SurvivalCopula


# TODO: move to pytest
def test_grad_of_phi() -> None:
    phi_net = MixExpPhi()
    PhiInv(phi_net)
    query = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [1., 1., 1.]]).requires_grad_(True)

    gradcheck(phi_net, (query), eps=1e-9)
    gradgradcheck(phi_net, (query,), eps=1e-9)


def test_grad_y_of_inverse() -> None:
    phi_net = MixExpPhi()
    phi_inv = PhiInv(phi_net)
    query = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.25, 0.7]]).requires_grad_(True)

    gradcheck(phi_inv, (query, ), eps=1e-10)
    gradgradcheck(phi_inv, (query, ), eps=1e-10)


def test_grad_w_of_inverse() -> None:
    phi_net = MixExpPhi2FixedSlope()
    phi_inv = PhiInv(phi_net)

    eps = 1e-8
    new_phi_inv = copy.deepcopy(phi_inv)

    # Jitter weights in new_phi.
    new_phi_inv.phi.mix.data = phi_inv.phi.mix.data + eps

    query = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.99, 0.99, 0.99]]).requires_grad_(True)
    old_value = phi_inv(query).sum()
    old_value.backward()
    anal_grad = phi_inv.phi.mix.grad
    new_value = new_phi_inv(query).sum()
    num_grad = (new_value-old_value)/eps

    print("gradient of weights (anal)", anal_grad)
    print("gradient of weights (num)", num_grad)


def test_grad_y_of_pdf() -> None:
    phi_net = MixExpPhi()
    query = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.99, 0.99, 0.99]]).requires_grad_(True)
    cop = SurvivalCopula(phi_net)
    def f(y): return cop(y, mode="pdf")
    gradcheck(f, (query, ), eps=1e-8)
    # This fails sometimes if rtol is too low..?
    gradgradcheck(f, (query, ), eps=1e-8, atol=1e-6, rtol=1e-2)


# FIXME: what should the "Copula" class be in updated test?
# def plot_pdf_and_cdf_over_grid() -> None:
#     phi_net = MixExpPhi()
#     cop = Copula(phi_net)

#     n = 500
#     x1 = np.linspace(0.001, 1, n)
#     x2 = np.linspace(0.001, 1, n)
#     xv1, xv2 = np.meshgrid(x1, x2)
#     xv1_tensor = torch.tensor(xv1.flatten())
#     xv2_tensor = torch.tensor(xv2.flatten())
#     query = torch.stack((xv1_tensor, xv2_tensor),
#                         ).double().t().requires_grad_(True)
#     cdf = cop(query, mode="cdf")
#     pdf = cop(query, mode="pdf")

#     assert abs(pdf.mean().detach().numpy().sum() -
#                1) < 1e-6, "Mean of pdf over grid should be 1"
#     assert abs(cdf[-1].detach().numpy().sum() -
#                1) < 1e-6, "CDF at (1..1) should be should be 1"


# def plot_cond_cdf() -> None:
#     phi_net = MixExpPhi()
#     cop = Copula(phi_net)

#     n = 500
#     xv2 = np.linspace(0.001, 1, n)
#     xv2_tensor = torch.tensor(xv2.flatten())
#     xv1_tensor = 0.9 * torch.ones_like(xv2_tensor)
#     x = torch.stack([xv1_tensor, xv2_tensor], dim=1).requires_grad_(True)
#     cond_cdf = cop(x, mode="cond_cdf", others={"cond_dims": [0]})

#     plt.figure()
#     plt.plot(cond_cdf.detach().numpy())
#     plt.title("Conditional CDF")
#     plt.draw()
#     plt.pause(0.01)


# def plot_samples() -> None:
#     phi_net = MixExpPhi()
#     cop = Copula(phi_net)

#     s = sample(cop, 2, 2000, seed=142857)
#     s_np = s.detach().numpy()

#     plt.figure()
#     plt.scatter(s_np[:, 0], s_np[:, 1])
#     plt.title("Sampled points from Copula")
#     plt.draw()
#     plt.pause(0.01)


# def plot_loss_surface() -> None:
#     phi_net = MixExpPhi2FixedSlope()
#     cop = Copula(phi_net)

#     s = sample(cop, 2, 2000, seed=142857)
#     s.detach().numpy()

#     losses = []
#     x = np.linspace(-1e-2, 1e-2, 1000)
#     for SS in x:
#         new_cop = copy.deepcopy(cop)
#         new_cop.phi.mix.data = cop.phi.mix.data + SS

#         loss = -torch.log(new_cop(s, mode="pdf")).sum()
#         losses.append(loss.detach().numpy().sum())

#     plt.figure()
#     plt.plot(x, losses)
#     plt.title("Loss surface")
#     plt.draw()
#     plt.pause(0.01)


# def test_training(test_grad_w=False) -> None:
#     gen_phi_net = MixExpPhi()
#     PhiInv(gen_phi_net)
#     gen_cop = Copula(gen_phi_net)

#     s = sample(gen_cop, 2, 2000, seed=142857)
#     s.detach().numpy()

#     ideal_loss = -torch.log(gen_cop(s, mode="pdf")).sum()

#     train_cop = copy.deepcopy(gen_cop)
#     train_cop.phi.mix.data *= 1.5
#     train_cop.phi.slope.data *= 1.5
#     print("Initial loss", ideal_loss)
#     optimizer = optim.Adam(train_cop.parameters(), lr=1e-3)

#     def numerical_grad(cop) -> None:
#         # Take gradients w.r.t to the first mixing parameter
#         print("Analytic gradients:", cop.phi.mix.grad[0])

#         _old_cop, new_cop = copy.deepcopy(cop), copy.deepcopy(cop)
#         # First order approximation of gradient of weights
#         eps = 1e-6
#         new_cop.phi.mix.data[0] = cop.phi.mix.data[0] + eps
#         x2 = -torch.log(new_cop(s, mode="pdf")).sum()
#         x1 = -torch.log(cop(s, mode="pdf")).sum()

#         first_order_approximate = (x2-x1)/eps
#         print("First order approx.:", first_order_approximate)

#     for iter in range(100000):
#         optimizer.zero_grad()
#         loss = -torch.log(train_cop(s, mode="pdf")).sum()
#         loss.backward()
#         print("iter", iter, ":", loss, "ideal loss:", ideal_loss)
#         if test_grad_w:
#             numerical_grad(train_cop)
#         optimizer.step()

if __name__ == "__main__":
    import copy

    from torch.autograd import gradcheck, gradgradcheck

    torch.set_default_tensor_type(torch.DoubleTensor)

    test_grad_of_phi()
    test_grad_y_of_inverse()
    test_grad_w_of_inverse()
    test_grad_y_of_pdf()

    ## Uncomment for rudimentary training. NOTE: very slow and unrealistic.
    # plot_pdf_and_cdf_over_grid()
    # plot_cond_cdf()
    # plot_samples()
    # plot_loss_surface()
    # test_training()
