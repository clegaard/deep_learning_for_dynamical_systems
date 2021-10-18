from argparse import ArgumentParser
from collections import defaultdict
from math import ceil, sin
from math import floor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.modules.container import T

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["vanilla", "autodiff", "pinn"],
        default="vanilla",
    )
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--n_layers", default=5, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_epochs", default=2000, type=int)
    parser.add_argument("--t_start", default=0.0, type=float)
    parser.add_argument("--t_end", type=float, default=np.pi * 4)
    args = parser.parse_args()

    ############### Setup Experiment ###############
    y0 = [np.pi / 4, 0]
    step_size = 0.01
    t_start = args.t_start
    t_end = args.t_end
    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end, step_size)  # 0.0 , 0.01

    g = 1.0  # gravitational acceleration [m/s^2]
    l = 1.0  # length of pendulum [m]

    n_epochs = args.n_epochs
    device = args.device
    subsample_every = int(2.5 / step_size)

    ############### Define Derivative ###############
    def f(t, y):
        θ, ω = y  # state variables go in
        g = 1.0
        l = 1.0
        dω = -(g / l) * np.sin(θ)
        dθ = ω  # special case (common for mechanical systems), the state variable ω is per definition dθ

        return dθ, dω  # derivatives of state variables go out

    ############### Solve ODE ###############

    res = solve_ivp(f, t_span, t_eval=t_eval, y0=y0, method="RK45")

    ############### Plot ###############

    def plot_colored(ax, x, y, c, cmap=plt.cm.jet, steps=10, **kwargs):
        a = c.size
        c = np.asarray(c)
        c -= c.min()
        c = c / c.max()
        it = 0
        while it < c.size - steps:
            x_segm = x[it : it + steps + 1]
            y_segm = y[it : it + steps + 1]
            c_segm = cmap(c[it + steps // 2])
            ax.plot(x_segm, y_segm, c=c_segm, **kwargs)
            it += steps

    θ, ω = res.y

    # train

    losses = defaultdict(lambda: defaultdict(list))

    def xavier_init(module):
        for m in module.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

    def construct_network(input_dim, output_dim, hidden_dim, hidden_layers):

        layers = [nn.Linear(input_dim, hidden_dim), nn.Softplus()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Softplus()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        net = nn.Sequential(*layers).double().to(device)
        xavier_init(net)
        return net

    hidden_dim = args.hidden_dim
    hidden_layers = args.n_layers

    # train.vanilla
    nn_vanilla = construct_network(1, 2, hidden_dim, hidden_layers)

    y_train = torch.tensor(res.y[:, ::subsample_every]).to(device)
    t_train = torch.tensor(t_eval[::subsample_every], requires_grad=True).to(device)
    # losses["vanilla"] = {"collocation": []}
    opt_vanilla = torch.optim.Adam(nn_vanilla.parameters())

    for epoch in tqdm(range(n_epochs), desc="vanilla: training epoch"):
        out = nn_vanilla(t_train.unsqueeze(-1)).T

        loss_collocation = F.mse_loss(out, y_train)

        loss_collocation.backward()
        opt_vanilla.step()
        nn_vanilla.zero_grad()
        losses["vanilla"]["collocation"].append(loss_collocation.item())

    # train.autodiff
    nn_autodiff = construct_network(1, 1, hidden_dim, hidden_layers)

    # losses["autodiff"] = {"collocation": []}
    # `torch.autograd.grad` supports only lists of scalar values (e.g. single evaluation of network).
    # however the function accepts a list of these.
    def listify(A):
        return [a for a in A.flatten()]

    opt_autodiff = torch.optim.Adam(nn_autodiff.parameters())

    for epoch in tqdm(range(n_epochs), desc="autodiff: training epoch"):
        θ_pred = nn_autodiff(t_train.unsqueeze(-1)).T

        θ_listed = listify(θ_pred)

        # [0] since we differentiate with respect to an "single input",
        # which is coincidentially a tensor.
        # in this case  ω ≜ dθ
        ω_pred = grad(
            θ_listed, t_train, only_inputs=True, retain_graph=True, create_graph=True
        )[0].unsqueeze(0)

        θω = torch.cat((θ_pred, ω_pred), dim=0)

        loss_collocation = F.mse_loss(θω, y_train)
        # loss_collocation = F.mse_loss(θ_pred, y_train[:1])
        # loss_collocation = F.mse_loss(ω_pred.squeeze(0), y_train[1])

        loss_collocation.backward()

        # sanity check
        max_grad = next(nn_autodiff.modules())[0].weight.grad.max()
        assert (
            max_grad != 0.0
        ), "maximal gradient of first layer was zero, something is up!"

        opt_autodiff.step()
        nn_autodiff.zero_grad()

        losses["autodiff"]["collocation"].append(loss_collocation.item())

    # train.autodiff+equations
    nn_pinn = construct_network(1, 1, hidden_dim, hidden_layers)

    opt_pinn = torch.optim.Adam(nn_pinn.parameters())
    t_train_dense = torch.tensor(t_eval, requires_grad=True).to(device)
    losses_pinn = {"collocation": [], "equation": []}
    for epoch in tqdm(range(n_epochs), desc="pinn: training epoch"):
        θ_pred = nn_pinn(t_train_dense.unsqueeze(-1)).T
        θ_listed = listify(θ_pred)

        ω_pred = grad(
            θ_listed,
            t_train_dense,
            only_inputs=True,
            retain_graph=True,
            create_graph=True,
        )[0].unsqueeze(0)
        ω_listed = listify(ω_pred)
        dω_pred = grad(
            ω_listed,
            t_train_dense,
            only_inputs=True,
            retain_graph=True,
            create_graph=True,
        )[0].unsqueeze(0)

        θω_dense = torch.cat((θ_pred, ω_pred), dim=0)

        # collocation loss is defined for sparse set of points
        θω = θω_dense[:, ::subsample_every]
        loss_collocation = F.mse_loss(θω, y_train)

        # equation based loss is defined for dense samples
        dω_eq = -torch.sin(θ_pred)
        loss_equation = F.mse_loss(dω_pred, dω_eq)

        loss_total = loss_collocation + loss_equation
        loss_total.backward()
        # next(net.modules())[0].weight.grad => this gives you gradients of the loss

        # sanity check
        max_grad = next(nn_pinn.modules())[0].weight.grad.max()
        assert (
            max_grad != 0.0
        ), "maximal gradient of first layer was zero, something is up!"

        opt_pinn.step()
        nn_pinn.zero_grad()

        losses["pinn"]["collocation"].append(loss_collocation.item())
        losses["pinn"]["equation"].append(loss_equation.item())

    # evaluate

    t = torch.tensor(t_eval, device=device, requires_grad=True)

    # vanilla
    θω_vanilla = nn_vanilla(t.unsqueeze(-1)).detach().detach().cpu().T

    # autodiff
    θ_autodiff = nn_autodiff(t.unsqueeze(-1)).T
    θ_autodiff_listed = listify(θ_autodiff)
    ω_autodiff = grad(θ_autodiff_listed, t, only_inputs=True)[0].unsqueeze(0)
    θω_autodiff = torch.cat((θ_autodiff, ω_autodiff), dim=0).detach().cpu()

    # pinn
    θ_pinn = nn_pinn(t.unsqueeze(-1)).T
    θ_pinn_listed = listify(θ_pinn)
    ω_pinn = grad(θ_pinn_listed, t, only_inputs=True)[0].unsqueeze(0)
    θω_pinn = torch.cat((θ_pinn, ω_pinn), dim=0).detach().cpu()

    # plot results

    for title, θω_pred in zip(
        ["vanilla", "autodiff", "pinn"],
        [θω_vanilla, θω_autodiff, θω_pinn],
    ):

        ω_numerical = np.diff(θω_pred[:1]) / step_size
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.canvas.set_window_title(title)
        ax1.set_ylabel("θ(t)")
        ax2.set_ylabel("ω(t)")
        ax2.set_xlabel("t")

        ax1.plot(t_eval, θ, c="black", label="true")
        ax1.plot(t_eval, θω_pred[0], c="b", linestyle="--", label="predicted")

        ax2.plot(t_eval, ω, c="black", label="true")
        ax2.plot(t_eval, θω_pred[1], c="r", linestyle="--", label="predicted")
        ax2.plot(
            t_eval[1:],
            ω_numerical.T,
            c="r",
            linestyle="dotted",
            label="numerical",
        )

        ax1.scatter(
            t_eval[::subsample_every],
            res.y[:, ::subsample_every][0],
            c="black",
            linestyle="None",
            label="collocation point",
        )
        ax2.scatter(
            t_eval[::subsample_every],
            res.y[:, ::subsample_every][1],
            c="black",
            linestyle="None",
            label="collocation point",
        )
        ax2.legend()
        plt.tight_layout()

    # plot loss functions as function of training steps
    for network_name, losses in losses.items():
        fig, ax = plt.subplots()
        fig.canvas.set_window_title(f"loss terms '{network_name}'")

        for loss_name, loss in losses.items():
            ax.plot(loss, label=loss_name)

        ax.legend()
        ax.set_xlabel("epoch")

    # plt.figure()
    # plot_colored(θ, ω, t_eval)
    # plt.xlabel("angle")
    # plt.ylabel("velocity")
    # plt.title("polar plot")
    import matplotlib as mpl

    fig = plt.figure()
    x, y = np.meshgrid(
        np.arange(-np.pi, np.pi, 0.01),
        np.arange(-np.pi, np.pi, 0.01),
    )
    dθ, dω = f(None, (x, y))
    plt.streamplot(x, y, dθ, dω, density=2)
    plt.xlabel("θ")
    plt.ylabel("ω")
    # plt.title("phase portrait")

    ax = plt.gca()
    plot_colored(ax, θ, ω, t_eval)
    cmap = plt.cm.jet
    norm = mpl.colors.Normalize(vmin=t_eval.min(), vmax=t_eval.max())
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    # draw initial state
    plt.scatter(θ[0], ω[0], label="$y_0$", marker="*", c="g", s=200, zorder=100)
    plt.legend(loc="upper right")
    ax.set_aspect(1)

    plt.show()
