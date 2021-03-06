from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from torch.autograd import grad
from tqdm import tqdm


def listify(A):
    return [a for a in A.flatten()]


def xavier_init(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)


def plot_loss(losses):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(f"loss terms")

    for loss_name, loss in losses.items():
        ax.plot(loss, label=loss_name)

    ax.legend()
    ax.set_xlabel("epoch")


def l_fun(t):
    return np.sin(t) + 2


def f(t, y):
    θ, ω = y
    g = 1.0
    l = l_fun(t)

    dω = -(g / l) * np.sin(θ)
    dθ = ω

    return dθ, dω


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_epochs", default=6000, type=int)
    parser.add_argument("--t_start", default=0.0, type=float)
    parser.add_argument("--t_end", type=float, default=np.pi * 4)
    args = parser.parse_args()

    device = args.device
    n_epochs = args.n_epochs

    """
    generate data

    """
    y0 = [np.pi / 4, 0]
    step_size = 0.01

    t_start = args.t_start
    t_end = args.t_end

    t_eval = np.arange(t_start, t_end, step_size)

    y_true = solve_ivp(
        f, t_span=(t_start, t_end), t_eval=t_eval, y0=y0, method="RK45"
    ).y
    θ, ω = y_true
    l = torch.tensor(l_fun(t_eval)).to(device)

    """
    network
    
    """
    nn_hidden = (
        nn.Sequential(
            nn.Linear(1, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 2),
        )
        .double()
        .to(device)
    )

    xavier_init(nn_hidden)
    optimizer = torch.optim.Adam(nn_hidden.parameters())

    t_train = torch.tensor(t_eval, requires_grad=True).to(device)
    y_train = torch.tensor(y_true).to(device)

    subsample_every = int(2.5 / step_size)
    losses = defaultdict(list)

    """
    training
    
    """
    for _ in tqdm(range(n_epochs), "training hidden physics model"):
        θ_pred, l_pred = nn_hidden(t_train[..., None]).T

        ω_pred = grad(
            listify(θ_pred),
            t_train,
            only_inputs=True,
            retain_graph=True,
            create_graph=True,
        )[0]

        dω_pred = grad(
            listify(ω_pred),
            t_train,
            only_inputs=True,
            retain_graph=True,
            create_graph=True,
        )[0]

        dω_eq = -(1.0 / l_pred) * torch.sin(θ_pred)

        y_pred = torch.column_stack((θ_pred, ω_pred)).T

        loss_collocation = F.mse_loss(
            y_pred[:, ::subsample_every], y_train[:, ::subsample_every]
        )
        loss_hidden = F.mse_loss(dω_pred, dω_eq)
        loss_length = F.mse_loss(l_pred, l)

        loss = loss_collocation + loss_hidden

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses["collocation"].append(loss_collocation.item())
        losses["hidden"].append(loss_hidden.item())

    plot_loss(losses)

    predicted = {
        "θ(t)": θ_pred.detach().cpu().flatten(),
        "ω(t)": ω_pred.detach().cpu().flatten(),
        "l(t)": l_pred.detach().cpu().flatten(),
    }
    true = {
        "θ(t)": θ,
        "ω(t)": ω,
        "l(t)": l.detach().cpu(),
    }

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True)
    fig.canvas.manager.set_window_title(f"states")

    ax0.set_ylabel("θ(t)")
    ax0.plot(t_eval, θ, c="black", label="true")
    ax0.plot(
        t_eval,
        θ_pred.detach().cpu().flatten(),
        c="b",
        linestyle="--",
        label="predicted",
    )
    ax0.scatter(
        t_eval[::subsample_every],
        θ[::subsample_every],
        c="black",
        linestyle="None",
        label="collocation point",
    )

    ax1.set_ylabel("ω(t)")
    ax1.plot(t_eval, ω, c="black", label="true")
    ax1.plot(
        t_eval,
        ω_pred.detach().cpu().flatten(),
        c="r",
        linestyle="--",
        label="predicted",
    )
    ax1.scatter(
        t_eval[::subsample_every],
        ω[::subsample_every],
        c="black",
        linestyle="None",
        label="collocation point",
    )

    ax2.set_ylabel("l(t)")
    ax2.set_xlabel("t")
    ax2.plot(t_eval, l, c="black", label="true")
    ax2.plot(
        t_eval,
        l_pred.detach().cpu().flatten(),
        c="g",
        linestyle="--",
        label="predicted",
    )
    # skip drawing misleading collocation points, since none are used for the pendulum length

    ax1.legend()
    plt.tight_layout()

    plt.show()
