from argparse import ArgumentParser
from itertools import chain

import torch.nn as nn
from torch.nn.modules.rnn import RNN
from torch.optim import Adam
from tqdm import tqdm
from pyDOE import lhs
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
import numpy as np

import torch
from torchdyn.numerics import odeint
from torchdyn.numerics.solvers import SolverTemplate
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from visualization import get_meshgrid, plot_colored


class DirectSolver(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = "fixed"

    def step(self, f, x, t, dt, k1=None):

        x_sol = f(t, x)
        return None, x_sol, None


class ResnetSolver(SolverTemplate):
    def __init__(self, step_size=None, dtype=torch.float32):
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = "fixed"

        self.step_size = 1 if step_size is None else step_size

    def step(self, f, x, t, dt, k1=None):
        x_sol = x + f(t, x) * self.step_size
        return None, x_sol, None


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--solver", choices=["direct", "resnet", "euler", "rk4"], default="euler"
    )
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_epochs", default=0, type=int)
    parser.add_argument("--n_traj_train", default=100, type=int)
    parser.add_argument("--n_traj_validate", default=10, type=int)
    parser.add_argument("--t_start_train", default=0.0, type=float)
    parser.add_argument("--t_end_train", default=1.0, type=float)
    parser.add_argument("--t_start_validate", default=0.0, type=float)
    parser.add_argument("--t_end_validate", default=4 * np.pi, type=float)
    parser.add_argument("--step_size_train", default=0.01, type=float)
    parser.add_argument("--step_size_validate", default=0.01, type=float)
    parser.add_argument("--noise_std", default=0.000001, type=float)
    parser.add_argument("--domain_train", default=1.0, type=float)
    parser.add_argument("--domain_validate", default=1.0, type=float)
    parser.add_argument("--latent_dim", default=4, type=int)
    parser.add_argument("--scatter", default=False, type=bool)
    args = parser.parse_args()

    # generate data

    def f(t, x):
        θ = x[..., 0]
        ω = x[..., 1]

        dθ = ω
        dω = -torch.sin(θ)

        return torch.stack((dθ, dω), dim=-1)

    domain_draw_factor = 1.3

    noise_log_var = 2 * torch.log(torch.tensor(args.noise_std))

    latent_dim = args.latent_dim
    domain_train = args.domain_train
    domain_validate = args.domain_validate

    x0_train = (
        torch.tensor(lhs(2, args.n_traj_train), device=args.device) * 2 - 1
    ) * domain_train
    x0_validate = (
        torch.tensor(lhs(2, args.n_traj_validate), device=args.device) * 2 - 1
    ) * domain_validate
    x0_grid = get_meshgrid(step_per_axis=0.01, domain=domain_validate)
    x0_example = torch.tensor((0.6, 0)).double().unsqueeze(0).to(args.device)

    step_size_train = args.step_size_train
    ε = 1e-10
    t_span_train = torch.arange(
        args.t_start_train, args.t_end_train + ε, step_size_train
    )
    t_span_validate = torch.arange(
        args.t_start_validate,
        args.t_end_validate + ε,
        args.step_size_validate,
    )

    if args.solver.lower() == "direct":
        solver = DirectSolver()
    elif args.solver.lower() == "resnet":
        solver = ResnetSolver()
    else:
        solver = args.solver

    _, x_train = odeint(f, x0_train, t_span_train, solver="rk4")
    x_train = x_train + torch.rand_like(x_train) * args.noise_std

    _, x_validate = odeint(f, x0_validate, t_span_train, solver="rk4")
    _, x_example = odeint(f, x0_example, t_span_validate, solver="rk4")

    ##################### model ##########################
    device = args.device

    class Encoder(nn.Module):
        def __init__(self, input_size, hidden_size) -> None:
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size)
            self.fc = nn.Linear(hidden_size, hidden_size * 2)

        def forward(self, x):
            x_flipped = torch.flip(x, (0,))
            _, h = self.rnn(x_flipped)
            z0 = self.fc(h)
            return z0

    class LatentODE(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.encoder = Encoder(2, 4)

            layers = []
            layers.append(nn.Linear(4, args.hidden_dim))
            for _ in range(args.n_layers):
                layers.append(nn.Linear(args.hidden_dim, args.hidden_dim))
                layers.append(nn.Softplus())

            layers.append(nn.Linear(args.hidden_dim, 4))
            self.dynamics = nn.Sequential(*layers)

            self.decoder = (
                nn.Sequential(nn.Linear(latent_dim, 20), nn.ReLU(), nn.Linear(20, 2))
                .to(device)
                .double()
            )

        def forward(self, t, x):
            z = self.encoder(x)[-1]
            qz0_mean, qz0_logvar = z[:, :latent_dim], z[:, latent_dim:]
            qz0_std = torch.exp(0.5 * qz0_logvar)
            ε = torch.randn_like(qz0_mean)
            z0 = qz0_mean + ε * qz0_std  # TODO
            # z0 = qz0_mean  # TODO

            _, z_pred = odeint(lambda t, z: self.dynamics(z), z0, t, solver=solver)

            x_pred = self.decoder(z_pred)

            return qz0_mean, qz0_logvar, x_pred

    net = LatentODE().to(args.device).double()

    for m in chain(net.modules()):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    # optimizer

    opt = Adam(net.parameters())

    # train
    losses = []

    def log_normal_pdf(x, mean, logvar):
        px = -0.5 * (
            torch.log(torch.tensor(np.pi))
            + logvar
            + (x - mean) ** 2.0 / torch.exp(logvar)
        )
        torch.tensor
        return px.sum((0, 2))

    def normal_kl(μ1, log_var1, μ2, log_var2):
        v1 = torch.exp(log_var1)
        v2 = torch.exp(log_var2)
        lstd1 = log_var1 / 2.0
        lstd2 = log_var2 / 2.0

        kl = lstd2 - lstd1 + ((v1 + (μ1 - μ2) ** 2.0) / (2.0 * v2)) - 0.5
        return kl.sum(-1)

    for _ in tqdm(range(args.n_epochs)):

        # x_pred_train = decoder(z_pred_train)
        qz0_mean, qz0_logvar, x_pred_train = net(t_span_train, x_train)

        log_px = log_normal_pdf(x_pred_train, x_train, noise_log_var)
        log_normal_kl = normal_kl(
            qz0_mean,
            qz0_logvar,
            torch.zeros_like(qz0_mean),
            torch.zeros_like(qz0_logvar),
        )

        loss = torch.mean(log_normal_kl - log_px, dim=0)
        # loss = mse_loss(x_pred_train, x_train)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    _, _, x_pred_train = net(t_span_train, x_train)
    _, _, x_pred_validate = net(t_span_validate, x_validate)

    # _, x_pred_example = odeint(
    #     lambda t, x: dynamics(x), x0_example, t_span_validate, solver=solver
    # )

    # derivatives
    # x0_grid_before = get_meshgrid(step_per_axis=0.01, domain=domain)
    x_derivative = f(None, x0_grid)

    # if args.solver == "direct":

    #     out = dynamics(x0_grid)
    #     # normalize for the step size used during training. If the network is trained with a step-size of 1/100 of a second
    #     # it will predict changes that are 100 times as small as those for 1 second.
    #     x_derivative_pred = (out - x0_grid) / step_size_train

    # elif args.solver == "resnet":
    #     x_derivative_pred = dynamics(x0_grid) / step_size_train
    # else:
    #     x_derivative_pred = dynamics(x0_grid)

    # plot
    x_pred_train = x_pred_train.detach().numpy()
    x_pred_validate = x_pred_validate.detach().numpy()
    # x_pred_example = x_pred_example.detach().numpy()
    # x_derivative_pred = x_derivative_pred.detach().numpy()
    # x_derivative = x_derivative.detach().numpy()
    # x0_grid = x0_grid.detach().numpy()

    # # streamplot
    # density = 1
    # fig, ax = plt.subplots()
    # fig.canvas.manager.set_window_title("stream plot")
    # ode_patch = mpatches.Patch(color="black", label="true")
    # nn_patch = mpatches.Patch(color="blue", label="pred")

    # ax.streamplot(
    #     x0_grid[..., 0],
    #     x0_grid[..., 1],
    #     x_derivative[..., 0],
    #     x_derivative[..., 1],
    #     color="black",
    #     density=density,
    # )

    # ax.streamplot(
    #     x0_grid[..., 0],
    #     x0_grid[..., 1],
    #     x_derivative_pred[..., 0],
    #     x_derivative_pred[..., 1],
    #     color="blue",
    #     density=density,
    # )

    # ax.set_xlabel("θ")
    # ax.set_ylabel("ω")
    # ax.set_xlim(-domain_validate, domain_validate)
    # ax.set_ylim(-domain_validate, domain_validate)

    # ax.legend(handles=[ode_patch, nn_patch])

    # # quiver
    # fig, ax = plt.subplots()
    # fig.canvas.manager.set_window_title("quiver")
    # ax.quiver(
    #     x0_grid[..., 0],
    #     x0_grid[..., 1],
    #     x_derivative[..., 0],
    #     x_derivative[..., 1],
    #     color="black",
    #     angles="xy",
    #     scale_units="xy",
    #     label="true"
    #     # scale=1,
    # )

    # ax.quiver(
    #     x0_grid[..., 0],
    #     x0_grid[..., 1],
    #     x_derivative_pred[..., 0],
    #     x_derivative_pred[..., 1],
    #     color="blue",
    #     angles="xy",
    #     scale_units="xy",
    #     # scale=1,
    #     label="pred",
    # )
    # ax.legend()

    # ax.set_xlabel("θ")
    # ax.set_ylabel("ω")
    # ax.set_ylim(-domain_validate, domain_validate)
    # ax.set_xlim(-domain_validate, domain_validate)

    # phase space, training
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("phase space: training")

    lines_true = LineCollection(
        [x for x in x_train.swapaxes(0, 1)],
        color="black",
        label="true",
    )
    # lines_pred = LineCollection(
    #     [x for x in x_pred_train.swapaxes(0, 1)], color="blue", label="pred"
    # )

    if args.scatter:
        ax.scatter(x_pred_train[..., 0], x_pred_train[..., 1])
    plot_colored(
        fig,
        ax,
        t_span_train,
        x_pred_train,
        label="pred",
        colorbar=True,
        # linestyle="dashed",
    )
    if args.scatter:
        ax.scatter(x_train[..., 0], x_train[..., 1])
    ax.add_collection(lines_true)
    # ax.add_collection(lines_pred)

    ax.set_xlabel("θ")
    ax.set_ylabel("ω")
    ax.set_xlim(-domain_draw_factor * domain_train, domain_draw_factor * domain_train)
    ax.set_ylim(-domain_draw_factor * domain_train, domain_draw_factor * domain_train)
    ax.legend()

    # # phase space, validation
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("phase space: validation")

    lines_true = LineCollection(
        [x for x in x_validate.swapaxes(0, 1)],
        color="black",
        label="true",
    )
    # lines_pred = LineCollection(
    #     [x for x in x_pred_train.swapaxes(0, 1)], color="blue", label="pred"
    # )
    if args.scatter:
        ax.scatter(x_pred_validate[..., 0], x_pred_validate[..., 1])
    plot_colored(
        fig,
        ax,
        t_span_validate,
        x_pred_validate,
        label="pred",
        colorbar=True,
        # linestyle="dashed",
    )

    if args.scatter:
        ax.scatter(x_validate[..., 0], x_validate[..., 1])
    ax.add_collection(lines_true)
    # ax.add_collection(lines_pred)

    ax.set_xlabel("θ")
    ax.set_ylabel("ω")
    ax.set_xlim(
        -domain_draw_factor * domain_validate, domain_draw_factor * domain_validate
    )
    ax.set_ylim(
        -domain_draw_factor * domain_validate, domain_draw_factor * domain_validate
    )
    ax.legend()

    # # time series validation, example (0.6,0)
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # fig.canvas.manager.set_window_title(f"states vs time: validation example")

    # ax1.plot(t_span_validate, x_example[..., 0], color="black")
    # ax1.plot(
    #     t_span_validate,
    #     x_pred_example[..., 0],
    #     linestyle="dashed",
    #     color="blue",
    # )
    # ax2.plot(t_span_validate, x_example[..., 1], label="true", color="black")
    # ax2.plot(
    #     t_span_validate,
    #     x_pred_example[..., 1],
    #     linestyle="dashed",
    #     color="blue",
    #     label="predicted",
    # )
    # ax1.set_ylabel("θ(t)")
    # ax2.set_ylabel("ω(t)")
    # ax2.set_xlabel("t")
    # ax2.legend()

    # show
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")

    plt.show()
