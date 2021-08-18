from argparse import ArgumentParser
from numpy import double
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pyDOE import lhs
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
import numpy as np

import torch
from torchdyn.numerics import odeint
from torchdyn.numerics.solvers import SolverTemplate

from visualization import get_meshgrid


class DirectSolver(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = "fixed"

    def step(self, f, x, t, dt, k1=None):
        if k1 == None:
            k1 = f(t, x)
        x_sol = k1
        return None, x_sol, None


class ResnetSolver(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = "fixed"

    def step(self, f, x, t, dt, k1=None):
        if k1 == None:
            k1 = f(t, x)
        x_sol = x + k1
        return None, x_sol, None


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--solver", choices=["direct", "resnet", "euler"], default="direct"
    )
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_epochs", default=1, type=int)
    parser.add_argument("--n_traj_train", default=1000, type=int)
    parser.add_argument("--n_traj_validate", default=10, type=int)
    parser.add_argument("--t_start_train", default=0.0, type=float)
    parser.add_argument("--t_end_train", type=float, default=0.002)
    parser.add_argument("--t_start_validate", default=0.0, type=float)
    parser.add_argument("--t_end_validate", type=float, default=4 * np.pi)
    parser.add_argument("--step_size_train", default=0.001, type=float)
    parser.add_argument("--step_size_validate", default=0.001, type=float)
    args = parser.parse_args()

    if args.solver.lower() == "direct":
        solver = DirectSolver()
    elif args.solver.lower() == "resnet":
        solver = ResnetSolver()
    else:
        solver = args.solver

    # generate data

    def f(t, x):
        θ = x[..., 0]
        ω = x[..., 1]

        dθ = ω
        dω = -torch.sin(θ)

        return torch.stack((dθ, dω), dim=-1)

    x0_train = torch.tensor(lhs(2, args.n_traj_train), device=args.device) * 2 - 1

    x0_validate = torch.tensor(lhs(2, args.n_traj_validate), device=args.device) * 2 - 1

    x0_example = torch.tensor((0.6, 0)).double().unsqueeze(0).to(args.device)
    step_size_train = args.step_size_train
    t_span_train = torch.arange(args.t_start_train, args.t_end_train, step_size_train)
    t_span_validate = torch.arange(
        args.t_start_validate, args.t_end_validate, args.step_size_validate
    )

    _, x_train = odeint(f, x0_train, t_span_train, solver="rk4")
    _, x_validate = odeint(f, x0_validate, t_span_validate, solver="rk4")
    _, x_example = odeint(f, x0_example, t_span_validate, solver="rk4")

    # model
    layers = []
    layers.append(nn.Linear(2, args.hidden_dim))
    for _ in range(args.n_layers):
        layers.append(nn.Linear(args.hidden_dim, args.hidden_dim))
        layers.append(nn.Softplus())

    layers.append(nn.Linear(args.hidden_dim, 2))

    net = nn.Sequential(*layers)
    net.to(args.device).double()

    # optimizer

    opt = Adam(net.parameters())

    # train
    losses = []

    x_pred_train = None
    for _ in tqdm(range(args.n_epochs)):

        _, x_pred_train = odeint(
            lambda t, x: net(x), x0_train, t_span_train, solver=solver
        )
        loss = mse_loss(x_pred_train, x_train)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    _, x_pred_validate = odeint(
        lambda t, x: net(x), x0_validate, t_span_validate, solver=solver
    )

    _, x_pred_example = odeint(
        lambda t, x: net(x), x0_example, t_span_validate, solver=solver
    )

    # derivatives
    x0_grid = get_meshgrid(0.01).double()
    x_derivative = f(None, x0_grid)

    if args.solver == "direct":

        out = net(x0_grid)
        # normalize for the step size used during training. If the network is trained with a step-size of 1/100 of a second
        # it will predict changes that are 100 times as small as those for 1 second.
        x_derivative_pred = (out - x0_grid) / step_size_train

    elif args.solver == "resnet":
        x_derivative_pred = net(x0_grid) / step_size_train
    else:
        x_derivative_pred = net(x0_grid)

    # plot
    x_pred_train = x_pred_train.detach().numpy()
    x_pred_validate = x_pred_validate.detach().numpy()
    x_pred_example = x_pred_example.detach().numpy()
    x_derivative_pred = x_derivative_pred.detach().numpy()
    x_derivative = x_derivative.detach().numpy()
    x0_grid = x0_grid.detach().numpy()

    # streamplot
    density = 1
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("stream plot")
    ax.streamplot(
        x0_grid[..., 0],
        x0_grid[..., 1],
        x_derivative[..., 0],
        x_derivative[..., 1],
        color="black",
        density=density,
    )

    ax.streamplot(
        x0_grid[..., 0],
        x0_grid[..., 1],
        x_derivative_pred[..., 0],
        x_derivative_pred[..., 1],
        color="blue",
        density=density,
    )

    ax.set_xlabel("θ")
    ax.set_ylabel("ω")

    # quiver
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("quiver")
    ax.quiver(
        x0_grid[..., 0],
        x0_grid[..., 1],
        x_derivative[..., 0],
        x_derivative[..., 1],
        color="black",
        angles="xy",
        scale_units="xy",
        # scale=1,
    )

    ax.quiver(
        x0_grid[..., 0],
        x0_grid[..., 1],
        x_derivative_pred[..., 0],
        x_derivative_pred[..., 1],
        color="blue",
        angles="xy",
        scale_units="xy",
        # scale=1,
    )

    ax.set_xlabel("θ")
    ax.set_ylabel("ω")

    # phase space, training
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("phase space: training")
    ax.plot(x_train[..., 0], x_train[..., 1], label="true", color="black")
    ax.plot(
        x_pred_train[..., 0],
        x_pred_train[..., 1],
        label="predicted",
        color="blue",
        linestyle="dashed",
    )
    ax.set_xlabel("θ")
    ax.set_ylabel("ω")

    # phase space, validation
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("phase space: validation")
    ax.plot(x_validate[..., 0], x_validate[..., 1], label="true", color="black")
    ax.plot(
        x_pred_validate[..., 0],
        x_pred_validate[..., 1],
        label="predicted",
        color="blue",
        linestyle="dashed",
        # markevery=[0],
        # marker=">",
    )
    ax.set_xlabel("θ")
    ax.set_ylabel("ω")

    # time series validation, specific idx
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    example_idx = 0
    fig.canvas.manager.set_window_title(f"states vs time: validation idx={example_idx}")

    ax1.plot(t_span_validate, x_validate[..., example_idx, 0], color="black")
    ax1.plot(
        t_span_validate,
        x_pred_validate[..., example_idx, 0],
        linestyle="dashed",
        color="blue",
    )
    ax2.plot(t_span_validate, x_validate[..., example_idx, 1], color="black")
    ax2.plot(
        t_span_validate,
        x_pred_validate[..., example_idx, 1],
        linestyle="dashed",
        color="red",
    )
    ax1.set_ylabel("θ(t)")
    ax2.set_ylabel("ω(t)")
    ax2.set_xlabel("t")

    # time series validation, example (0.6,0)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.canvas.manager.set_window_title(f"states vs time: validation example")

    ax1.plot(t_span_validate, x_example[..., 0], color="black")
    ax1.plot(
        t_span_validate,
        x_pred_example[..., 0],
        linestyle="dashed",
        color="blue",
    )
    ax2.plot(t_span_validate, x_example[..., 1], label="true", color="black")
    ax2.plot(
        t_span_validate,
        x_pred_example[..., 1],
        linestyle="dashed",
        color="red",
        label="predicted",
    )
    ax1.set_ylabel("θ(t)")
    ax2.set_ylabel("ω(t)")
    ax2.set_xlabel("t")
    ax2.legend()

    # show
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()

    plt.show()
