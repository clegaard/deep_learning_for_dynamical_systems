from typing import Tuple
import numpy as np
import torch

from enum import Enum
from smt.sampling_methods import LHS
from torchdiffeq import odeint


class Sampling(Enum):
    RANDOM = 0
    GRID = 1


def grid_init_samples(domain, n_trajectories: int) -> np.ndarray:
    x = np.linspace(domain[0][0], domain[0][1], n_trajectories)
    y = np.linspace(domain[1][0], domain[1][1], n_trajectories)

    xx, yy = np.meshgrid(x, y)
    return np.concatenate((xx.flatten()[..., np.newaxis], yy.flatten()[..., np.newaxis]), axis=1)


def random_init_samples(domain, n_trajectories: int) -> np.ndarray:
    values = LHS(xlimits=np.array(domain))
    return values(n_trajectories)


def simulate_ode(f, y0s: torch.Tensor, n_steps: int, step_size: float) -> torch.Tensor:
    time_points = torch.arange(0., step_size * (n_steps + 1), step_size)
    ys = [(odeint(f, y0, time_points)) for y0 in y0s]
    return torch.stack(ys)


def pendulum(t, y):
    theta = y[0]
    omega = y[1]
    d_theta = omega
    d_omega = - torch.sin(theta)
    return torch.tensor([d_theta, d_omega]).float()


def load_pendulum_data(y0s_domain=None, n_trajectories=100, n_steps=2, 
step_size=0.001, sampling=Sampling.RANDOM) -> Tuple[torch.Tensor, torch.Tensor]:
    if not y0s_domain:
        y0s_domain = [[-1., 1.], [-1., 1.]]

    if sampling == Sampling.RANDOM:
        y0s = random_init_samples(y0s_domain, n_trajectories)
    elif sampling == Sampling.GRID:
        y0s = grid_init_samples(y0s_domain, n_trajectories)

    y0s = torch.tensor(y0s).float()
    y = simulate_ode(pendulum, y0s, n_steps, step_size)
    return y[:, 0, :].unsqueeze(dim=1), y
