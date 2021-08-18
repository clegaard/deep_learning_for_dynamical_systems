from typing import Tuple
import numpy as np
import torch

from enum import Enum
from smt.sampling_methods import LHS
from torchdyn.numerics import odeint

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


def pendulum(t, y):
    θ = y[:, 0]
    ω = y[:, 1]

    dθ = ω
    dω = -torch.sin(θ)

    return torch.stack((dθ, dω), dim=1)


def load_pendulum_data(t_span, y0s_domain=None, n_trajectories=100, sampling=Sampling.RANDOM, solver='rk4') -> Tuple[torch.Tensor, torch.Tensor]:
    if not y0s_domain:
        y0s_domain = [[-1., 1.], [-1., 1.]]

    if sampling == Sampling.RANDOM:
        y0s = random_init_samples(y0s_domain, n_trajectories)
    elif sampling == Sampling.GRID:
        y0s = grid_init_samples(y0s_domain, n_trajectories)

    y0s = torch.tensor(y0s)
    _, ys = odeint(pendulum, y0s, t_span, solver) 

    return y0s, ys
