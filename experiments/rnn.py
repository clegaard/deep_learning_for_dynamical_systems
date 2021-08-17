from matplotlib import pyplot as plt
from dataloader import Sampling, load_pendulum_data
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from smt.sampling_methods import LHS
from torchdiffeq import odeint


def run():
    """
    Training data

    """
    y0s_domain = [[-1., 1.], [-1., 1.]]
    step_size = 0.001

    y0s, y = load_pendulum_data(y0s_domain, n_trajectories=100, n_steps=2, step_size=step_size, sampling=Sampling.RANDOM)

    """
    Network
    
    """
    class RNN(nn.Module):

        def __init__(self, n_states):
            super().__init__()
            hidden_dim = 32
            self.rnn = nn.RNN(n_states, hidden_dim, batch_first=True)
            self.out = nn.Linear(hidden_dim, n_states)

        def forward(self, y0s):
            out, hidden = self.rnn(y0s)
            return self.out(out)

    model = RNN(n_states=2)
    opt = torch.optim.Adam(model.parameters())

    """
    Training

    """
    epochs = 200
    progress = tqdm(range(epochs), 'Training')
    mses = []

    for _ in progress:
        y_pred = model(y)

        loss = F.mse_loss(y_pred, y)
        mses.append(loss.detach().numpy())
        loss.backward()
        opt.step()
        opt.zero_grad()

        progress.set_description(f'loss: {loss.item()}')

    """
    Test data

    """
    y0s, y_test = load_pendulum_data(y0s_domain, n_trajectories=10, n_steps=3, step_size=step_size, sampling=Sampling.GRID)
    y_pred = model(y0s)

    """
    Plot results

    """
    plt.plot(y_pred.detach().numpy()[
            :, :, 1].T, y_pred.detach().numpy()[:, :, 0].T, color='r')
    plt.plot(y.numpy()[:, :, 1].T, y.numpy()[:, :, 0].T, color='b')
    plt.scatter(y[:, 1, 1], y[:, 1, 0])
    plt.ylim(y0s_domain[0])
    plt.xlim(y0s_domain[1])
    plt.show()



if __name__ == "__main__":
    run()
