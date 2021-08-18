import numpy as np
import torch

from matplotlib import pyplot as plt

from tqdm import tqdm
from torch import nn
from torch.nn.functional import mse_loss
from torchdiffeq import odeint

from dataloader import Sampling, load_pendulum_data

def run():
    """
    Training data

    """
    y0s_domain = [[-1., 1.], [-1., 1.]]
    step_size = 0.01

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

        def simulate_direct(self, y0s, n_steps):
            ys = [y0s]
            for _ in range(n_steps):
                ys.append(self(ys[-1]))
            return torch.swapaxes(torch.stack(ys), 0, 1).squeeze(dim=2)

        def simulate_euler(self, y0s, n_steps, step_size):
            ys = [y0s]
            for _ in range(n_steps):
                ys.append(ys[-1] + step_size * self(ys[-1]))
            return torch.swapaxes(torch.stack(ys), 0, 1)


    model = RNN(n_states=2)
    opt = torch.optim.Adam(model.parameters())

    """
    Training

    """
    epochs = 2000
    progress = tqdm(range(epochs), 'Training')
    losses = []

    for _ in progress:
        y_pred = model(y)

        loss = mse_loss(y_pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())
        progress.set_description(f'loss: {loss.item()}')

    """
    Test data

    """
    y0s, y = load_pendulum_data(y0s_domain, n_trajectories=10, n_steps=20, step_size=step_size, sampling=Sampling.GRID)
    y_pred = model.simulate_direct(y0s, n_steps=20)

    """
    Plot results

    """
    plt.plot(y_pred.detach().numpy()[:, :, 1].T, y_pred.detach().numpy()[:, :, 0].T, color='r')
    plt.plot(y.numpy()[:, :, 1].T, y.numpy()[:, :, 0].T, color='b')
    plt.scatter(y[:, 1, 1], y[:, 1, 0], color='g')
    plt.ylim(y0s_domain[0])
    plt.xlim(y0s_domain[1])
    plt.show()



if __name__ == "__main__":
    run()
