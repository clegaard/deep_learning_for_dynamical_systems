from matplotlib import cm, pyplot as plt
from matplotlib.collections import LineCollection
import torch
import numpy as np


def get_meshgrid(step_per_axis, domain=1.0, flatten=False, dtype=torch.float64):
    ε = 1e-10
    x = torch.arange(-domain, domain + ε, step_per_axis, dtype=dtype)
    y = x = torch.arange(-domain, domain + ε, step_per_axis, dtype=dtype)

    xy = torch.stack(torch.meshgrid((x, y)))
    if flatten:
        return xy.flatten(1).T
    else:
        return xy.T


def plot_colored(fig, ax, t, x, cmap="jet", label=None, colorbar=False, **kwargs):
    """
    t : (time x traj)
    x : (time x traj x state)
    """
    # x = torch.tensor(x)
    # t = torch.tensor(t)
    x = torch.as_tensor(x)
    t = torch.as_tensor(t)

    if t.ndim == 1:
        t = t.unsqueeze(-1).expand(-1, x.shape[1])

    norm = plt.Normalize(t.min(), t.max())

    for i in range(t.shape[1]):
        xi = x[:, i]
        ti = t[:, i]
        segments = torch.stack([xi[:-1], xi[1:]], axis=1)

        if i == t.shape[1] - 1:
            lc = LineCollection(segments, cmap=cmap, norm=norm, label=label, **kwargs)
        else:
            lc = LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
        lc.set_array(ti)
        ax.add_collection(lc)

    if colorbar:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(x.min(), x.max())
