import torch


def get_meshgrid(step_per_axis, flatten=False):
    x = torch.arange(-1, 1 + step_per_axis, step_per_axis)
    y = x = torch.arange(-1, 1 + step_per_axis, step_per_axis)

    xy = torch.stack(torch.meshgrid((x, y)))
    if flatten:
        return xy.flatten(1).T
    else:
        return xy.T


if __name__ == "__main__":

    a = 10
