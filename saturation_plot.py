import torch
import matplotlib.pyplot as plt


def plot_saturation(activations: torch.Tensor, range_min: float = 0.0, range_max: float = 1.0, ax=None):
    left_saturation = range_min + 0.1 * (range_max - range_min)
    right_saturation = range_max - 0.1 * (range_max - range_min)

    batch_size = activations.size(0)
    num_activations = activations.size(-1)
    activations_flattened = activations.reshape(-1, num_activations)
    left_saturated = (activations_flattened < left_saturation).sum(dim=0) / float(batch_size)
    right_saturated = (activations_flattened > right_saturation).sum(dim=0) / float(batch_size)
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("fraction right saturated")
        ax.set_ylabel("fraction left saturated")
    ax.scatter(right_saturated, left_saturated, alpha=0.5)
    return ax


def plot_saturations(activations: torch.Tensor, layer_names: list, range_min: float = 0.0, range_max: float = 1.0, ax = None):
    assert(activations.size(0) == len(layer_names))
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(len(layer_names)):
        ax = plot_saturation(activations[i], range_min, range_max, ax)
    ax.set_xlabel("fraction right saturated")
    ax.set_ylabel("fraction left saturated")
    ax.margins(0)
    ax.legend(layer_names)
    ax.plot([1,0], [0,1], color="black", linewidth=1)
    return ax
