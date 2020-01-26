import torch
import matplotlib.pyplot as plt
from saturation_plot import plot_saturations


def main():
    num_layers = 3
    batch_size = 128
    num_activations = 10
    activations = torch.rand(num_layers, batch_size, num_activations)
    fig, ax = plt.subplots()
    ax = plot_saturations(activations, ["layer_{}".format(i) for i in range(1, num_layers + 1)], ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
