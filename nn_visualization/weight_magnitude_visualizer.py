import torch
import matplotlib.pyplot as plt
import numpy as np


class WeightChangeVisualizer(object):
    def __init__(self):
        self.weights_before = {}
        self.weights_after = {}

    def _set_params(self, net: torch.nn.Module, dic: dict):
        for name, param in net.named_parameters():
            name_split = name.split(".")
            layer_path = ".".join(name_split[:-1])
            param_name = name_split[-1]
            if layer_path not in dic.keys():
                dic[layer_path] = {}
            dic[layer_path][param_name] = param.detach().clone()

    def set_params_before(self, net: torch.nn.Module):
        self._set_params(net, self.weights_before)

    def set_params_after(self, net: torch.nn.Module):
        self._set_params(net, self.weights_after)

    def plot_bar(self, filepath=None, show=False):
        fig, ax = plt.subplots(nrows=len(self.weights_before), ncols=1,
                               figsize=(4, 2*len(self.weights_before)))
        for i, layer_name in enumerate(self.weights_before.keys()):
            params_before = self.weights_before[layer_name]
            params_after = self.weights_after[layer_name]
            x = np.arange(len(params_before))
            diff = [(params_before[param_name] - params_after[param_name]).abs().mean() for param_name in params_before.keys()]
            param_names = params_before.keys()
            ax[i].grid()
            ax[i].bar(x, diff, width=0.6, color="gray", align="center")
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(param_names)
            ax[i].set_ylabel(layer_name)
            ax[i].set_axisbelow(True)
        fig.subplots_adjust(right=1.0, bottom=0.03, top=1.0, left=0.196)
        if show:
            plt.show()
        if filepath is not None:
            fig.savefig(filepath)
