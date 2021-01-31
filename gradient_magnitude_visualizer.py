import torch
import matplotlib.pyplot as plt
import numpy as np


class GradientHistory(object):
    def __init__(self):
        self.gradients = []

    def add_gradient(self, grad):
        self.gradients.append(grad.detach().clone())


class ParametersForLayer(object):
    def __init__(self, layer_name: str):
        self.parameters = {}

    def add_parameter(self, param_name: str, param):
        gh = GradientHistory()
        self.parameters[param_name] = gh
        param.register_hook(gh.add_gradient)


class GradientMagnitudeVisualizer(object):
    def __init__(self):
        self.gradients = {}

    def register_hooks(self, net: torch.nn.Module):
        # Parse network structure: Create an object for each layer
        for name, param in net.named_parameters():
            name_split = name.split(".")
            layer_path = ".".join(name_split[:-1])
            param_name = name_split[-1]
            if layer_path not in self.gradients.keys():
                self.gradients[layer_path] = ParametersForLayer(layer_path)
            self.gradients[layer_path].add_parameter(param_name, param)

    def plot_bar(self, end_idx=-1, filepath=None, show=False):
        fig, ax = plt.subplots(nrows=len(self.gradients), ncols=1, figsize=(4, 2*len(self.gradients)))
        for i, layer_name in enumerate(self.gradients.keys()):
            params = self.gradients[layer_name]
            x = np.arange(len(params.parameters))
            height_beginning = [params.parameters[param_name].gradients[0].abs().mean().item() for param_name in params.parameters.keys()]
            mean_gradients = []
            for param_name in params.parameters.keys():
                mean_grad = 0.0
                for grad in params.parameters[param_name].gradients:
                    mean_grad += grad.abs().mean().item()
                mean_gradients.append(mean_grad / len(params.parameters[param_name].gradients))
            height_end = [params.parameters[param_name].gradients[end_idx].abs().mean().item() for param_name in params.parameters.keys()]
            param_names = params.parameters.keys()
            ax[i].grid()
            ax[i].bar(x-0.2, height_beginning, width=0.2, color="green", align="center")
            ax[i].bar(x, mean_gradients, width=0.2, color="gray", align="center")
            ax[i].bar(x+0.2, height_end, width=0.2, color="red", align="center")
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(param_names)
            ax[i].set_ylabel(layer_name)
            ax[i].set_axisbelow(True)
        fig.legend(labels=["Beginning", "Mean", "End"])
        fig.subplots_adjust(right=1.0, bottom=0.03, top=1.0, left=0.196)
        if show:
            plt.show()
        if filepath is not None:
            fig.savefig(filepath)
