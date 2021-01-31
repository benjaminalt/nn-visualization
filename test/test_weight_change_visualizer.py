import torch
from nn_visualization.weight_magnitude_visualizer import WeightChangeVisualizer
from tqdm import tqdm


def main():
    viz = WeightChangeVisualizer()
    net = torch.nn.Sequential(
        torch.nn.Linear(8, 24),
        torch.nn.SELU(),
        torch.nn.BatchNorm1d(24),
        torch.nn.Linear(24, 13),
        torch.nn.SELU(),
        torch.nn.BatchNorm1d(13),
        torch.nn.Linear(13, 7)
    )
    num_data = 32
    inputs = torch.rand(num_data, 8)
    labels = torch.rand(num_data, 7)
    ds = torch.utils.data.TensorDataset(inputs, labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, drop_last=True)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    viz.set_params_before(net)
    for inp, label in tqdm(dl):
        outp = net(inp)
        loss = torch.nn.MSELoss()(outp, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
    viz.set_params_after(net)
    viz.plot_bar(show=True)


if __name__ == '__main__':
    main()
