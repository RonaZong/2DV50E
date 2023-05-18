import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tkinter import Tcl
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset, normalize, load_data


# def read_data(input_path):
#     # Open image in RGB mode
#     image_filename = sorted(os.listdir(input_path), key=lambda x: int(x.split("_")[0]))
#     for file in image_filename:
#         input = Image.open(os.path.join(input_path, image_filename))
#     print(input)
#
#
# def read_csv(args):
#     csv_filename = args
#     with open(csv_filename) as file:
#         df = file.read()


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# Remember to zero gradients before the backward call with a call to zero_grad()
def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        # In the backward() call on the loss tensor, gradients are accumulated in the params tensor (.grad)
        optimizer.zero_grad()
        loss.backward()
        # The optimizer can then use the gradients to compute new params with a call to step()
        optimizer.step()
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


if __name__ == "__main__":
    path_sample = "../Data/sample_resized_150/"
    csv_sample = "../Data/sample.csv"

    # image_filename = sorted(os.listdir(path_sample), key=lambda x: int(x.split("_")[0]))

    image_sample = load_data(path_sample)

    # Track gradients across the computation graph involving this tensor
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2

    # Use Stochastic Gradient Descent (SGD)
    # Disregard the term stochastic for now, the optimizer itself is just regular gradient descent (with the default setting of momentum = 0)
    # optimizer = optim.SGD([params], lr=learning_rate)
    # training_loop(
    #     n_epochs=100,
    #     optimizer=optimizer,
    #     params=params,
    #     t_u=t_u,
    #     t_c=t_c)

    # t_range = torch.arange(20., 90.).unsqueeze(1)

    # fig = plt.figure(dpi=600)
    # plt.xlabel("")
    # plt.ylabel("")
    # plt.plot(t_u.numpy(), t_c.numpy(), 'o')
    # plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
    # plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')

    # Tcl().call("lsort", "-dict", image_filename)

