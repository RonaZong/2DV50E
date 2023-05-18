import os
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import tensorflow as tf

from tkinter import Tcl
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset, normalize, load_data


writer = SummaryWriter()



class Net(nn.Module):
    def __init__(self, img_size=28):
        super(Net, self).__init__()
        # Add code here .... (see e.g. 'Switch to CNN' at https://pytorch.org/tutorials/beginner/nn_tutorial.html)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=(1,1))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=(1,1))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc3 = nn.Linear(in_features=512, out_features=43, bias=True)

    def forward(self, x):
        # And here ...
        x = x.view(-1, 3, 28, 28)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.elu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 512)
        x = F.elu(self.fc3(x))
        return x.view(-1, x.size(1))


tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)



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


# def model(t_u, w, b):
#     return w * t_u + b


# def loss_fn(t_p, t_c):
#     squared_diffs = (t_p - t_c) ** 2
#     return squared_diffs.mean()


# Remember to zero gradients before the backward call with a call to zero_grad()
# def training_loop(n_epochs, optimizer, params, t_u, t_c):
#     for epoch in range(1, n_epochs + 1):
#         t_p = model(t_u, *params)
#         loss = loss_fn(t_p, t_c)
#         # In the backward() call on the loss tensor, gradients are accumulated in the params tensor (.grad)
#         optimizer.zero_grad()
#         loss.backward()
#         # The optimizer can then use the gradients to compute new params with a call to step()
#         optimizer.step()
#         if epoch % 500 == 0:
#             print('Epoch %d, Loss %f' % (epoch, float(loss)))
#     return params

def main():
    image_sample = "../Data/sample_resized2_150/"
    csv_sample = "../Data/sample.csv"
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.eval()
    print(1)

if __name__ == "__main__":
    main()

    dataset_train = DRDataset()
    batch_size = 128
    generator_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()
    ## switch optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001, eps=0.1)

    cont = False
    if cont:
        net.load_state_dict(torch.load('traffic_simple'))

    no_epochs = 200
    for epoch in range(no_epochs):  # Loop over the dataset multiple times
        epoch_loss = running_loss = 0.0
        for i, data in enumerate(generator_train, 0):
            # Get the inputs; data is a list of [inputs, labels]
            # if (gpu):
            #     inputs, labels = data[0].to(device), data[1].to(device)
            # else:
            inputs, labels = data
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            reporting_interval = 100
            epoch_loss += loss.item()
            running_loss += loss.item()
            if i % reporting_interval == reporting_interval-1:  # Print every reporting_interval mini-batches
                # report_loss = running_loss / reproint
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / reporting_interval))
                running_loss = 0.0

        # Log to tensorboard
        writer.add_scalar("Loss/train", epoch_loss/(i+1.), epoch)

    print('Finished Training')
    writer.flush()



    # # Track gradients across the computation graph involving this tensor
    # params = torch.tensor([1.0, 0.0], requires_grad=True)
    # learning_rate = 1e-2

    # # Use Stochastic Gradient Descent (SGD)
    # # Disregard the term stochastic for now, the optimizer itself is just regular gradient descent (with the default setting of momentum = 0)
    # optimizer = optim.SGD([params], lr=learning_rate)
    # training_loop(
    #     n_epochs=100,
    #     optimizer=optimizer,
    #     params=params,
    #     t_u=t_u,
    #     t_c=t_c)
    #
    # t_range = torch.arange(20., 90.).unsqueeze(1)
    #
    # fig = plt.figure(dpi=600)
    # plt.xlabel("")
    # plt.ylabel("")
    # plt.plot(t_u.numpy(), t_c.numpy(), 'o')
    # plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
    # plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')

    # Tcl().call("lsort", "-dict", image_filename)

    # dataset = DRDataset(
    #     images_folder="../Data/sample_resized_150/",
    #     path_to_csv="../Data/sample.csv",
    #     transform=config.val_transforms
    # )
    # loader = DataLoader(
    #     dataset=dataset, batch_size=32, num_workers=7, shuffle=True, pin_memory=True
    # )
    #
    # for x, label, file in tqdm(loader):
    #     print(x.shape)
    #     print(label.shape)
    #     import sys
    #
    #     sys.exit()

