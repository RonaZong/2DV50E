import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DRDataset(Dataset):
    def __init__(self, input_path, root="../Data/", train=True, transform=None):
        super().__init__()
        self.img_width = 32
        self.img_height = self.img_width
        self.img_width_crop = 28
        self.img_height_crop = self.img_width_crop
        self.input_path = input_path
        self.train = train

        if self.train:
            self.input_path = os.path.join(root, input_path)
        else:
            self.input_path = os.path.join(root, input_path)

        if os.path.exists(self.input_path):
            print("Using existing", self.input_path)

        self.dataset_train = datasets.ImageFolder(self.input_path)

    def __getitem__(self, item):
        image, label = self.dataset_train.__getitem__(item)
        image = transforms.Resize((self.img_width, self.img_height))(image)

        if self.train:
            image = transforms.RandomAffine((-5, 5))(image)
            image = transforms.RandomCrop((self.img_width_crop, self.img_height_crop))(image)
            image = transforms.ColorJitter(0.8, contrast=0.4)(image)
            # if label in [11, 12, 13, 17, 18, 26, 30, 35]:
            #     image = transforms.RandomHorizontalFlip(p=0.5)(image)
        else:
            image = transforms.CenterCrop((self.img_width_crop, self.img_height_crop))(image)

        image = transforms.ToTensor()(image)

        return image, label

    def __len__(self):
        # Number of data in csv or input
        # return self.data.shape[0] if self.train else len(self.input)
        return self.dataset_train.__len__()


def normalize(arr):
    # Function to scale an input array to [-1, 1]
    arr_min = arr.min()
    arr_max = arr.max()

    # Check the original min and max values
    print("Min: %.3f, Max: %.3f" % (arr_min, arr_max))
    arr_range = arr_max - arr_min
    scaled = np.array((arr - arr_min) / float(arr_range), dtype="f")
    arr_new = -1 + (scaled * 2)

    # Make sure min value is -1 and max value is 1
    print("Min: %.3f, Max: %.3f" % (arr_new.min(), arr_new.max()))
    return arr_new


def load_data(path):
    # Loop through all files in the directory
    for filename in os.listdir(path):
        # Load image
        image = Image.open(path + filename)
        # Convert to numpy array
        image = np.array(image)
        # Find number of channels
        if image.ndim == 2:
            channels = 1
            print(filename + " has 1 channel")
        else:
            channels = image.shape[-1]
            print(filename + " has", channels, "channels")
        # Scale to [-1,1]
        image = normalize(image)
        # Convert to Tensor
        image = transforms.ToTensor()(image)
    return image


def image_show(image):
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


def main():
    image_sample = "sample_resized2_150/"
    # csv_sample = "../Data/sample.csv"
    dataset = DRDataset(input_path=image_sample)
    loader = DataLoader(
        dataset=dataset, batch_size=32, num_workers=7, shuffle=True, pin_memory=True
    )
    batch_size = 128
    generator_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # get some random training images
    dataiter = iter(generator_train)
    images, labels = dataiter.next()


    # image_filename = sorted(os.listdir(image_sample), key=lambda x: int(x.split("_")[0]))


if __name__ == "__main__":
    main()


