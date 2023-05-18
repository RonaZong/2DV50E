import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import config
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DRDataset(Dataset):
    def __init__(self, input_path, label_path, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(label_path)
        self.input_path = input_path
        self.input = os.listdir(input_path)
        self.train = train
        self.transform = transform

    def __len__(self):
        # Number of data in csv or input
        return self.data.shape[0] if self.train else len(self.input)

    def __getitem__(self, item):
        if self.train:
            input, label = self.data.iloc[item]
            print(input, label)
        else:
            # if test simply return -1 for label, I do this inorder to
            # re-use same dataset class for test set submission later on
            input, label = self.input[item], -1
            input = input.replace("jpeg", "")
            # print(input, label)

        image = np.array(Image.open(os.path.join(self.input_path, input + ".jpeg")))

        if self.transform:
            image = self.transform(image=image)["image"]

        return input, label, image


if __name__ == "__main__":
    path_sample = "../Data/train1_resized_150/"
    label_sample = "../Data/train1.csv"
    # image_filename = sorted(os.listdir(path_sample), key=lambda x: int(x.split("_")[0]))

    dataset = DRDataset(
        input_path=path_sample,
        label_path=label_sample,
        train=True,
        transform=config.val_transforms
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=7,
        shuffle=True,
        pin_memory=True
    )

    for x, label, file in tqdm(loader):
        print("x:", x)
        import sys
        sys.exit()

