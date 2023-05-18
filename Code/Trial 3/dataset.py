import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import cv2
import torch_directml


class DRDataset(Dataset):
    def __init__(self, train, test, csv):
        super().__init__()
        self.train = train
        self.test = test
        self.csv = csv

        self.image_names = self.csv[:]["Image name"]

        self.labels = np.array(self.csv.drop(["Image name"], axis=1))
        self.retino_label = self.csv[:]["Retinopathy grade"]
        self.macula_label = self.csv[:]['Risk of macular edema']

        self.train_ratio = int(0.85 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.images_names.count}")
            self.image_names = list(self.images_names[:self.train_ratio])
            #self.labels = list(self.labels)
            self.retino_label = list(self.retinolabel[:self.train_ratio])
            self.macula_label = list(self.maculalabel[:self.train_ratio])
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])

        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(self.images_names[:self.valid_ratio:-10])
            #self.labels = list(self.labels)
            self.retino_label = list(self.retinolabel[:self.valid_ratio:-10])
            self.macula_label = list(self.maculalabel[:self.valid_ratio:-10])
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
            ])


        elif self.test == True and self.train == False:
            self.image_names = list(self.image_names[-10:])
            self.retino_label = list(self.retinolabel[-10:])
            self.macula_label = list(self.maculalabel[-10:])
            # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,index):
        myDevice = torch_directml.device()

        image = cv2.imread(f"/home/user/Thesis/MLThesis/Disease Grading/Original Images/Training Set/{self.image_names[index]}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        #targets = self.labels[index]
        retino_label = self.retinolabel[index]
        macula_label = self.maculalabel[index]
        return {
            'image': torch.tensor(image, dtype=torch.float32, device=myDevice),
            'label1': torch.tensor(retino_label, dtype=torch.float32, device=myDevice),
            'label2': torch.tensor(macula_label, dtype=torch.float32,device=myDevice)
        }
