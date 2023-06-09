from torchvision import models as models
import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

def model(pretrained, requires_grad):
    model = models.resnet101(progress=True, pretrained=pretrained)
    # To freeze hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False

    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True

    # Amount of classes
    model.fc = nn.Sequential(
        nn.Linear(2048, 4),
        nn.Linear(2048, 3)
    )
    return model

class CNN1(nn.Module):
    def __init__(self, pretrained):
        super(CNN1, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet101"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet101"](pretrained=None)
        self.fc1 = nn.Linear(2048, 4)
        self.fc2 = nn.Linear(2048, 3)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.fc1(x)
        label2 = self.fc2(x)
        return {"label1:": label1, "label2:": label2}