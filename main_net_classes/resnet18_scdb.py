import torch
from torch import nn
from torchvision import transforms


def ResNet18():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    for param in model.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(nn.Linear(512, 128),
                             nn.ReLU(),
                             nn.Linear(128, 64),
                             nn.ReLU(),
                             nn.Linear(64, 1),
                             nn.Sigmoid())
    return model


NUM_CHANNELS = 3
IMG_SIDE_SIZE = 256
transformation = transforms.Compose([transforms.Resize(size=(IMG_SIDE_SIZE, IMG_SIDE_SIZE)),
                                     # transforms.RandomRotation(30),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
