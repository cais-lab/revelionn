import geffnet
from torch import nn
from torchvision import transforms


def MobileNet():
    model = geffnet.create_model('mobilenetv2_100', pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(1280, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1),
                                     nn.Sigmoid())

    for param in model.blocks[2:].parameters():
        param.requires_grad = True
    for param in model.conv_head.parameters():
        param.requires_grad = True
    for param in model.bn2.parameters():
        param.requires_grad = True
    for param in model.act2.parameters():
        param.requires_grad = True
    for param in model.global_pool.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


NUM_CHANNELS = 3
IMG_SIDE_SIZE = 224
transformation = transforms.Compose([transforms.Resize(size=(IMG_SIDE_SIZE, IMG_SIDE_SIZE)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
