from torch import nn
from torchvision import transforms


class CustomNet(nn.Module):
    """
    A class for representing the main neural network.

    ...

    Attributes
    ----------
    cnn_layers : torch.nn.Sequential
        A container containing descriptions of layers specific to a convolutional neural network.
    linear_layers : torch.nn.Sequential
        A container containing descriptions of fully connected layers.

    Methods
    -------
    forward(x)
        Determines how the data will pass through the neural network.
    """

    def __init__(self):
        """
        Sets all the necessary attributes for the MainNet object.
        """

        super(CustomNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(12544, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Determines how the data will pass through the neural network. Returns the data received after processing by
        the neural network.

        Parameters
        ----------
        x
            Input data.

        Returns
        -------
        x
            Output data.
        """

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


NUM_CHANNELS = 3
IMG_SIDE_SIZE = 224
transformation = transforms.Compose([transforms.Resize(size=(IMG_SIDE_SIZE, IMG_SIDE_SIZE)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
