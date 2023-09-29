import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    '''
      This class should be implemented similarly to the BasicBlock from the pytorch implementation.

      Since in resnet18 all blocks have only two parameters that are responsible for the number of channels,
      we will also use a simplified notation.

      The first convolutional layer has the dimension (in_channels, out_channels),
      the second convolutional layer has the dimension (out_channels, out_channels).

      You are required to implement the correct forward() and __init__() methods for this block.

      Remember to use batch normalizations and activation functions.
      Shorcut will require you to understand what projection convolution is.

      In general, it is recommended to refer to the original article, the pytorch implementation and
      other sources of information to successfully assemble this block.

      Hint! You can use nn.Identity() to implement shorcut.
    '''

    def __init__(self, in_channels, out_channels):
        '''
        The block must have the following fields:
            *self.shorcut
            *self.activation
            *self.conv1
            *self.conv2
            *self.bn1
            *self.bn2

        Hint! Don't forget the bias, padding, and stride parameters for convolutional layers.
        '''

        super().__init__()
        stride = (2, 2) if in_channels != out_channels else (1, 1)

        # <----- your code here ----->
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers = nn.Sequential(
            self.conv1,
            self.bn1,
            self.activation,
            self.conv2,
            self.bn2
        )

    def forward(self, x):
        '''
        '''

        residual = self.shortcut(x)

        # <----- your code here ----->
        x = self.layers(x)

        return x + residual


class ResNetLayer(nn.Module):
    '''
    This class should be implemented similarly to layer from the pytorch implementation.

    To implement the layer, you will need to create two ResidualBlocks inside.
    Determining the appropriate dimensions is up to you.
    '''

    def __init__(self, in_channels, out_channels):
        '''
        The layer must have the following field declared:
            *self.blocks
        '''

        super().__init__()

        # <----- your code here ----->
        self.blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        '''
        Note that blocks must be packed to make forward work in its original form.
        '''
        x = self.blocks(x)
        return x


class ResNet18(nn.Module):
    '''
    Finally, this class should consist of three main components:
      1. Four preparatory layers
      2. A set of internal ResNetLayers
      3. Final classifier

    Hint! In order for the network to process images from CIFAR10, you should replace the parameters
          of the first convolutional layer on kernel_size=(3, 3), stride=(1, 1) and padding=(1, 1).
    '''

    def __init__(self, in_channels=3, n_classes=10):
        '''
        The class must have the following fields declared:
            *self.conv1
            *self.bn1
            *self.activation
            *self.maxpool
            *self.layers
            *self.avgpool
            *self.flatten
            *self.fc

        A different grouping of parameters is allowed that does not violate the idea of the network architecture.
        '''

        super().__init__()

        # <----- your code here ----->
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layers =  nn.Sequential(
            ResNetLayer(64, 64),
            ResNetLayer(64, 128),
            ResNetLayer(128, 256),
            ResNetLayer(256, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        self.resnet = nn.Sequential(
            self.conv1,
            self.bn1,
            self.activation,
            self.maxpool,
            self.layers,
            self.avgpool,
            self.flatten,
            self.fc
        )

    def forward(self, x):
        # <----- your code here ----->

        x = self.resnet(x)

        return x
