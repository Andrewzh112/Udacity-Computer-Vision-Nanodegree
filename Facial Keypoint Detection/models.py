## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.vgg = models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
          param.requires_grad = False
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        in_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = Identity()
        self.seq1 = nn.Sequential(
            nn.Linear(in_features,int(in_features/8)),
            nn.BatchNorm1d(int(in_features/8)),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.seq2 = nn.Sequential(
            nn.Linear(int(in_features/8),int(in_features/64)),
            nn.BatchNorm1d(int(in_features/64)),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.output = nn.Linear(int(in_features/64), 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.vgg(x)
        x = self.seq1(x)
        x = self.seq2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return self.output(x)
