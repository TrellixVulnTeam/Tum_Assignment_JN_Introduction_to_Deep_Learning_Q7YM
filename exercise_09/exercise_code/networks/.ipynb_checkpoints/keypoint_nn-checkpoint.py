"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        #super(KeypointModel, self).__init__()
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        # input layer (image size=96x96)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 2)
        #self.conv5 = nn.Conv2d(256, 512, 1)

        self.pool = nn.MaxPool2d(2)
        self.pool1=nn.MaxPool2d(3)
        self.relu=nn.ELU()
        self.bn1= nn.BatchNorm2d(32)
        self.bn2= nn.BatchNorm2d(64)
        self.bn3= nn.BatchNorm2d(128)
        self.bn4= nn.BatchNorm2d(256)
        #self.bn5= nn.BatchNorm1d(500)
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2= nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.25)
        self.dropout4=nn.Dropout2d(0.25)
        self.dropout5=nn.Dropout2d(0.3)
        self.dropout6=nn.Dropout2d(0.1)
        
        self.fc1 = nn.Linear(6400, 500)
        #self.fc2 = nn.Linear(350, 256)
        #self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(500, 30)
        self.tan = nn.Tanh()
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        if (x.shape == torch.Size([1, 96, 96])):
            x = x.unsqueeze(0)

        x = (self.relu(self.pool1(self.conv1(x))))
        x = (self.relu(self.pool((self.conv2(x)))))
        x = (self.relu(self.pool(self.conv3(x))))
        x = (self.relu(self.conv4(x)))
        #x = (self.pool(self.relu(self.conv5(x))))

        # Flatten
        x = x.view(x.size()[0], -1)

        x = (self.relu(self.dropout6((self.fc1(x)))))
        
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        
        x = (self.fc4(x))
        x=self.tan(x)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
