import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from database import CustomDataset
import torch.nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Transformations we can perform on our dataset for augmentation
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# class CNN(nn.Module):
#     def __init__(self, in_channels=4, num_classes=7):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=4,out_channels=8,kernel_size=(3,3),stride=(1, 1),padding=(1, 1))
#         conv1=nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
#         conv2=nn.ReLU()
#         self.fc1 = nn.Linear(64*4*28, num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         # x = x.reshape(x.shape[0], -1)
#         x = self.fc1(x)
#
#         return x
    # input_features = 3*28*28
    # def __init__(self):
    #     super().__init__()
    #     # 5 Hidden Layer Network
    #     self.fc1 = nn.Linear(3*28*28, 512)
    #     self.fc2 = nn.Linear(512, 256)
    #     self.fc3 = nn.Linear(256, 128)
    #     self.fc4 = nn.Linear(128, 64)
    #     self.fc5 = nn.Linear(64, 3)
    #
    #     # Dropout module with 0.2 probbability
    #     self.dropout = nn.Dropout(p=0.2)
    #     # Add softmax on output layer
    #     self.log_softmax = F.log_softmax
    #
    # def forward(self, x):
    #     x = x.view(x.size(0), -1)
    #     x = self.dropout(F.relu(self.fc1(x)))
    #     x = self.dropout(F.relu(self.fc2(x)))
    #     x = self.dropout(F.relu(self.fc3(x)))
    #     x = self.dropout(F.relu(self.fc4(x)))
    #
    #     x = self.log_softmax(self.fc5(x), dim=1)
    #
    #     return x


