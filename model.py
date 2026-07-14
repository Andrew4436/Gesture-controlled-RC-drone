import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__() 
        
        # layer 1: 1 -> 64 kernels. trims 100x100 to 98x98, maxpool cuts to 49x49
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # layer 2: 64 -> 128 kernels. trims 49x49 to 47x47, maxpool cuts to 23x23
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_input = nn.Linear(in_features=128 * 23 * 23, out_features=128)
        self.fc_hidden = nn.Linear(in_features=128, out_features=64)
        self.fc_output = nn.Linear(in_features=64, out_features=num_classes)
    
    def forward(self, x):    
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = x.view(-1, 128 * 23 * 23)
        
        x = F.relu(self.fc_input(x))
        x = F.relu(self.fc_hidden(x))
        x = self.fc_output(x)
        return x
