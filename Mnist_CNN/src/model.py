import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1 , out_channels = 32 , kernel_size=3 , stride = 1, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size = 2 , stride = 2)
    self.conv2 = nn.Conv2d(in_channels=32 , out_channels = 64 , kernel_size=3 , stride = 1)
    self.pool2 = nn.MaxPool2d(kernel_size = 2 , stride = 2)


    self.fc1 = nn.Linear(in_features = 64*6*6 , out_features=128)
    self.fc2 = nn.Linear(in_features = 128, out_features=10)


  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = torch.flatten(x , 1)
    #FC LAYERS:
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
