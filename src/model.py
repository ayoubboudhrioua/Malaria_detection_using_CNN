import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models


"""
Build a Convolutional Neural Network from scratch
Architecture: Multiple conv blocks + fully connected layers
"""

class MalariaCNN(nn.Module):
    """
    Custom CNN for Malaria Detection
    
    Architecture:
    - 4 Convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
    - Fully connected layers with dropout
    - Output: 2 classes (Parasitized, Uninfected)
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(MalariaCNN, self).__init__()
        
        # ==========================
        # CONVOLUTIONAL LAYERS
        # ==========================
        
        # Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, 
                              kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, 
                              kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.flatten_size = 10*10*256
        
        # ==========================
        # FULLY CONNECTED LAYERS
        # ==========================
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, 160, 160]
        
        Returns:
            Output tensor [batch_size, num_classes]
        """
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

print("\n" + "="*70)
print("CNN ARCHITECTURE DEFINED")
print("="*70)
print("\n MalariaCNN class created!")
print("\nArchitecture summary will be displayed in the next cell.")