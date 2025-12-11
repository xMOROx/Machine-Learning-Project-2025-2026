"""CNN Models for CIFAR-10 classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SimpleCNN(nn.Module):
    """Simple CNN architecture for CIFAR-10 classification.
    
    This is a baseline convolutional network optimized for 32x32 images
    with 10-class classification output.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Store the last conv layer output for GradCAM
        self.features = None
        self.gradients = None
    
    def activations_hook(self, grad):
        """Hook to capture gradients."""
        self.gradients = grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        
        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))  # 4x4
        
        # Store features for GradCAM
        self.features = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        
        x = self.pool(x)  # 4x4 -> 2x2
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def get_activations_gradient(self):
        """Return the gradients."""
        return self.gradients
    
    def get_activations(self):
        """Return the feature maps."""
        return self.features


class ResNetCIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-10.
    
    Uses pretrained weights and modifies the final layer for 10-class output.
    Also exposes intermediate features for GradCAM visualization.
    """
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet18(weights=weights)
        
        # Modify first conv layer for 32x32 input (smaller kernel, no stride)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for small images
        
        # Modify final layer for CIFAR-10
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Store features for GradCAM
        self.features = None
        self.gradients = None
        
        # Register hook on layer4 (last conv block)
        self.resnet.layer4.register_forward_hook(self._forward_hook)
    
    def _forward_hook(self, module, input, output):
        """Forward hook to capture feature maps."""
        self.features = output
        if output.requires_grad:
            output.register_hook(self._backward_hook)
    
    def _backward_hook(self, grad):
        """Backward hook to capture gradients."""
        self.gradients = grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
    def get_activations_gradient(self):
        """Return the gradients."""
        return self.gradients
    
    def get_activations(self):
        """Return the feature maps."""
        return self.features
