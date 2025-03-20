import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        
        # Use ResNet34 as encoder
        self.encoder = models.resnet34(pretrained=True)
        
        self.conv1 = self.encoder.conv1
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.conv_last = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder with skip connections
        x = self.upconv4(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv1(x)
        
        x = self.conv_last(x)
        
        return x
