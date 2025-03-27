import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # Encoder (ResNet34)
        self.encoder = models.resnet34(weights='IMAGENET1K_V1')
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                self.encoder.conv1,
                self.encoder.bn1,
                self.encoder.relu,
                self.encoder.maxpool
            ), # Output: [B, 64, 64, 64]
            self.encoder.layer1, # Output: [B, 64, 64, 64]
            self.encoder.layer2, # Output: [B, 128, 32, 32]
            self.encoder.layer3, # Output: [B, 256, 16, 16]
            self.encoder.layer4  # Output: [B, 512, 8, 8]
        ])

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            self._make_decoder_block(512, 256), # Output: [B, 256, 16, 16]
            self._make_decoder_block(512, 128), # Output: [B, 128, 32, 32]
            self._make_decoder_block(256, 64),  # Output: [B, 64, 64, 64]
            self._make_decoder_block(128, 32)   # Output: [B, 32, 128, 128]
        ])
        
        # Final upsampling to reach 256x256
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        
        # Final output
        self.final = nn.Conv2d(32, n_classes, kernel_size=1) # Output: [B, n_classes, 256, 256]

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): # Input: [B, 3, 256, 256]
        # Encoder
        encoder_features = []
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_features.append(x)
        
        # Decoder with skip connections
        # First decoder block - no skip connection for the bottleneck
        x = self.decoder_blocks[0](x)
        
        # Remaining decoder blocks with skip connections
        for i in range(1, len(self.decoder_blocks)):
            # Concatenate with corresponding encoder feature
            skip_feature = encoder_features[-(i+1)]
            
            # Check if dimensions match for concatenation
            if x.shape[2:] != skip_feature.shape[2:]:
                # Resize skip feature to match decoder output if needed
                skip_feature = nn.functional.interpolate(
                    skip_feature, 
                    size=x.shape[2:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Concatenate along channel dimension
            x = torch.cat([x, skip_feature], dim=1)
            
            # Apply decoder block
            x = self.decoder_blocks[i](x)
        
        # Final upsampling to reach 256x256
        x = self.final_upsample(x)
        
        # Final convolution
        x = self.final(x)
        
        return x
