import torch
from torch import nn
import torch.nn.functional as F

# 가중치 초기화 함수
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)  # Xavier Initialization
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)  # Zero Initialization
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)  # Scale = 1
        nn.init.constant_(module.bias, 0)    # Offset = 0

# Encoding Block
class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=True):
        super().__init__()
        padding = (kernel_size - 1) // 2  # To preserve spatial dimensions

        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1),
            nn.PReLU(),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers += [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1),
            nn.PReLU(),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Decoding Block
class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, upsampling=True):
        super().__init__()
        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = EncodingBlock(out_channels * 2, out_channels, batch_norm=batch_norm)

    def forward(self, x1, x2):
        x2 = self.up(x2)  # Upsample or Transpose Convolution
        x1 = F.interpolate(x1, size=x2.size()[2:], mode="bilinear", align_corners=False)  # Align spatial dimensions
        return self.conv(torch.cat([x1, x2], dim=1))

# UNet Architecture
class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Encoding
        self.enc1 = EncodingBlock(3, 64)
        self.enc2 = EncodingBlock(64, 128)
        self.enc3 = EncodingBlock(128, 256)
        self.enc4 = EncodingBlock(256, 512)

        # Bottleneck
        self.bottleneck = EncodingBlock(512, 1024)

        # Decoding
        self.dec4 = DecodingBlock(1024, 512)
        self.dec3 = DecodingBlock(512, 256)
        self.dec2 = DecodingBlock(256, 128)
        self.dec1 = DecodingBlock(128, 64)

        # Final Layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Binary Segmentation

        # Initialize Weights
        self.apply(initialize_weights)

    def forward(self, x):
        # Encoding Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoding Path
        dec4 = self.dec4(enc4, bottleneck)
        dec3 = self.dec3(enc3, dec4)
        dec2 = self.dec2(enc2, dec3)
        dec1 = self.dec1(enc1, dec2)

        # Final Output
        out = self.final(dec1)
        return self.sigmoid(out)

# UNetSmall Architecture
class UNetSmall(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Encoding
        self.enc1 = EncodingBlock(3, 32)
        self.enc2 = EncodingBlock(32, 64)
        self.enc3 = EncodingBlock(64, 128)
        self.enc4 = EncodingBlock(128, 256)

        # Bottleneck
        self.bottleneck = EncodingBlock(256, 512)

        # Decoding
        self.dec4 = DecodingBlock(512, 256)
        self.dec3 = DecodingBlock(256, 128)
        self.dec2 = DecodingBlock(128, 64)
        self.dec1 = DecodingBlock(64, 32)

        # Final Layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Binary Segmentation

        # Initialize Weights
        self.apply(initialize_weights)

    def forward(self, x):
        # Encoding Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoding Path
        dec4 = self.dec4(enc4, bottleneck)
        dec3 = self.dec3(enc3, dec4)
        dec2 = self.dec2(enc2, dec3)
        dec1 = self.dec1(enc1, dec2)

        # Final Output
        out = self.final(dec1)
        return self.sigmoid(out)

if __name__ == "__main__":
    dummy_input = torch.randn(4, 3, 224, 224)
    model = UNet()
    print(model(dummy_input).shape)

    small_model = UNetSmall()
    print(small_model(dummy_input).shape)
