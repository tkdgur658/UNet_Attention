#Noori, Mehrdad, Ali Bahri, and Karim Mohammadi. "Attention-guided version of 2D UNet for automatic brain tumor segmentation." 2019 9th International Conference on Computer and Knowledge Engineering (ICCKE). IEEE, 2019.
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        shape = 0
        
        if channels == 512:
            shape = 128
        elif channels == 256:
            shape = 256
        elif channels == 128:
            shape = 512
        else:
            assert shape == 0, "shape is 0"
        self.gap = nn.AvgPool2d(shape, stride=1)
        r = int(channels / reduction)
        self.fc1 = nn.Linear(channels, r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(r, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        out = x.view(x.size(0), x.size(1), 1, 1)
        return out


class UNet_Attention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, device = torch.device('cuda')):
        super(UNet_Attention, self).__init__()
        features = init_features
        self.device = device
        self.stem = nn.Conv2d(in_channels=in_channels,
                              out_channels=features, kernel_size=2, padding=0)
        self.encoder1 = UNet_Attention._block(features, features, name="enc1")

        self.downconv1 = nn.Conv2d(
            in_channels=features, out_channels=features * 2, kernel_size=2, stride=2, padding=0)
        self.encoder2 = UNet_Attention._block(features * 2, features * 2, name="enc2")

        self.downconv2 = nn.Conv2d(
            in_channels=features * 2, out_channels=features * 4, kernel_size=2, stride=2, padding=0)
        self.encoder3 = UNet_Attention._block(features * 4, features * 4, name="enc3")

        self.downconv3 = nn.Conv2d(
            in_channels=features * 4, out_channels=features * 8, kernel_size=2, stride=2, padding=0)
        self.encoder4 = UNet_Attention._block(features * 8, features * 8, name="enc4")

        self.upconv4 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2)
        self.seblock3 = SEBlock(512)
        self.decoder3 = UNet_Attention._block(features * 8, features * 4, name="dec3")
        self.oneconv3 = nn.Conv2d(
            features * 8, features * 4, kernel_size=1, padding=0, bias=False)

        self.upconv3 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2)
        self.seblock2 = SEBlock(256)
        self.decoder2 = UNet_Attention._block(features * 4, features * 2, name="dec2")
        self.oneconv2 = nn.Conv2d(
            features * 4, features * 2, kernel_size=1, padding=0, bias=False)

        self.upconv2 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2)
        self.seblock1 = SEBlock(128)
        self.decoder1 = UNet_Attention._block(features * 2, features, name="dec1")
        self.oneconv1 = nn.Conv2d(
            features * 2, features, kernel_size=1, padding=0, bias=False)

        self.bn = nn.BatchNorm2d(num_features=features)
        self.prelu = nn.PReLU()

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        if self.training== True : 
            x = self.gaussian_noise(x)
        
        stem = self.stem(x)
        stem = F.pad(stem, (0, 1, 0, 1))
        enc1 = self.encoder1(stem)
        enc1 += stem  # layer 1 (concat)

        down1 = self.downconv1(enc1)
        enc2 = self.encoder2(down1)
        enc2 += down1  # layer 2

        down2 = self.downconv2(enc2)
        enc3 = self.encoder3(down2)
        enc3 += down2  # layer 3

        down3 = self.downconv3(enc3)
        enc4 = self.encoder4(down3)
        enc4 += down3
 ###########################################################

        dec4 = self.upconv4(enc4)
        dec4 = torch.cat((dec4, enc3), dim=1)
        up3 = self.seblock3(dec4)
        up3 = up3 * dec4

        dec3 = self.decoder3(up3)
        up3 = self.oneconv3(up3)
        dec3 += up3
###########################################################

        dec3 = self.upconv3(dec3)
        dec3 = torch.cat((dec3, enc2), dim=1)
        up2 = self.seblock2(dec3)
        up2 = up2 * dec3

        dec2 = self.decoder2(up2)
        up2 = self.oneconv2(up2)
        dec2 += up2
###########################################################

        dec2 = self.upconv2(dec2)
        dec2 = torch.cat((dec2, enc1), dim=1)
        
        up1 = self.seblock1(dec2)
        up1 = up1 * dec2

        dec1 = self.decoder1(up1)
        up1 = self.oneconv1(up1)
        dec1 += up1
###########################################################

        dec1 = self.bn(dec1)
        dec1 = self.prelu(dec1)

        return self.conv(dec1)

    def gaussian_noise(self, inputs, mean=0, stddev=0.01):
        noise = torch.randn(inputs.shape)*stddev+mean
        noise = noise.to(self.device)
        out = torch.add(inputs, noise)        
        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "norm1", nn.BatchNorm2d(num_features=in_channels)),
                    (name + "prelu1", nn.PReLU()),
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "prelu2", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                ]
            )
        )
# device = torch.device("cuda")
# devices =  [0,1,2,3]
# model = UNet_Attention(1,1, device=device)
# model = torch.nn.DataParallel(model, device_ids = devices ).cuda()