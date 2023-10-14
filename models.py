"""
model implementation
"""

import torch
import torch.nn as nn


def weights_init(m):
    """
    initialize weights of the model
    authors of the paper recommend normal distribution with mean=0 and std=0.02
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(
            self,
            dim_channel: int=3,
            dim_latent: int=128,
            dim_feature: int=64
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dim_latent,
                out_channels=dim_feature * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=dim_feature * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=dim_feature * 8,
                out_channels=dim_feature * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=dim_feature * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=dim_feature * 4,
                out_channels=dim_feature * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=dim_feature * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=dim_feature * 2,
                out_channels=dim_feature,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=dim_feature),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=dim_feature,
                out_channels=dim_channel,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(
            self,
            dim_channel: int=3,
            dim_feature: int=64
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_channel,
                out_channels=dim_feature,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=dim_feature,
                out_channels=dim_feature * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=dim_feature * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=dim_feature * 2,
                out_channels=dim_feature * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=dim_feature * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=dim_feature * 4,
                out_channels=dim_feature * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=dim_feature * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=dim_feature * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

