import torch.nn as nn
import torch
import torchvision.models as models


class D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        return self.layer(x)


class Discriminator(nn.Module):
    def __init__(self, img_size, in_channels=3):
        super().__init__()

        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1), nn.LeakyReLU()
        )

        self.block_1_1 = D_Block(64, 64, stride=2)  # stride= 2 if output 4x
        self.block_1_2 = D_Block(64, 128, stride=1)
        self.block_1_3 = D_Block(128, 128)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1), nn.LeakyReLU()
        )

        self.block_2_2 = D_Block(64, 128, stride=1)

        self.block3 = D_Block(256, 256, stride=1)
        self.block4 = D_Block(256, 256)
        self.block5 = D_Block(256, 512, stride=1)
        self.block6 = D_Block(512, 512)
        self.block7 = D_Block(512, 1024)
        self.block8 = D_Block(1024, 1024)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1024 * img_size[0] * img_size[1] // 256, 100) # Change based on input image size
        self.fc2 = nn.Linear(100, 2)

        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

        x_1 = self.block_1_3(self.block_1_2(self.block_1_1(self.conv_1_1(x1))))
        x_2 = self.block_2_2(self.conv_2_1(x2))

        x = torch.cat([x_1, x_2], dim=1)
        x = self.block8(
            self.block7(self.block6(self.block5(self.block4(self.block3(x)))))
        )

        x = self.flatten(x)

        print(x.shape)

        x = self.fc1(x)
        x = self.fc2(self.relu(x))

        return self.sigmoid(x)
