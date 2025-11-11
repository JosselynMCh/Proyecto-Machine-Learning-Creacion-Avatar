import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        if self.shortcut:
            x = self.shortcut(x)
        out += x
        return F.relu(out)

class CAM(nn.Module):
    def __init__(self, in_channels):
        super(CAM, self).__init__()
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = torch.mean(x, dim=[2, 3])
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        return x * torch.sigmoid(avg_out)

class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        max_out, _ = torch.max(x, dim=[2, 3])
        max_out = self.fc(max_out).view(b, c, 1, 1)
        return x * torch.sigmoid(max_out)

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.cam = CAM(in_channels)
        self.sam = SAM(in_channels)

    def forward(self, x):
        x = self.cam(x) + self.sam(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.final = nn.Conv2d(64, 3, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 256)
        )
        self.final = nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_blocks(x)
        return self.final(x)

class UGatIT(nn.Module):
    def __init__(self):
        super(UGatIT, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        fake_image = self.generator(x)
        disc_output = self.discriminator(fake_image)
        return fake_image, disc_output
