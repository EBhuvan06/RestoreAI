import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)
        self.sg2 = SimpleGate()

        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = inp + x * self.beta

        inp = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = inp + x * self.gamma
        return x


class KLARestorer(nn.Module):
    def __init__(self, width=32, num_blocks=[2, 2, 4, 8]):
        super().__init__()
        self.intro = nn.Conv2d(1, width, 3, padding=1)

        self.enc1 = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(width, width*2, 2, stride=2)

        self.enc2 = nn.Sequential(*[NAFBlock(width*2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(width*2, width*4, 2, stride=2)

        self.enc3 = nn.Sequential(*[NAFBlock(width*4) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(width*4, width*8, 2, stride=2)

        self.middle = nn.Sequential(*[NAFBlock(width*8) for _ in range(num_blocks[3])])

        self.up3 = nn.Sequential(
            nn.Conv2d(width*8, width*16, 1),
            nn.PixelShuffle(2)
        )
        self.dec3 = nn.Sequential(*[NAFBlock(width*4) for _ in range(num_blocks[2])])

        self.up2 = nn.Sequential(
            nn.Conv2d(width*4, width*8, 1),
            nn.PixelShuffle(2)
        )
        self.dec2 = nn.Sequential(*[NAFBlock(width*2) for _ in range(num_blocks[1])])

        self.up1 = nn.Sequential(
            nn.Conv2d(width*2, width*4, 1),
            nn.PixelShuffle(2)
        )
        self.dec1 = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks[0])])

        self.final_up = nn.Sequential(
            nn.Conv2d(width, width*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1)
        )

        self.skip_conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        inp = x

        x = self.intro(x)

        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        x = self.middle(x)

        x = self.up3(x) + e3
        x = self.dec3(x)

        x = self.up2(x) + e2
        x = self.dec2(x)

        x = self.up1(x) + e1
        x = self.dec1(x)

        x = self.final_up(x)

        inp_up = F.interpolate(inp, scale_factor=2, mode='bicubic', align_corners=False)
        x = x + self.skip_conv(inp_up)

        return x.clamp(0, 1)
