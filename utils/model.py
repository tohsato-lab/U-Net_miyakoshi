import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    Double Convolution and BN and ReLU
    (3x3 conv -> BN -> ReLU) ** 2
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """
    DoubleConvとMaxPool2dを組み合わせる.
    これを通して一つ下の層へ移る.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch=in_ch, out_ch=out_ch)
        )
    
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """
    Upsampling
    スキップコネクションを通したmapと結合させる
    """
    def __init__(self, in_ch, out_ch, bilinear=False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        """
        x1 = 下層から持ってきたfeature map
        x2 = スキップコネクションで持ってきたfeature map
        """
        x1 = self.upsample(x1)
        
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        
        x1 = F.pad(x1, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.layer1 = DoubleConv(self.in_ch, 64)
        self.layer2 = Down(64, 128)
        self.layer3 = Down(128, 256)
        self.layer4 = Down(256, 512)
        self.layer5 = Down(512, 1024)
        
        self.layer6 = Up(1024, 512, bilinear=self.bilinear)
        self.layer7 = Up(512, 256, bilinear=self.bilinear)
        self.layer8 = Up(256, 128, bilinear=self.bilinear)
        self.layer9 = Up(128, 64, bilinear=self.bilinear)
        
        self.layer10 = nn.Conv2d(64, self.out_ch, kernel_size=1)
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        x6 = self.layer6(x5, x4)
        x6 = self.layer7(x6, x3)
        x6 = self.layer8(x6, x2)
        x6 = self.layer9(x6, x1)
        
        x6 = self.layer10(x6)
        
        return x6
