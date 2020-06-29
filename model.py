import torch.nn.functional as F
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_boundary(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear= False):
        super(Up_boundary, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1,1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_edg, x):
        x_up = self.up(x_edg)

        cat_x = torch.cat((x_up, x), 1)
        output = self.conv(cat_x)
        return output


class Up_mask(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up_mask, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1,1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_mask, x):
        x_up = self.up(x_mask)

        cat_x = torch.cat((x_up, x), 1)
        output = self.conv(cat_x)
        return output



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



'''
define DGFNet
'''
class DGFNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear= False, n_features = 2):
        super(DGFNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder include two branch
        self.b_up1 = Up_boundary(1024, 512, bilinear)
        self.b_up2 = Up_boundary(512, 256, bilinear)
        self.b_up3 = Up_boundary(256, 128, bilinear)
        self.b_up4 = Up_boundary(128, 64, bilinear)
        self.b_out = OutConv(64, n_classes)

        self.m_up1 = Up_mask(1024, 512, bilinear)
        self.m_up2 = Up_mask(512, 256, bilinear)
        self.m_up3 = Up_mask(256, 128, bilinear)
        self.m_up4 = Up_mask(128, 64, bilinear)
        self.m_out = OutConv(64, n_classes)

        self.branch_1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(5, 1, kernel_size=1)


    def forward(self, x):

        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder

        # guidance mask
        o_m_4 = self.m_up1(x5, x4)
        o_m_3 = self.m_up2(o_m_4, x3)
        o_m_2 = self.m_up3(o_m_3, x2)
        o_m_1 = self.m_up4(o_m_2, x1)
        o_m = self.m_out(o_m_1)

        # guidance edge
        o_b_4 = self.b_up1(x5, x4)
        o_b_3 = self.b_up2(o_b_4, x3)
        o_b_2 = self.b_up3(o_b_3, x2)
        o_b_1 = self.b_up4(o_b_2, x1)
        o_b = self.b_out(o_b_1)

        seg_in = torch.cat((o_b, o_m), dim=1)
        o_branch_1 = self.branch_1(seg_in)
        o_branch_2 = self.branch_2(seg_in)
        o_branch_3 = self.branch_3(seg_in)
        seg_out = self.output(torch.cat((o_branch_1, o_branch_2, o_branch_3, o_b, o_m), dim=1))

        if self.n_classes > 1:
            boundary = F.softmax(o_b, dim=1)
            seg = F.softmax(seg_out, dim=1)
            mask = F.softmax(o_m, dim=1)
            return boundary, mask, seg
        else:
            boundary = torch.sigmoid(o_b)
            mask = torch.sigmoid(o_m)
            seg = torch.sigmoid(seg_out)
            return boundary, mask, seg


