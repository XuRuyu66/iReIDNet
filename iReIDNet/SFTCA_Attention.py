#####SFT+ CA  ###############
import torch
from torch import nn
import torch.nn.functional as F
class SFTCA(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(SFTCA, self).__init__()

        self.h = h
        self.w = w
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
        self.SFT_shift_conv0 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_shift_conv1 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_scale_conv0 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_scale_conv1 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1,
                                         stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        avg_pool = self.avg_pool(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(avg_pool), 0.1, inplace=True))
        scale = torch.sigmoid(scale)
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x), 0.1, inplace=True))
        shift = (x + shift) * scale

        out = (x * s_h.expand_as(x) * s_w.expand_as(x)) * shift
        return out