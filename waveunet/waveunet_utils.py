import torch.nn as nn


# Based on https://arxiv.org/abs/1603.05027
# Order of conv, bn, act can be  modified.
class ResidualConvUnit(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride=1, padding=7):
        super(ResidualConvUnit, self).__init__()
        self.conv1d = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(n_outputs)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        out = self.conv1d(x)
        out = self.bn(out)
        out = self.act(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=15, stride=1, padding=7):
        super(ResidualBlock, self).__init__()
        self.conv_in = ResidualConvUnit(n_inputs, n_outputs // 2, 1, 1, 0)
        self.conv_mid = ResidualConvUnit(n_outputs // 2, n_outputs // 2, kernel_size, stride, padding)
        self.conv_out = nn.Conv1d(n_outputs // 2, n_outputs, 1, 1, 0)
        self.bn = nn.BatchNorm1d(n_outputs)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.conv_mid(out)
        out = self.conv_out(out) + x
        out = self.bn(out)
        out = self.act(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, transpose=False):
        super(ConvBlock, self).__init__()
        if transpose:
            self.conv1d = nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size=kernel_size,
                                             stride=stride, padding=padding)
        else:
            self.conv1d = nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(n_outputs)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        out = self.conv1d(x)
        out = self.bn(out)
        out = self.act(out)

        return out


class DownBlock(nn.Module):
    def __init__(self, n_inputs, n_skips, n_outputs, kernel_size=16, stride=2, padding=7):
        super(DownBlock, self).__init__()
        self.pre_skip_conv = ConvBlock(n_inputs, n_skips, 15, 1, padding)
        self.post_skip_conv = ResidualBlock(n_skips, n_skips, 15, 1, padding)
        self.down_conv = ConvBlock(n_skips, n_outputs, kernel_size, stride, padding)

    def forward(self, x):
        skip = self.pre_skip_conv(x)
        out = self.post_skip_conv(skip)
        out = self.down_conv(out)

        return out, skip


class UpBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=16, stride=2, padding=7):
        super(UpBlock, self).__init__()
        self.up_conv = ConvBlock(n_inputs, n_outputs, kernel_size, stride, padding, transpose=True)
        self.skip_conv = ResidualBlock(n_outputs, n_outputs, 15, 1, padding)
        self.out_conv = ConvBlock(n_outputs, n_outputs, 15, 1, padding)

    def forward(self, x, skip):
        out = self.up_conv(x)
        out = self.skip_conv(out + skip)
        out = self.out_conv(out)

        return out


class BottleneckFC(nn.Module):
    def __init__(self, n_inputs):
        super(BottleneckFC, self).__init__()
        self.linear_in = nn.Linear(n_inputs, n_inputs // 2)
        self.linear_mid = nn.Linear(n_inputs // 2, n_inputs // 2)
        self.linear_out = nn.Linear(n_inputs // 2, n_inputs)
        self.act = nn.ReLU()
        self.bn_in = nn.BatchNorm1d(n_inputs // 2)
        self.bn_mid = nn.BatchNorm1d(n_inputs // 2)
        self.bn_out = nn.BatchNorm1d(n_inputs)

    def forward(self, x):
        out = self.linear_in(x.permute(0, 2, 1))  # B, T, C
        out = self.bn_in(out.permute(0, 2, 1))  # B, C, T
        out = self.act(out)
        out = self.linear_mid(out.permute(0, 2, 1))  # B, T, C
        out = self.bn_mid(out.permute(0, 2, 1))  # B, C, T
        out = self.act(out)
        out = self.linear_out(out.permute(0, 2, 1)) + x.permute(0, 2, 1)  # B, T, C
        out = self.bn_out(out.permute(0, 2, 1))  # B, C, T

        return out
